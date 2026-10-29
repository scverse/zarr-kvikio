from __future__ import annotations

import pickle
import warnings
from contextlib import contextmanager
from importlib.util import find_spec
from typing import TYPE_CHECKING

import numcodecs
import numpy as np
import pytest
import zarr
from zarr.codecs import (
    BloscCodec,
    BytesCodec,
    Crc32cCodec,
    GzipCodec,
    ShardingCodec,
    ZstdCodec,
)
from zarr.core.buffer import cpu, gpu
from zarr.core.codec_pipeline import FusedCodecPipeline

import zarr_kvikio
from zarr_kvikio import _pipeline

cp = pytest.importorskip("cupy")

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    not (zarr_kvikio.available() and cp.is_available()), reason="needs KvikIO, nvCOMP and a GPU"
)

PIPELINE = "zarr_kvikio.KvikioCodecPipeline"
ZSTD = (ZstdCodec(level=3),)


@contextmanager
def gpu_pipeline() -> Generator[None]:
    with zarr_kvikio.enable():
        yield


@pytest.fixture(autouse=True, params=["gpu", "host"])
def read_path(request: pytest.FixtureRequest) -> str:
    """Run every test through both read paths: KvikIO + nvCOMP, and host reads + CPU decoding
    (which is used for reads smaller than `kvikio.min_bytes`)."""
    zarr.config.set({"kvikio.min_bytes": 0 if request.param == "gpu" else 1 << 40})
    return request.param


@pytest.fixture
def fused_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records reads delegated to zarr's fused pipeline, by where their output is."""
    calls: list[str] = []
    orig = FusedCodecPipeline.read

    async def spy(self, batch_info, out, drop_axes=()):
        calls.append("host" if isinstance(out.as_ndarray_like(), np.ndarray) else "gpu")
        return await orig(self, batch_info, out, drop_axes)

    monkeypatch.setattr(FusedCodecPipeline, "read", spy)
    return calls


@pytest.fixture
def fast_path_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records reads that zarr-kvikio handles itself (instead of zarr's own pipeline)."""
    calls: list[str] = []
    for name in ["_read_gpu", "_read_host", "_read_gpu_contiguous"]:
        orig = getattr(_pipeline, name)

        def spy(*args, _orig=orig, _name=name, **kwargs):
            calls.append(_name)
            return _orig(*args, **kwargs)

        monkeypatch.setattr(_pipeline, name, spy)
    return calls


def write(
    path: Path,
    data: np.ndarray,
    chunks: tuple[int, ...],
    *,
    shards: tuple[int, ...] | None = None,
    compressors=ZSTD,
    fill_value=0,
    **kwargs,
) -> zarr.Array:
    arr = zarr.create_array(
        path,
        shape=data.shape,
        dtype=data.dtype,
        chunks=chunks,
        shards=shards,
        compressors=compressors,
        fill_value=fill_value,
        **kwargs,
    )
    arr[...] = data
    return arr


def read_gpu(path: Path, selection, *, how: str = "getitem") -> np.ndarray:
    with gpu_pipeline():
        arr = zarr.open_array(path, mode="r")
        assert type(arr._async_array.codec_pipeline).__name__ == "KvikioCodecPipeline"
        res = getattr(arr, how)[selection] if how != "getitem" else arr[selection]
    assert isinstance(res, cp.ndarray)
    return cp.asnumpy(res)


def read_cpu(path: Path, selection, *, how: str = "getitem") -> np.ndarray:
    arr = zarr.open_array(path, mode="r")
    return getattr(arr, how)[selection] if how != "getitem" else arr[selection]


def check(path: Path, selection, *, how: str = "getitem") -> None:
    np.testing.assert_array_equal(
        read_gpu(path, selection, how=how), read_cpu(path, selection, how=how)
    )


rng = np.random.default_rng(0)

SELECTIONS_1D = [
    slice(None),
    slice(3, 997),
    slice(250, 750),  # whole inner chunks in the middle
    slice(5, 900, 7),
    17,
    -1,
]


@pytest.mark.parametrize("dtype", ["float32", "int64", "uint8", "float64"])
@pytest.mark.parametrize("compressors", [(ZstdCodec(level=3),), ()], ids=["zstd", "raw"])
@pytest.mark.parametrize("shards", [None, (500,)], ids=["unsharded", "sharded"])
@pytest.mark.parametrize("selection", SELECTIONS_1D, ids=repr)
def test_1d(tmp_path, fast_path_calls, *, dtype, compressors, shards, selection):
    data = (rng.random(1003) * 100).astype(dtype)
    write(tmp_path / "a.zarr", data, (50,), shards=shards, compressors=compressors)
    check(tmp_path / "a.zarr", selection)
    assert fast_path_calls


@pytest.mark.parametrize("shards", [None, (20, 30, 8)], ids=["unsharded", "sharded"])
@pytest.mark.parametrize(
    ("selection", "how"),
    [
        (slice(None), "getitem"),
        ((slice(3, 37), slice(None), 5), "getitem"),
        ((slice(None, None, 3), slice(4, 50, 5), slice(None)), "getitem"),
        ((7, 11, slice(None)), "getitem"),
        (([0, 3, 21, 39], slice(None), [1, 6]), "oindex"),
        ((np.arange(40) % 3 == 0, slice(2, 9), slice(None)), "oindex"),
        pytest.param(
            ([0, 3, 21, 39], [1, 59, 2, 30], [7, 0, 3, 5]),
            "vindex",
            marks=pytest.mark.xfail(
                reason="zarr's vindex calls np.array on the output, so it fails for GPU buffers "
                "with any pipeline",
                raises=TypeError,
                strict=True,
            ),
        ),
        ((slice(1, 2), slice(0, 1), slice(None)), "blocks"),
    ],
    ids=["all", "slices", "steps", "ints", "orthogonal", "mask", "coords", "blocks"],
)
def test_nd(tmp_path, fast_path_calls, shards, selection, how):
    data = rng.random((40, 60, 8)).astype("float32")
    write(tmp_path / "a.zarr", data, (10, 15, 4), shards=shards)
    check(tmp_path / "a.zarr", selection, how=how)
    assert fast_path_calls


@pytest.mark.parametrize("index_location", ["start", "end"])
@pytest.mark.parametrize(
    "index_codecs",
    [(BytesCodec(),), (BytesCodec(), Crc32cCodec())],
    ids=["plain", "crc"],
)
def test_shard_index_variants(tmp_path, fast_path_calls, index_location, index_codecs):
    data = rng.random((64, 64)).astype("float32")
    arr = zarr.create_array(
        tmp_path / "a.zarr",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(32, 32),
        serializer=ShardingCodec(
            chunk_shape=(8, 8),
            codecs=[BytesCodec(), ZstdCodec(level=1)],
            index_codecs=index_codecs,
            index_location=index_location,
        ),
        compressors=None,
    )
    arr[...] = data
    check(tmp_path / "a.zarr", (slice(5, 60), slice(None)))
    assert fast_path_calls


@pytest.mark.parametrize("shards", [None, (100,)], ids=["unsharded", "sharded"])
def test_missing_chunks(tmp_path, fast_path_calls, shards):
    data = np.full(400, 7, dtype="float32")
    data[:25] = np.arange(25)  # all other (inner) chunks equal the fill value and are not written
    write(tmp_path / "a.zarr", data, (20,), shards=shards, fill_value=7)
    check(tmp_path / "a.zarr", slice(None))
    check(tmp_path / "a.zarr", slice(150, 390))  # only missing chunks / an empty shard
    assert fast_path_calls


def test_corrupt_index(tmp_path):
    data = rng.random(100).astype("float32")
    write(tmp_path / "a.zarr", data, (10,), shards=(50,))
    shard = tmp_path / "a.zarr" / "c" / "0"
    raw = bytearray(shard.read_bytes())
    raw[-10] ^= 0xFF
    shard.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="checksum"):
        read_gpu(tmp_path / "a.zarr", slice(None))


@pytest.mark.parametrize(
    "compressors",
    [(GzipCodec(),), (BloscCodec(),), (ZstdCodec(level=3, checksum=True),)],
    ids=["gzip", "blosc", "zstd-checksum"],
)
def test_unsupported_codecs_fall_back(tmp_path, fast_path_calls, fused_reads, compressors):
    data = rng.random(300).astype("float32")
    write(tmp_path / "a.zarr", data, (50,), compressors=compressors)
    check(tmp_path / "a.zarr", slice(10, 290))
    assert not fast_path_calls
    # decoded on the host by zarr's fused pipeline, then copied to the GPU
    assert fused_reads
    assert set(fused_reads) == {"host"}


@pytest.mark.parametrize("dtype", ["float32", "V4", [("a", "u2"), ("b", "i2")]], ids=repr)
def test_zarr_v2_falls_back(tmp_path, fast_path_calls, fused_reads, dtype):
    """Like most existing AnnData stores: zarr v2, blosc-compressed (also of other data types)."""
    data = rng.random((40, 30)).astype("float32").view(dtype)
    arr = zarr.create_array(
        tmp_path / "a.zarr",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(10, 30),
        zarr_format=2,
        compressors=numcodecs.Blosc(),
    )
    arr[...] = data
    check(tmp_path / "a.zarr", (slice(5, 37), slice(None)))
    assert not fast_path_calls
    assert fused_reads


def test_host_buffers_fall_back(tmp_path, fast_path_calls, fused_reads):
    data = rng.random(300).astype("float32")
    write(tmp_path / "a.zarr", data, (50,), shards=(150,))
    with zarr.config.set({"codec_pipeline.path": PIPELINE}):
        res = zarr.open_array(tmp_path / "a.zarr", mode="r")[:]
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, data)
    assert not fast_path_calls
    assert fused_reads


HOST_PIPELINES = [  # (set, or the pipeline zarr was configured with before `enable()`)
    pytest.param({}, id="default"),
    pytest.param({"kvikio.host_pipeline": PIPELINE}, id="own"),  # (zarr's fused one)
    pytest.param(
        {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"},
        id="zarrs",
        marks=pytest.mark.skipif(find_spec("zarrs") is None, reason="needs zarrs"),
    ),
    pytest.param(
        {"kvikio.host_pipeline": "zarrs.ZarrsCodecPipeline"},
        id="zarrs-set",
        marks=pytest.mark.skipif(find_spec("zarrs") is None, reason="needs zarrs"),
    ),
]


@pytest.mark.parametrize("host_pipeline", HOST_PIPELINES)  # (zarrs doesn't cast)
@pytest.mark.parametrize(
    ("dtype", "out_dtype"),
    [("float32", "float64"), ("float64", "float32"), ("datetime64[s]", "int64")],
)
def test_other_out_dtype_falls_back(tmp_path, fast_path_calls, host_pipeline, dtype, out_dtype):
    data = np.arange(300).astype(dtype)
    write(tmp_path / "a.zarr", data, (50,), shards=(150,))
    with zarr.config.set(host_pipeline), gpu_pipeline():
        arr = zarr.open_array(tmp_path / "a.zarr", mode="r")
        for sel in [slice(None), slice(None, None, 2)]:  # (zarrs reads steps with zarr's pipeline)
            out = cp.zeros(len(data[sel]), dtype=out_dtype)
            arr.get_basic_selection(sel, out=gpu.NDBuffer(out))
            np.testing.assert_array_equal(cp.asnumpy(out), data[sel].astype(out_dtype))
    assert not fast_path_calls


@pytest.mark.parametrize("host_pipeline", HOST_PIPELINES)
@pytest.mark.parametrize("shards", [None, (150,)])
def test_memory_store_falls_back(fast_path_calls, host_pipeline, shards):
    data = rng.random(300).astype("float32")
    store = zarr.storage.MemoryStore()
    with zarr.config.set(host_pipeline), gpu_pipeline():
        arr = zarr.create_array(
            store, shape=data.shape, dtype=data.dtype, chunks=(50,), shards=shards
        )
        arr[...] = cp.asarray(data)
        res = arr[:]
    np.testing.assert_array_equal(cp.asnumpy(res), data)
    assert not fast_path_calls


def test_write_and_read_roundtrip(tmp_path, fast_path_calls):
    data = rng.random((30, 30)).astype("float32")
    with gpu_pipeline():
        arr = zarr.create_array(
            tmp_path / "a.zarr",
            shape=data.shape,
            dtype=data.dtype,
            chunks=(10, 10),
            shards=(30, 30),
        )
        arr[...] = cp.asarray(data)
        res = zarr.open_array(tmp_path / "a.zarr", mode="r")[:]
    np.testing.assert_array_equal(cp.asnumpy(res), data)
    assert fast_path_calls


def test_pickle_keeps_pipeline(tmp_path):
    data = rng.random(300).astype("float32")
    write(tmp_path / "a.zarr", data, (50,), shards=(150,))
    with gpu_pipeline():
        arr = zarr.open_array(tmp_path / "a.zarr", mode="r")
        arr.get_basic_selection(slice(1), prototype=cpu.buffer_prototype)  # (on the host)
        assert "_host" in arr._async_array.codec_pipeline.__dict__
        arr = pickle.loads(pickle.dumps(arr))
        pipeline = arr._async_array.codec_pipeline
        assert type(pipeline).__name__ == "KvikioCodecPipeline"
        # (made again from the zarr config of the process that unpickles it, e.g. a Dask worker)
        assert "_host" not in pipeline.__dict__
        np.testing.assert_array_equal(cp.asnumpy(arr[:]), data)


@pytest.mark.parametrize("shards", [None, (200,)], ids=["unsharded", "sharded"])
def test_decode_in_several_calls(tmp_path, monkeypatch, fast_path_calls, read_path, shards):
    import nvidia.nvcomp as nv

    zarr.config.set({"kvikio.max_decode_bytes": 300})  # < 2 chunks of 40 float32 per call
    as_arrays, frames = nv.as_arrays, []  # (frames per call, and as many outputs)
    monkeypatch.setattr(nv, "as_arrays", lambda a: frames.append(len(a)) or as_arrays(a))
    data = rng.random(1000).astype("float32")
    write(tmp_path / "a.zarr", data, (40,), shards=shards)
    check(tmp_path / "a.zarr", slice(7, 993))
    assert fast_path_calls
    if read_path == "gpu":
        assert len(frames) > 2
        assert set(frames) == {1}


def test_release_memory(tmp_path, read_path):
    if read_path == "host":
        pytest.skip("host reads hold no nvCOMP codecs")
    data = rng.random(10_000).astype("float32")
    write(tmp_path / "a.zarr", data, (1000,), shards=(5000,))
    check(tmp_path / "a.zarr", slice(None))
    assert _pipeline._pools
    zarr_kvikio.release_memory()
    assert not _pipeline._pools
    check(tmp_path / "a.zarr", slice(None))  # a new codec is created on demand


def test_enable(tmp_path, fast_path_calls):
    data = rng.random(1000).astype("float32")
    write(tmp_path / "a.zarr", data, (100,), shards=(500,))
    before = zarr.config.get("codec_pipeline.path"), zarr.config.get("ndbuffer")
    with zarr_kvikio.enable():
        res = zarr.open_array(tmp_path / "a.zarr", mode="r")[:]
    assert isinstance(res, cp.ndarray)
    np.testing.assert_array_equal(cp.asnumpy(res), data)
    assert fast_path_calls
    assert (
        zarr.config.get("codec_pipeline.path"),
        zarr.config.get("ndbuffer"),
    ) == before


def test_dask_enable(tmp_path, read_path):
    distributed = pytest.importorskip("distributed")
    da = pytest.importorskip("dask.array")
    import zarr_kvikio.dask

    data = rng.random(1000).astype("float32")
    write(tmp_path / "a.zarr", data, (100,), shards=(500,))
    with (
        distributed.LocalCluster(
            n_workers=1, threads_per_worker=2, processes=True, dashboard_address=":0"
        ) as cluster,
        distributed.Client(cluster) as client,
        zarr.config.set({"kvikio.num_threads": 2, "kvikio.fetch": "kvikio"}),  # (uses KvikIO)
        zarr_kvikio.dask.enable(client),
    ):
        # arrays opened here are read on the workers, into GPU memory
        x = da.from_zarr(zarr.open_array(tmp_path / "a.zarr", mode="r"), chunks=250)
        # check on the workers (moving GPU arrays between processes needs extra dependencies)
        on_gpu = x.map_blocks(lambda b: np.array([isinstance(b, cp.ndarray)]), chunks=(1,))
        assert on_gpu.compute().all()
        np.testing.assert_array_equal(x.map_blocks(cp.asnumpy).compute(), data)
        # while this process keeps reading into host memory
        assert isinstance(zarr.open_array(tmp_path / "a.zarr", mode="r")[:], np.ndarray)
        workers = client.run(lambda: zarr.config.get("codec_pipeline.path"))
        assert set(workers.values()) == {PIPELINE}
        if read_path == "gpu":  # this process's settings, used on the workers
            threads = client.run(lambda: __import__("kvikio").defaults.get("num_threads"))
            assert set(threads.values()) == {2}


@pytest.mark.parametrize("read_path", ["gpu"], indirect=True)  # (sets the threshold itself)
@pytest.mark.parametrize(
    ("selection", "path"),
    [
        pytest.param(slice(1000, 5000), "_read_host", id="small"),  # 16 KB
        # a few values from all 64 chunks (4 MiB) / from 8 of them (512 KiB)
        pytest.param(np.arange(0, 1 << 20, 4099), "_read_gpu", id="many-chunks"),
        pytest.param(np.arange(0, 1 << 20, 1 << 17), "_read_host", id="few-chunks"),
        pytest.param(slice(None), "_read_gpu_contiguous", id="large"),  # 4 MiB
    ],
)
def test_small_reads_decoded_on_host(tmp_path, fast_path_calls, selection, path):
    """Unless they decode many bytes from many chunks."""
    zarr.config.set({"kvikio.min_bytes": 1 << 20})
    data = rng.random(1 << 20).astype("float32")
    write(tmp_path / "a.zarr", data, (1 << 14,), shards=(1 << 18,))
    with warnings.catch_warnings():
        # not through zarr's own pipeline, which warns about copying each chunk to the GPU
        warnings.simplefilter("error")
        check(tmp_path / "a.zarr", (selection,), how="oindex")
    assert fast_path_calls == [path]


@pytest.mark.parametrize("shards", [None, (500, 20)], ids=["unsharded", "sharded"])
def test_thin_reads_decode_in_parts(tmp_path, monkeypatch, read_path, shards):
    """A column touches every chunk: decoded a few at a time, not all at once."""
    zarr.config.set({"kvikio.max_decode_bytes": 3 * 100 * 20 * 4})  # (3 chunks)
    fetch, fetches = _pipeline._fetch, []
    monkeypatch.setattr(_pipeline, "_fetch", lambda pieces: fetches.append(1) or fetch(pieces))
    data = rng.random((1000, 20)).astype("float32")
    write(tmp_path / "a.zarr", data, (100, 20), shards=shards)
    check(tmp_path / "a.zarr", (slice(None), 3))
    if read_path == "gpu":  # (10 chunks, or 2 shards of 5)
        assert 1 < len(fetches) <= (10 if shards is None else 2)


@pytest.mark.parametrize("sharded", [False, True], ids=["unsharded", "sharded"])
@pytest.mark.parametrize("values", [500, 2000], ids=["short", "long"])
def test_wrong_size_chunks_raise(tmp_path, values, sharded):
    """Of a corrupt store: instead of returning what the memory held."""
    import google_crc32c
    from numcodecs import Zstd

    shards = (2000,) if sharded else None
    write(tmp_path / "a.zarr", rng.random(4000).astype("float32"), (1000,), shards=shards)
    wrong = Zstd().encode(np.zeros(values, "float32"))
    if sharded:  # (inner chunks of 1000 values; the index at the end, with a checksum)
        right = Zstd().encode(np.zeros(1000, "float32"))
        index = np.array([[0, len(wrong)], [len(wrong), len(right)]], "<u8").tobytes()
        wrong += right + index + google_crc32c.value(index).to_bytes(4, "little")
    (tmp_path / "a.zarr" / "c" / "0").write_bytes(wrong)
    with pytest.raises(ValueError, match=f"{4 * values}") as error:
        read_gpu(tmp_path / "a.zarr", slice(0, 2000))
    # nothing of nvCOMP's lives on with the error: freed after `release_memory`, it crashes
    tb = error.tb
    while tb is not None:
        held = [type(v) for v in tb.tb_frame.f_locals.values()]
        assert not [t for t in held if t.__module__.startswith("nvidia.nvcomp")], held
        tb = tb.tb_next


def test_large_chunks_fall_back(tmp_path, monkeypatch, fast_path_calls, fused_reads):
    """nvCOMP decodes zstd frames of less than 2 GiB."""
    monkeypatch.setattr(_pipeline, "_NVCOMP_MAX_FRAME", 399)  # (chunks of 400 bytes)
    write(tmp_path / "a.zarr", rng.random(1000).astype("float32"), (100,), shards=(500,))
    check(tmp_path / "a.zarr", slice(None))
    assert not fast_path_calls
    assert fused_reads


@pytest.mark.parametrize("decoders", [1, 3])
def test_concurrent_reads(tmp_path, decoders):
    from concurrent.futures import ThreadPoolExecutor

    zarr.config.set({"kvikio.decoders": decoders})
    zarr_kvikio.release_memory()
    data = rng.random(200_000).astype("float32")
    write(tmp_path / "a.zarr", data, (4096,), shards=(4096 * 8,))
    with gpu_pipeline():
        arr = zarr.open_array(tmp_path / "a.zarr", mode="r")
        sels = [slice(i * 7919, i * 7919 + 20_000) for i in range(16)]
        with ThreadPoolExecutor(8) as ex:
            results = list(ex.map(lambda s: cp.asnumpy(arr[s]), sels))
    for s, r in zip(sels, results, strict=True):
        np.testing.assert_array_equal(r, data[s])
    zarr_kvikio.release_memory()


@pytest.mark.parametrize("fetch", ["host", "kvikio"])
@pytest.mark.parametrize(
    "selection",
    [slice(None), slice(37, 4900), slice(1000, 1001), slice(512, 1536)],
    ids=["all", "partial-chunks-several-shards", "one-value", "whole-chunks"],
)
def test_contiguous_reads(tmp_path, fast_path_calls, read_path, *, fetch, selection):
    """Contiguous ranges of sharded 1-D arrays, like a sparse matrix's data and indices."""
    zarr.config.set({"kvikio.fetch": fetch})
    data = rng.random(5000).astype("float32")
    write(tmp_path / "a.zarr", data, (256,), shards=(1024,))
    check(tmp_path / "a.zarr", selection)
    if read_path == "gpu":
        assert fast_path_calls == ["_read_gpu_contiguous"]


@pytest.mark.skipif(find_spec("zarrs") is None, reason="needs zarrs")
def test_contiguous_reads_unordered_shards(tmp_path, fast_path_calls, read_path):
    """zarrs writes the chunks of a shard in whatever order they are encoded."""
    data = rng.random(1 << 16).astype("float32")
    path = tmp_path / "a.zarr"
    with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}):
        import zarrs  # noqa: F401

        write(path, data, (256,), shards=(1 << 14,))
    index = np.frombuffer((path / "c" / "0").read_bytes()[-(16 * 64 + 4) : -4], "<u8")
    if np.all(np.diff(index[0::2].astype(np.int64)) > 0):
        pytest.skip("zarrs wrote this shard in order")
    check(path, slice(300, 60_000))
    if read_path == "gpu":
        assert fast_path_calls == ["_read_gpu_contiguous"]


def test_contiguous_reads_missing_chunks(tmp_path, fast_path_calls, read_path):
    """Unwritten inner chunks are filled by the general path."""
    data = np.full(4096, 7, dtype="float32")
    data[:1000] = rng.random(1000)
    write(tmp_path / "a.zarr", data, (256,), shards=(1024,), fill_value=7)
    check(tmp_path / "a.zarr", slice(10, 4000))
    if read_path == "gpu":
        assert fast_path_calls == ["_read_gpu_contiguous", "_read_gpu"]


def test_contiguous_reads_in_parallel(tmp_path, monkeypatch, fast_path_calls, read_path):
    """Large stored ranges are read by several threads."""
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(_pipeline, "_PARALLEL_READ_BYTES", 0)
    used = []
    submit = ThreadPoolExecutor.map

    def spy(self, *args, **kwargs):
        used.append(self._thread_name_prefix)
        return submit(self, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "map", spy)
    data = rng.random(1 << 15).astype("float32")
    write(tmp_path / "a.zarr", data, (512,), shards=(1 << 13,))
    check(tmp_path / "a.zarr", slice(5, (1 << 15) - 5))
    if read_path == "gpu":
        assert "zarr-kvikio-read" in used


def test_release_memory_staging(tmp_path, read_path):
    if read_path == "host":
        pytest.skip("host reads use no pinned staging buffers")
    zarr.config.set({"kvikio.fetch": "host"})
    data = rng.random(1 << 15).astype("float32")
    write(tmp_path / "a.zarr", data, (512,), shards=(1 << 13,))
    check(tmp_path / "a.zarr", slice(None))
    assert _pipeline._staging
    zarr_kvikio.release_memory()
    assert not _pipeline._staging
    check(tmp_path / "a.zarr", slice(None))  # works again afterwards
