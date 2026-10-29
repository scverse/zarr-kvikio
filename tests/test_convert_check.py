from __future__ import annotations

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import entry_points

import numpy as np
import pytest
import zarr
import zarr.registry
from zarr.codecs import BloscCodec, BytesCodec, ZstdCodec

import zarr_kvikio

rng = np.random.default_rng(0)


def blosc_array(path, data) -> zarr.Array:
    zarr.create_array(path, data=data, chunks=data.shape, compressors=[BloscCodec()])
    with zarr_kvikio.cpu_pipeline():
        return zarr.open_array(path, mode="r")


def test_layout():
    kw = {"chunk_bytes": 1024, "chunks_per_shard": 4}
    assert zarr_kvikio.layout((10_000,), 4, **kw) == ((256,), (1024,))
    assert zarr_kvikio.layout((100, 100), 4, **kw) == ((2, 100), (8, 100))  # whole rows
    assert zarr_kvikio.layout((100, 300), 4, **kw) == ((1, 256), (2, 512))  # row pieces
    with pytest.raises(ValueError, match="1-D and 2-D"):
        zarr_kvikio.layout((10, 10, 10), 4)


@pytest.mark.parametrize("shape", [(10_000,), (100, 100), (30, 300)])
def test_convert_array(tmp_path, shape):
    data = rng.random(shape).astype(np.float32)
    src = blosc_array(tmp_path / "src.zarr", data)
    dst = zarr.open_group(tmp_path / "dst.zarr", mode="w")
    new = zarr_kvikio.convert_array(src, dst, "x", chunk_bytes=1024, chunks_per_shard=4)
    np.testing.assert_array_equal(new[...], data)

    before, after = zarr_kvikio.describe(src), zarr_kvikio.describe(new)
    assert not before["gpu_readable"]
    assert any("cannot be decoded on the GPU" in p for p in before["problems"])
    assert after["gpu_readable"]
    assert after["sharded"]
    assert after["problems"] == []


def test_convert_array_dtype(tmp_path):
    dst = zarr.open_group(tmp_path / "dst.zarr", mode="w")
    small = blosc_array(tmp_path / "small.zarr", np.arange(1000, dtype=np.int64))
    new = zarr_kvikio.convert_array(small, dst, "small", dtype=np.uint16, chunk_bytes=256)
    assert new.dtype == np.uint16
    np.testing.assert_array_equal(new[...], np.arange(1000))

    large = blosc_array(tmp_path / "large.zarr", np.arange(70_000, dtype=np.int64))
    with pytest.raises(ValueError, match="cannot hold"):
        zarr_kvikio.convert_array(large, dst, "large", dtype=np.uint16)


def test_convert_array_validate(tmp_path):
    src = blosc_array(tmp_path / "src.zarr", np.arange(1000, dtype=np.int64))
    dst = zarr.open_group(tmp_path / "dst.zarr", mode="w")

    def at_most_500(block, start, stop):
        if block.max() > 500:
            msg = f"too large in [{start}, {stop})"
            raise ValueError(msg)

    with pytest.raises(ValueError, match="too large"):
        zarr_kvikio.convert_array(src, dst, "x", validate=at_most_500, chunk_bytes=256)


def test_check():
    report = zarr_kvikio.check()
    assert report["available"] == zarr_kvikio.available()
    assert report["versions"]["zarr"] == zarr.__version__
    assert "status" in report["gds"]


@pytest.mark.skipif(not zarr_kvikio.available(), reason="needs KvikIO and nvCOMP")
def test_check_kvikio_compat_mode(monkeypatch):
    """KvikIO doesn't use GPUDirect Storage in compatibility mode, whatever gdscheck says."""
    monkeypatch.setattr("kvikio.defaults.is_compat_mode_preferred", lambda: True)
    gds = zarr_kvikio.check()["gds"]
    assert gds["kvikio_compat_mode"]
    assert gds["status"] == "compatibility mode"


def test_describe_unordered_shards(tmp_path):
    """zarrs writes the chunks of a shard in whatever order they were compressed in."""
    pytest.importorskip("zarrs")
    data = rng.random(1 << 16).astype(np.float32)
    with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}):
        arr = zarr.create_array(
            tmp_path / "zarrs.zarr",
            shape=data.shape,
            dtype=data.dtype,
            chunks=(256,),
            shards=(1 << 14,),
        )
        arr[:] = data
    problems = zarr_kvikio.describe(arr)["problems"]
    ordered = zarr_kvikio.convert_array(arr, zarr.open_group(tmp_path / "dst.zarr", mode="w"), "x")
    assert zarr_kvikio.describe(ordered)["problems"] == []
    if not problems:
        pytest.skip("zarrs wrote these shards in order")
    assert any("out of order" in p for p in problems)


@pytest.mark.parametrize(
    ("codecs", "needs"),
    [
        pytest.param({}, None, id="zstd"),
        pytest.param({"compressors": [ZstdCodec(checksum=True)]}, "without checksum", id="crc"),
        pytest.param({"serializer": BytesCodec(endian="big")}, "little-endian", id="big-endian"),
    ],
)
def test_describe_memory_store(codecs, needs):
    """Only arrays in a LocalStore are read with KvikIO and decoded on the GPU."""
    arr = zarr.create_array(
        zarr.storage.MemoryStore(), shape=(100,), dtype="f4", chunks=(10,), shards=(100,), **codecs
    )
    desc = zarr_kvikio.describe(arr)
    assert desc["gpu_readable"] == (needs is None)
    assert any("not a LocalStore" in p for p in desc["problems"])
    assert needs is None or any(needs in p for p in desc["problems"])


def test_found_by_zarr(tmp_path):
    """Selected and set up in zarr's config alone (here from the environment), like zarrs."""
    code = (
        "import sys, numpy as np, zarr; from zarr.registry import get_pipeline_class\n"
        "assert get_pipeline_class().__name__ == 'KvikioCodecPipeline'\n"
        "assert 'zarr_kvikio' in sys.modules  # imported by zarr, through the entry point\n"
        "assert zarr.config.get('kvikio.min_bytes') == 0\n"
        f"a = zarr.create_array({str(tmp_path / 'a.zarr')!r}, shape=(9,), dtype='f4')\n"
        "a[:] = np.arange(9)  # in host memory: with zarr's fused pipeline\n"
        "assert (a[:] == np.arange(9)).all()"
    )
    env = {
        **os.environ,
        "ZARR_CODEC_PIPELINE__PATH": "zarr_kvikio.KvikioCodecPipeline",
        "ZARR_KVIKIO__MIN_BYTES": "0",
    }
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.parametrize("error", [ImportError, AttributeError])
def test_entry_point_import_error(tmp_path, monkeypatch, error):
    """zarr loads every pipeline entry point on each open: if zarr_kvikio fails to import (e.g.
    because zarr moved its internals), only selecting this pipeline fails."""
    name = "zarr_kvikio.KvikioCodecPipeline"
    ep = entry_points(group="zarr.codec_pipeline")[name]
    assert ep.value == "_zarr_kvikio_entry:KvikioCodecPipeline"
    assert ep.load() is zarr_kvikio.KvikioCodecPipeline
    path = tmp_path / "a.zarr"
    zarr.create_array(path, data=np.arange(10))

    class Broken:
        """Fails to import zarr_kvikio (without a module file, which coverage would report)."""

        @staticmethod
        def find_spec(fullname, *_):
            if fullname == "zarr_kvikio":
                raise error("zarr moved")

    monkeypatch.setattr(sys, "meta_path", [Broken(), *sys.meta_path])
    monkeypatch.delitem(sys.modules, "zarr_kvikio")
    monkeypatch.delitem(sys.modules, ep.module)
    # zarr's pipeline registry before its first codec pipeline, with the entry point to load
    registry = zarr.registry.Registry()
    registry.update(zarr.registry._pipeline_registry)
    registry.lazy_load_list.append(ep)
    monkeypatch.setattr(zarr.registry, "_pipeline_registry", registry)

    np.testing.assert_array_equal(zarr.open_array(path, mode="r")[...], np.arange(10))
    assert registry[name] is not zarr_kvikio.KvikioCodecPipeline  # the stand-in
    with (
        zarr.config.set({"codec_pipeline.path": name}),
        pytest.raises(ImportError, match="failed to import") as err,
    ):
        zarr.open_array(path, mode="r")
    assert isinstance(err.value.__cause__, error)


def test_convert_array_nan_cast(tmp_path):
    """A lossless cast of data with NaN (which never equals itself) is accepted."""
    data = rng.random(1000).astype(np.float32)
    data[::7] = np.nan
    src = blosc_array(tmp_path / "src.zarr", data)
    dst = zarr.open_group(tmp_path / "dst.zarr", mode="w")
    new = zarr_kvikio.convert_array(src, dst, "x", dtype=np.float64, chunk_bytes=256)
    np.testing.assert_array_equal(new[...], data.astype(np.float64))


def test_convert_array_needs_chunks_and_shards(tmp_path):
    src = blosc_array(tmp_path / "src.zarr", np.arange(1000))
    dst = zarr.open_group(tmp_path / "dst.zarr", mode="w")
    with pytest.raises(ValueError, match="both"):
        zarr_kvikio.convert_array(src, dst, "x", chunks=(100,))


def test_convert_array_1d_or_2d(tmp_path):
    src = blosc_array(tmp_path / "src.zarr", np.zeros((2, 3, 4)))
    dst = zarr.open_group(tmp_path / "dst.zarr", mode="w")
    with pytest.raises(ValueError, match="1-D and 2-D"):
        zarr_kvikio.convert_array(src, dst, "x", chunks=(1, 3, 4), shards=(2, 3, 4))
    assert "x" not in dst


@pytest.mark.skipif(not zarr_kvikio.available(), reason="needs KvikIO and nvCOMP")
def test_convert_array_after_enable(tmp_path):
    """Converts in host memory, also when zarr reads into GPU memory, and returns the new array
    with the GPU pipeline."""
    data = rng.random(1000).astype(np.float32)
    src = blosc_array(tmp_path / "src.zarr", data)
    dst = zarr.open_group(tmp_path / "dst.zarr", mode="w")
    with zarr_kvikio.enable():
        new = zarr_kvikio.convert_array(src, dst, "x", chunk_bytes=256)
    assert isinstance(new._async_array.codec_pipeline, zarr_kvikio.KvikioCodecPipeline)
    np.testing.assert_array_equal(new[...], data)


def test_convert_array_one_at_a_time(tmp_path):
    """Concurrent conversions don't restore zarr's buffer settings while another one runs."""
    src = blosc_array(tmp_path / "src.zarr", np.arange(1000))
    dst = zarr.open_group(tmp_path / "dst.zarr", mode="w")
    a_started, b_started, a_done = threading.Event(), threading.Event(), threading.Event()

    def a_blocks(block, start, stop):
        a_started.set()
        if start == 0:
            b_started.wait(0.5)  # times out, as `b` waits for `a`

    def b_blocks(block, start, stop):
        b_started.set()
        a_done.wait(5)

    kw = {"chunks": (10,), "shards": (100,), "n_threads": 1, "verify": False}
    # stand-ins for those of `zarr_kvikio.enable()`: reading with them fails
    settings = {"buffer": "no.Buffer", "ndbuffer": "no.NDBuffer"}
    with zarr.config.set(settings), ThreadPoolExecutor(2) as ex:
        a = ex.submit(zarr_kvikio.convert_array, src, dst, "a", validate=a_blocks, **kw)
        assert a_started.wait(5)
        b = ex.submit(zarr_kvikio.convert_array, src, dst, "b", validate=b_blocks, **kw)
        a.result()
        a_done.set()
        b.result()
        assert {k: zarr.config.get(k) for k in settings} == settings
    np.testing.assert_array_equal(dst["b"][...], np.arange(1000))
