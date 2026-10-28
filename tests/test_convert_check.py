from __future__ import annotations

import numpy as np
import pytest
import zarr
from zarr.codecs import BloscCodec

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


def test_found_by_zarr():
    """Through the entry point, without importing zarr_kvikio first."""
    import subprocess
    import sys

    code = (
        "import sys, zarr; from zarr.registry import get_pipeline_class\n"
        "zarr.config.set({'codec_pipeline.path': 'zarr_kvikio.KvikioCodecPipeline'})\n"
        "assert get_pipeline_class().__name__ == 'KvikioCodecPipeline'\n"
        "assert 'zarr_kvikio' in sys.modules  # imported by zarr"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


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


@pytest.mark.skipif(not zarr_kvikio.available(), reason="needs KvikIO and nvCOMP")
def test_convert_array_after_enable(tmp_path):
    """Converts in host memory, also when zarr reads into GPU memory."""
    data = rng.random(1000).astype(np.float32)
    src = blosc_array(tmp_path / "src.zarr", data)
    dst = zarr.open_group(tmp_path / "dst.zarr", mode="w")
    with zarr_kvikio.enable():
        zarr_kvikio.convert_array(src, dst, "x", chunk_bytes=256)
    np.testing.assert_array_equal(zarr.open_array(tmp_path / "dst.zarr/x", mode="r")[...], data)
