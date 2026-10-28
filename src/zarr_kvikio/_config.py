from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import zarr

from ._pipeline import HOST_PIPELINE_KEY

if TYPE_CHECKING:
    from donfig.config_obj import ConfigSet

PIPELINE_PATH = "zarr_kvikio.KvikioCodecPipeline"
INSTALL_HINT = (
    "Reading zarr into GPU memory needs KvikIO and nvCOMP: "
    "`pip install 'zarr-kvikio[cuda13]'` (or `[cuda12]`)."
)


def available() -> bool:
    """Whether zarr arrays can be read into GPU memory (KvikIO and nvCOMP are installed)."""
    # `find_spec("nvidia.nvcomp")` raises if there is no `nvidia` package at all
    return all(find_spec(m) is not None for m in ("kvikio", "nvidia", "nvidia.nvcomp"))


def require() -> None:
    """Raise an :class:`ImportError` with install instructions unless :func:`available`."""
    if not available():
        raise ImportError(INSTALL_HINT)


def enable(*, num_threads: int | None = 8) -> ConfigSet:
    """Read zarr arrays into GPU memory with KvikIO and nvCOMP, in this process.

    Applies right away, or only within a ``with`` block::

        zarr_kvikio.enable()
        with zarr_kvikio.enable():
            ...

    The GPU decodes chunks that are zstd-compressed (or uncompressed), ideally sharded with
    small inner chunks (see :func:`convert_array` and :func:`describe`). Other chunks
    (e.g. blosc-compressed) are decompressed on the CPU, with the codec pipeline zarr was
    configured with before (e.g. zarrs) or else zarr's fused pipeline, and then copied to the GPU,
    so any array can be read. For a Dask cluster, use :func:`zarr_kvikio.dask.enable`.

    Parameters
    ----------
    num_threads
        Number of threads KvikIO reads with (a global KvikIO setting; its default is 1).
        ``None`` leaves it unchanged.
    """
    require()
    return _enable(num_threads, _host_pipeline())


def _enable(num_threads: int | None, host_pipeline: str | None) -> ConfigSet:
    if num_threads is not None:
        import kvikio.defaults

        kvikio.defaults.set({"num_threads": num_threads})
    return zarr.config.set(
        {
            "codec_pipeline.path": PIPELINE_PATH,
            HOST_PIPELINE_KEY: host_pipeline,
            "buffer": "zarr.buffer.gpu.Buffer",
            "ndbuffer": "zarr.buffer.gpu.NDBuffer",
        }
    )


def _host_pipeline() -> str | None:
    """The pipeline zarr is configured with, unless it is ours (then the one before that)."""
    path = zarr.config.get("codec_pipeline.path")
    return zarr.config.get(HOST_PIPELINE_KEY, None) if path == PIPELINE_PATH else path
