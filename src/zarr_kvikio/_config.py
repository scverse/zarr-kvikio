from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import zarr

if TYPE_CHECKING:
    from donfig.config_obj import ConfigSet

# zarr-kvikio's settings (e.g. `kvikio.min_bytes`): a section of zarr's config of its own, so that
# undoing a change of one key (e.g. by `enable()`) keeps the others
zarr.config.update_defaults({"kvikio": {}})

PIPELINE_PATH = "zarr_kvikio.KvikioCodecPipeline"
# zarr config key of the pipeline that reads into host buffers and decodes what the GPU can't
# (e.g. zarrs; zarr's fused pipeline if not set)
HOST_PIPELINE_KEY = "kvikio.host_pipeline"
ZARR_DEFAULT_PIPELINE = "zarr.core.codec_pipeline.BatchedCodecPipeline"
GPU_BUFFERS = {"buffer": "zarr.buffer.gpu.Buffer", "ndbuffer": "zarr.buffer.gpu.NDBuffer"}
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


def enable() -> ConfigSet:
    """Read zarr arrays into GPU memory with KvikIO and nvCOMP, in this process.

    Applies right away, or only within a ``with`` block::

        zarr_kvikio.enable()
        with zarr_kvikio.enable():
            ...

    It selects the pipeline and GPU buffers in zarr's config, as other packages can without
    importing zarr-kvikio::

        zarr.config.set({"codec_pipeline.path": "zarr_kvikio.KvikioCodecPipeline"})
        zarr.config.enable_gpu()

    but first checks that KvikIO and nvCOMP are installed, and keeps the pipeline zarr was
    configured with before (e.g. zarrs) for the CPU, as ``kvikio.host_pipeline`` (unless that is
    set). As with ``zarr.config.set``, nested calls must be undone in reverse order, also
    relative to :func:`zarr_kvikio.dask.enable` in this process.

    The GPU decodes chunks that are zstd-compressed (or uncompressed), ideally sharded with
    small inner chunks (see :func:`convert_array` and :func:`describe`). Other chunks
    (e.g. blosc-compressed) are decompressed on the CPU, with ``kvikio.host_pipeline`` or else
    zarr's fused pipeline, and then copied to the GPU, so any array can be read. For a Dask
    cluster, use :func:`zarr_kvikio.dask.enable`.
    """
    require()
    return zarr.config.set(
        {"codec_pipeline.path": PIPELINE_PATH, **GPU_BUFFERS, **_host_pipeline()}
    )


def _host_pipeline() -> dict[str, str]:
    """The pipeline zarr is configured with, to keep as ``kvikio.host_pipeline``.

    Unless that is set, or it is ours or zarr's default (the fused one is faster).
    """
    path = zarr.config.get("codec_pipeline.path")
    if zarr.config.get(HOST_PIPELINE_KEY, None) is not None or path in (
        PIPELINE_PATH,
        ZARR_DEFAULT_PIPELINE,
    ):
        return {}
    return {HOST_PIPELINE_KEY: path}
