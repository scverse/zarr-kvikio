"""Read zarr chunks straight into GPU memory with KvikIO and decode them with nvCOMP."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from zarr.registry import register_pipeline

from ._check import check, describe
from ._config import available, enable, require
from ._convert import convert_array, cpu_pipeline, layout
from ._pipeline import KvikioCodecPipeline, release_memory

try:
    __version__ = version("zarr-kvikio")
except PackageNotFoundError:  # imported from a source tree that isn't installed
    __version__ = "0+unknown"

__all__ = [
    "KvikioCodecPipeline",
    "__version__",
    "available",
    "check",
    "convert_array",
    "cpu_pipeline",
    "describe",
    "enable",
    "layout",
    "release_memory",
    "require",
]

# selected via `codec_pipeline.path` as "zarr_kvikio.KvikioCodecPipeline" (also found through the
# `zarr.codec_pipeline` entry point, without importing this package first)
register_pipeline(KvikioCodecPipeline)
