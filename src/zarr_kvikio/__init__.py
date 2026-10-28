"""Read zarr chunks straight into GPU memory with KvikIO and decode them with nvCOMP."""

from __future__ import annotations

from zarr.registry import register_pipeline

from ._check import check, describe
from ._config import available, enable, require
from ._convert import convert_array, cpu_pipeline, layout
from ._pipeline import KvikioCodecPipeline, release_memory

__all__ = [
    "KvikioCodecPipeline",
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
