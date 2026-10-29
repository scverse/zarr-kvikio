from __future__ import annotations

import pytest
import zarr

from zarr_kvikio._pipeline import HOST_PIPELINE_KEY

_KEYS = ("codec_pipeline.path", "buffer", "ndbuffer", HOST_PIPELINE_KEY)


@pytest.fixture(autouse=True)
def _restore_zarr_config():
    """`zarr_kvikio.enable()` without a ``with`` block changes zarr's global config."""
    before = {k: zarr.config.get(k, None) for k in _KEYS}
    yield
    zarr.config.set(before)
