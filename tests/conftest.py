from __future__ import annotations

import pytest
import zarr

_KEYS = ("codec_pipeline.path", "buffer", "ndbuffer")


@pytest.fixture(autouse=True)
def _restore_zarr_config():
    """`zarr_kvikio.enable()` without a ``with`` block changes zarr's global config."""
    before = {k: zarr.config.get(k) for k in _KEYS}
    yield
    zarr.config.set(before)
