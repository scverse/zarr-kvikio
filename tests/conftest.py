from __future__ import annotations

import copy

import pytest
import zarr

_KEYS = ("codec_pipeline.path", "buffer", "ndbuffer")


@pytest.fixture(autouse=True)
def _restore_zarr_config():
    """`zarr_kvikio.enable()` without a ``with`` block, and tests, change zarr's global config."""
    before = {k: zarr.config.get(k, None) for k in _KEYS}
    before["kvikio"] = copy.deepcopy(zarr.config.get("kvikio", {}))
    yield
    zarr.config.set(before)
