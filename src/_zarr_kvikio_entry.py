"""The ``zarr.codec_pipeline`` entry point of zarr-kvikio.

zarr loads every pipeline entry point whenever it creates a codec pipeline, and one that fails to
load makes every array open fail. zarr_kvikio uses zarr internals, so a zarr release that moves
them would break all zarr reads wherever zarr-kvikio is installed. If zarr_kvikio cannot be
imported, this registers a stand-in that raises the import error only when it is selected.
"""

from __future__ import annotations

from typing import NoReturn


def _load() -> type:
    try:
        from zarr_kvikio import KvikioCodecPipeline
    except Exception as err:  # noqa: BLE001
        # not only ImportError: zarr_kvikio subclasses zarr internals at import, so moved ones
        # also raise e.g. AttributeError, and this guard must never break zarr
        return _stub(err)
    return KvikioCodecPipeline


def _stub(err: Exception) -> type:
    class KvikioCodecPipeline:
        """Stands in for :class:`zarr_kvikio.KvikioCodecPipeline` if it can't be imported."""

        # registered under the same name as the real pipeline
        __module__ = "zarr_kvikio"
        __qualname__ = "KvikioCodecPipeline"

        @classmethod
        def from_codecs(cls, *args: object, **kwargs: object) -> NoReturn:
            msg = (
                "zarr's codec pipeline is zarr_kvikio.KvikioCodecPipeline, "
                f"but zarr_kvikio failed to import: {err}"
            )
            raise ImportError(msg) from err

        from_array_metadata_and_store = from_codecs

    return KvikioCodecPipeline


KvikioCodecPipeline = _load()
