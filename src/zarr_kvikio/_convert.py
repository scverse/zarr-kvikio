from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from importlib.util import find_spec
from math import ceil, prod
from typing import TYPE_CHECKING

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from donfig.config_obj import ConfigSet
    from numpy.typing import DTypeLike


def layout(
    shape: tuple[int, ...],
    itemsize: int,
    *,
    chunk_bytes: int = 256 << 10,
    chunks_per_shard: int = 512,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Chunk and shard shape for fast GPU reads of a 1-D or 2-D array, as ``(chunks, shards)``.

    Chunks of about `chunk_bytes` decoded bytes (nvCOMP decodes every chunk with a single thread
    block, so large chunks decode slowly): whole rows of a 2-D array, or row pieces if a single
    row is larger. `chunks_per_shard` chunks per shard (file), so a read opens a few files instead
    of one per chunk.
    """
    _check_ndim(shape)
    if len(shape) == 1:
        inner = max(1, chunk_bytes // itemsize)
        return (inner,), (inner * chunks_per_shard,)
    n_cols = shape[1]
    if n_cols * itemsize <= chunk_bytes:  # whole rows per chunk
        rows = max(1, chunk_bytes // (n_cols * itemsize))
        return (rows, n_cols), (rows * chunks_per_shard, n_cols)
    # every row in pieces of `chunk_bytes`
    piece = max(1, chunk_bytes // itemsize)
    per_row = ceil(n_cols / piece)
    return (1, piece), (max(1, chunks_per_shard // per_row), per_row * piece)


def convert_array(
    src: zarr.Array,
    dst: zarr.Group,
    name: str,
    *,
    dtype: DTypeLike | None = None,
    chunks: tuple[int, ...] | None = None,
    shards: tuple[int, ...] | None = None,
    validate: Callable[[np.ndarray, int, int], None] | None = None,
    chunk_bytes: int = 256 << 10,
    chunks_per_shard: int = 512,
    level: int = 3,
    n_threads: int = 16,
    verify: bool = True,
) -> zarr.Array:
    """Stream a 1-D or 2-D numeric array into ``dst[name]``, laid out for fast GPU reads.

    The new array is zstd-compressed (which nvCOMP decodes on the GPU) with the chunks and shards
    of :func:`layout` (unless given), and the chunks of every shard are stored in order, so a
    contiguous range is a single read. It is written shard by shard with `n_threads` threads,
    so `src` is never read into memory at once; open `src` within :func:`cpu_pipeline` to read
    it fast. The attributes of `src` are copied. Returns the new array, opened with the codec
    pipeline zarr is configured with (e.g. the GPU one after :func:`enable`).

    Conversions run one at a time, in host memory: while one runs, zarr's process-wide buffer
    settings are host buffers (also after :func:`enable`), so zarr reads in other threads
    return NumPy arrays, and while it creates the new array, the codec pipeline is zarr's fused
    one, which arrays opened in other threads meanwhile keep.

    Parameters
    ----------
    src
        The (non-empty) array to convert.
    dst
        The group to create the new array in.
    name
        Name of the new array.
    dtype
        Store the values with this dtype; raises if a value changes (e.g. ``np.uint16`` for
        indices below 65,536).
    chunks, shards
        Chunk and shard shape (both or neither), instead of those of :func:`layout`.
    validate
        Called as ``validate(block, start, stop)`` with every block of values read from `src` and
        the half-open row range ``[start, stop)`` it covers, before the block is written; raise
        to stop. It runs in worker threads while conversions are serialised, so it must not call
        :func:`convert_array`.
    chunk_bytes, chunks_per_shard
        See :func:`layout`.
    level
        zstd compression level.
    n_threads
        Threads to convert shards with.
    verify
        Read the new array back and check it is identical to `src`.
    """
    _check_ndim(src.shape)
    dtype = np.dtype(src.dtype if dtype is None else dtype)
    if (chunks is None) != (shards is None):
        msg = "Give both `chunks` and `shards`, or neither."
        raise ValueError(msg)
    if chunks is None or shards is None:
        chunks, shards = layout(
            src.shape, dtype.itemsize, chunk_bytes=chunk_bytes, chunks_per_shard=chunks_per_shard
        )
    # in host memory, also after `zarr_kvikio.enable()`; zarr's config is process-wide, so one
    # conversion at a time sets (and restores) it
    with _LOCK, zarr.config.set(_HOST_BUFFERS):
        with _write_pipeline():
            dx = dst.create_array(
                name,
                shape=src.shape,
                dtype=dtype,
                chunks=chunks,
                shards=shards,
                compressors=[ZstdCodec(level=level)],
            )
        dx.attrs.update(dict(src.attrs))
        n_rows, per_row, step = src.shape[0], prod(src.shape[1:]), shards[0]
        label = src.path or name

        def convert(k: int) -> int:
            s, e = k * step, min(n_rows, (k + 1) * step)
            block = np.asarray(src[s:e])
            if validate is not None:
                validate(block, s, e)
            values = block.astype(dtype, copy=False)
            if values.dtype != block.dtype and not np.array_equal(values, block, equal_nan=True):
                msg = f"`{label}` has values in [{s}, {e}) that {dtype} cannot hold."
                raise ValueError(msg)
            dx[s:e] = values
            return (e - s) * per_row

        def check(k: int) -> int:
            s, e = k * step, min(n_rows, (k + 1) * step)
            if not np.array_equal(dx[s:e], src[s:e], equal_nan=True):
                msg = f"The converted `{label}` differs from the source in [{s}, {e})."
                raise AssertionError(msg)
            return (e - s) * per_row

        steps = [(f"Converting {label}", convert)]
        if verify:
            steps.append((f"Verifying {label}", check))
        for desc, fn in steps:
            with ThreadPoolExecutor(n_threads) as ex, _progress(n_rows * per_row, desc) as update:
                for n in ex.map(fn, range(ceil(n_rows / step))):
                    update(n)
        # `dx` keeps the fused pipeline it was created with: reopen it with the caller's (e.g.
        # the GPU one), within the lock so that it can't get another conversion's fused one
        return zarr.Array.open(dx.store_path)


@contextmanager
def _progress(total: int, desc: str) -> Iterator[Callable[[int], object]]:
    """A progress bar's ``update``, if tqdm is installed."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None
    if tqdm is None:  # outside `except`: errors in the block must not chain to the ImportError
        yield lambda n: None
        return
    with tqdm(total=total, desc=desc, unit_scale=True, unit=" values") as bar:
        yield bar.update


def cpu_pipeline() -> ConfigSet:
    """Host buffers and a fast CPU codec pipeline (zarrs if installed, else zarr's fused one).

    For the arrays opened within, e.g. the source of :func:`convert_array`::

        with zarr_kvikio.cpu_pipeline():
            src = zarr.open_array("data.zarr", mode="r")

    It sets zarr's process-wide config, so overlapping blocks in several threads are not supported.
    """
    settings = dict(_HOST_BUFFERS)
    if find_spec("zarrs") is not None:
        import zarrs  # noqa: F401  (registers the pipeline)

        settings["codec_pipeline.path"] = "zarrs.ZarrsCodecPipeline"
    else:
        settings["codec_pipeline.path"] = "zarr.core.codec_pipeline.FusedCodecPipeline"
    return zarr.config.set(settings)


_HOST_BUFFERS = {"buffer": "zarr.buffer.cpu.Buffer", "ndbuffer": "zarr.buffer.cpu.NDBuffer"}
_LOCK = threading.Lock()


def _check_ndim(shape: tuple[int, ...]) -> None:
    if len(shape) not in (1, 2):
        msg = f"Only 1-D and 2-D arrays are supported, not shape {shape}."
        raise ValueError(msg)


def _write_pipeline() -> ConfigSet:
    """The fused zarr pipeline, for the arrays created within.

    It stores the chunks of a shard in order, so a contiguous range is one read. zarrs (faster
    at writing) stores them in whatever order they were compressed in, scattering a range
    over the whole shard.
    """
    return zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline"})
