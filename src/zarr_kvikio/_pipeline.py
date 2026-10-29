"""A zarr codec pipeline that reads chunks into GPU memory with KvikIO and decodes them with nvCOMP.

Reads that can take the fast path

1. read the (small) shard indices on the host,
2. read every needed chunk's compressed bytes straight into device memory with KvikIO,
   one read per contiguous byte range (i.e. typically one read per shard); contiguous ranges of
   zstd-compressed sharded 1-D arrays are read into pinned host memory and copied instead, by
   default (``kvikio.fetch``),
3. decode the chunks of the request in batched nvCOMP calls of at most
   ``kvikio.max_decode_bytes`` decoded bytes each (usually a single call),
4. copy the decoded values into the output buffer
   (fully selected chunks of 1-D arrays are decoded in place, without that copy).

Reads of fewer than ``kvikio.min_bytes`` bytes are read and decoded on the CPU instead, and
copied to the GPU at once. (The settings are keys of zarr's config.)

Everything else works with any store: reads into host buffers and writes of host buffers are
delegated to a host pipeline (``kvikio.host_pipeline``, e.g. zarrs, else zarr's
:class:`~zarr.core.codec_pipeline.FusedCodecPipeline`), and reads into GPU buffers the GPU can't
decode (e.g. blosc-compressed or remote chunks) are decoded by it on the host and then copied.
Writes of GPU buffers, and reads of other data types (e.g. strings) into them, use zarr's
:class:`~zarr.core.codec_pipeline.BatchedCodecPipeline`.
"""

from __future__ import annotations

import asyncio
import atexit
import importlib
import os
import queue
import sys
import threading
from collections import deque
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from functools import cache, cached_property
from itertools import batched, chain, groupby, product
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import google_crc32c
import numpy as np
import zarr
from zarr.abc.codec import CodecPipeline, GetResult
from zarr.codecs import BytesCodec, Crc32cCodec, ShardingCodec, ZstdCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import core, cpu
from zarr.core.chunk_utils import fill_value_or_default
from zarr.core.codec_pipeline import BatchedCodecPipeline, FusedCodecPipeline
from zarr.storage import LocalStore, StorePath

from ._config import HOST_PIPELINE_KEY, ZARR_DEFAULT_PIPELINE, require

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
    from zarr.abc.store import ByteGetter, ByteSetter, Store
    from zarr.core.buffer import NDBuffer
    from zarr.core.indexing import SelectorTuple
    from zarr.core.metadata import ArrayMetadata

    type BatchInfo = tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]

__all__ = ["KvikioCodecPipeline", "release_memory"]

_MAX_UINT_64 = 2**64 - 1
# Byte ranges of one file closer together than this are fetched with a single read.
_MERGE_GAP = 1 << 20


# ---------------------------------------------------------------------------------------------
# Deciding whether the fast path applies
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Layout:
    """What the fast path needs to know about how chunks are encoded on disk."""

    compressed: bool  # zstd if True, raw bytes if False
    inner_chunk_shape: tuple[int, ...] | None = None  # None if not sharded
    index_at_end: bool = True
    index_has_crc: bool = True


def _is_little_bytes(codec: object) -> bool:
    return isinstance(codec, BytesCodec) and codec.endian in (None, "little")


def _compression(codecs: Sequence[object]) -> bool | None:
    """True for zstd, False for no compression, None if unsupported."""
    if not codecs:
        return False
    if len(codecs) == 1 and isinstance(codecs[0], ZstdCodec) and not codecs[0].checksum:
        return True
    return None


def _layout_of(pipeline: BatchedCodecPipeline) -> _Layout | None:
    if pipeline.array_array_codecs or sys.byteorder != "little":
        return None
    ab = pipeline.array_bytes_codec
    if _is_little_bytes(ab):
        compressed = _compression(pipeline.bytes_bytes_codecs)
        return None if compressed is None else _Layout(compressed=compressed)
    if not isinstance(ab, ShardingCodec) or pipeline.bytes_bytes_codecs:
        return None
    inner, idx = ab.codecs, ab.index_codecs
    if not inner or not _is_little_bytes(inner[0]):
        return None
    if (compressed := _compression(inner[1:])) is None:
        return None
    if not idx or not _is_little_bytes(idx[0]):
        return None
    if len(idx) > 2 or (len(idx) == 2 and not isinstance(idx[1], Crc32cCodec)):
        return None
    return _Layout(
        compressed=compressed,
        inner_chunk_shape=tuple(ab.chunk_shape),
        index_at_end=ab.index_location == "end",
        index_has_crc=len(idx) == 2,
    )


def _local_path(byte_getter: object) -> Path | None:
    if isinstance(byte_getter, StorePath) and type(byte_getter.store) is LocalStore:
        return Path(byte_getter.store.root) / byte_getter.path
    return None


def _is_device_array(arr: object) -> bool:
    return type(arr).__module__.split(".")[0] == "cupy"


def _supported_dtype(spec: ArraySpec) -> bool:
    dtype = spec.dtype.to_native_dtype()
    return dtype.kind in "biufcmM" and not dtype.hasobject and dtype.itemsize > 0


class _HostNDBuffer(cpu.NDBuffer):
    """Host buffer to decode into before copying to the GPU.

    zarr's codec for zarr v2 arrays returns the globally configured buffer class (a GPU one
    after `zarr_kvikio.enable()`) instead of the requested one, so accept any.
    """

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, core.NDBuffer) and not isinstance(value, cpu.NDBuffer):
            value = value.as_numpy_array()
        super().__setitem__(key, value)


def _on_host(spec: ArraySpec) -> ArraySpec:
    return ArraySpec(
        shape=spec.shape,
        dtype=spec.dtype,
        fill_value=spec.fill_value,
        config=spec.config,
        prototype=cpu.buffer_prototype,
    )


class KvikioCodecPipeline(BatchedCodecPipeline):
    """Codec pipeline that reads chunks from a local file system straight into GPU memory.

    Reads of arrays whose chunks are zstd-compressed (without checksum) or uncompressed,
    optionally sharded, into GPU buffers take the KvikIO/nvCOMP fast path.
    Reads into host buffers and writes of host buffers go to the pipeline in zarr's config
    ``kvikio.host_pipeline`` (e.g. zarrs), or else to zarr's
    :class:`~zarr.core.codec_pipeline.FusedCodecPipeline`, which also decodes the reads into GPU
    buffers that the GPU can't (copied afterwards).
    Writes of GPU buffers, and reads of other data types (e.g. strings) into them, go to zarr's
    :class:`~zarr.core.codec_pipeline.BatchedCodecPipeline`. So the pipeline works with any array.

    zarr finds it (as ``"zarr_kvikio.KvikioCodecPipeline"``) through an entry point.
    The pipeline is bound when an array is opened, so select it before opening,
    e.g. with :func:`zarr_kvikio.enable`.
    """

    # the name zarr's registry and pickle know it by
    __module__ = "zarr_kvikio"

    def __getstate__(self) -> dict[str, Any]:
        # the host pipeline is made from the zarr config of the process that reads (a Dask worker)
        return {k: v for k, v in self.__dict__.items() if k != "_host"}

    @cached_property
    def _layout(self) -> _Layout | None:
        return _layout_of(self)

    @classmethod
    def from_array_metadata_and_store(
        cls, array_metadata: ArrayMetadata, store: Store
    ) -> KvikioCodecPipeline:
        # built from the codecs as zarr does without a store, keeping what host pipelines
        # like zarrs are built from
        from zarr.core.array import create_codec_pipeline

        pipeline = create_codec_pipeline(array_metadata)
        pipeline.__dict__["_array"] = (array_metadata, store)
        return cast("KvikioCodecPipeline", pipeline)

    async def read(
        self,
        batch_info: Iterable[BatchInfo],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> tuple[GetResult, ...]:
        batch = list(batch_info)
        if not batch:
            return ()
        out_arr: Any = out.as_ndarray_like()
        if not _is_device_array(out_arr):
            return await self._host_pipeline(batch[0][1]).read(batch, out, drop_axes)
        if not all(_supported_dtype(spec) for _, spec, *_ in batch):
            return await super().read(batch, out, drop_axes)
        layout = self._layout
        paths = [_local_path(bg) for bg, *_ in batch]
        # (zarr also takes an `out` of another dtype: `_read_via_host` casts into it)
        if (
            layout is None
            or any(p is None for p in paths)
            or any(spec.dtype.to_native_dtype() != out_arr.dtype for _, spec, *_ in batch)
        ):
            return await self._read_via_host(batch, out_arr, drop_axes)
        local = cast("list[Path]", paths)
        _require_once()
        if out_arr.nbytes < _int_setting(_MIN_BYTES_KEY, _DEFAULT_MIN_BYTES, 0):
            return await asyncio.to_thread(_read_host, layout, batch, local, out_arr, drop_axes)
        if not _contiguous_reads(layout, batch, out_arr, drop_axes):
            return await asyncio.to_thread(_read_gpu, layout, batch, local, out_arr, drop_axes)
        fetch = _fetch_mode()
        # each entry is a shard file, kept open while it is read: at most this many at once
        n = _open_files.files(_CONTIGUOUS_FDS[fetch])
        results: list[GetResult] = []
        for i in range(0, len(batch), n):
            part, part_paths = batch[i : i + n], local[i : i + n]
            got = await asyncio.to_thread(
                _read_gpu_contiguous, layout, part, part_paths, out_arr, fetch
            )
            if got is None:
                got = await asyncio.to_thread(
                    _read_gpu, layout, part, part_paths, out_arr, drop_axes
                )
            results.extend(got)
        return tuple(results)

    async def write(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch = list(batch_info)
        if not batch:
            return
        if _is_device_array(value.as_ndarray_like()):
            await super().write(batch, value, drop_axes)
        else:
            await self._host_pipeline(batch[0][1]).write(batch, value, drop_axes)

    def _host_pipeline(self, spec: ArraySpec) -> CodecPipeline:
        """The host pipeline for this array."""
        if (pipeline := self.__dict__.get("_host")) is None:
            host_cls = _host_pipeline_class()
            codecs: list[ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec] = [
                *self.array_array_codecs,
                self.array_bytes_codec,
                *self.bytes_bytes_codecs,
            ]
            # (zarr's batched pipeline, which zarrs falls back to and builds from codecs, can't
            # read shards into host buffers while GPU buffers are the default: the fused one can)
            if (array := self.__dict__.get("_array")) is not None:
                metadata, store = array
                with suppress(NotImplementedError):  # e.g. zarrs is built from these
                    pipeline = host_cls.from_array_metadata_and_store(
                        array_metadata=metadata, store=store
                    )
                if type(getattr(pipeline, "python_impl", None)) is BatchedCodecPipeline:
                    fallback = FusedCodecPipeline.from_codecs(codecs)
                    cast("Any", pipeline).python_impl = fallback.evolve_from_array_spec(
                        _on_host(spec)
                    )
            if pipeline is None:
                pipeline = host_cls.from_codecs(codecs)
                if type(pipeline) is BatchedCodecPipeline:
                    pipeline = FusedCodecPipeline.from_codecs(codecs)
                pipeline = pipeline.evolve_from_array_spec(_on_host(spec))
            # (a frozen dataclass: cache it the way `cached_property` does)
            self.__dict__["_host"] = pipeline
        return pipeline

    async def _read_via_host(
        self, batch: list[BatchInfo], out_arr: Any, drop_axes: tuple[int, ...]
    ) -> tuple[GetResult, ...]:
        """Decode chunks the GPU can't on the host, then copy them into the GPU buffer."""
        import cupy as cp

        pipeline = self._host_pipeline(batch[0][1])
        dtype = out_arr.dtype
        if not isinstance(pipeline, BatchedCodecPipeline | FusedCodecPipeline):
            # others (e.g. zarrs) may not cast into `out`: decoded in the array's dtype then, and
            # cast on the host (CuPy has no datetime dtypes)
            dtype = batch[0][1].dtype.to_native_dtype().newbyteorder("=")
        host = _HostNDBuffer.from_numpy_array(np.empty(out_arr.shape, dtype=dtype))
        host_batch: list[BatchInfo] = [
            (bg, _on_host(spec), chunk_sel, out_sel, is_complete)
            for bg, spec, chunk_sel, out_sel, is_complete in batch
        ]
        results = await pipeline.read(host_batch, host, drop_axes)
        with out_arr.device:  # (this thread's current device may be another one)
            out_arr[...] = cp.asarray(host.as_ndarray_like().astype(out_arr.dtype, copy=False))
        return results


def _host_pipeline_class() -> type[CodecPipeline]:
    path = zarr.config.get(HOST_PIPELINE_KEY, None)
    if path is None or path == ZARR_DEFAULT_PIPELINE:
        return FusedCodecPipeline
    cls, error = None, None
    try:
        module, _, name = path.rpartition(".")
        cls = getattr(importlib.import_module(module), name)
    except (AttributeError, ImportError, ValueError) as err:
        error = err
    if not (isinstance(cls, type) and issubclass(cls, CodecPipeline)):
        setting = _setting_name(HOST_PIPELINE_KEY)
        msg = f"zarr config {setting} must name a codec pipeline, got {path!r}"
        raise ValueError(msg) from error
    # (ours would hand reads to itself)
    return FusedCodecPipeline if issubclass(cls, KvikioCodecPipeline) else cls


# ---------------------------------------------------------------------------------------------
# The GPU read
# ---------------------------------------------------------------------------------------------

_thread_local = threading.local()

# nvCOMP's batched zstd decode needs about 3.2x the decoded size as scratch space, which the codec
# object keeps (and reuses) for as long as it lives. Decoding therefore happens in calls of at most
# this many decoded bytes, by a small pool of cached codecs per device, bounding the memory held to
# ~3.2x that per codec. Each call also costs a few milliseconds, so larger limits are faster for
# large reads. (The settings are keys of zarr's config, e.g. ZARR_KVIKIO__MAX_DECODE_BYTES in the
# environment.)
_MAX_DECODE_BYTES_KEY = "kvikio.max_decode_bytes"
_DEFAULT_MAX_DECODE_BYTES = 1 << 30
# Codecs per device. More allow decoding from several threads at once, but reading sparse matrix
# blocks with 10 threads per GPU was not faster with 2-4 of them,
# and each holds its own scratch space.
_DECODERS_KEY = "kvikio.decoders"
_DEFAULT_DECODERS = 1
# Reads of fewer bytes than this are decoded on the CPU (`_read_host`): every nvCOMP call has a
# fixed cost of a few milliseconds, more than decoding e.g. a small slice of a sparse matrix's
# `indptr` takes.
_MIN_BYTES_KEY = "kvikio.min_bytes"
_DEFAULT_MIN_BYTES = 1 << 20
# KvikIO's thread count, a setting of the whole process (KvikIO's own default is 1), set before
# reads with KvikIO (`_kvikio`; KVIKIO_NTHREADS replaces the default)
_NUM_THREADS_KEY = "kvikio.num_threads"
_DEFAULT_NUM_THREADS = 8
# How contiguous reads of zstd-compressed sharded 1-D arrays (e.g. a sparse matrix's data and
# indices) fetch the stored bytes: "host" reads them into pinned host memory and copies them to
# the GPU, which is fastest when the data comes from the page cache or GPUDirect Storage runs in
# compatibility mode; "kvikio" reads them straight into GPU memory with KvikIO (for GPUDirect
# Storage).
_FETCH_KEY = "kvikio.fetch"
_DEFAULT_FETCH = "host"
# Threads for host reads. Reads of at least `_PARALLEL_READ_BYTES` on average are spread over them;
# smaller ones are faster in one thread than handed between threads competing for the GIL.
_READ_THREADS_KEY = "kvikio.read_threads"
_DEFAULT_READ_THREADS = 16
_PARALLEL_READ_BYTES = 256 << 10
# Host fetches are staged in pinned host memory, in buffers kept for later fetches. A fetch takes
# one that fits, else one grown (in steps of `_STAGING_STEP_BYTES`) as far as all buffers together
# stay within `_STAGING_LIMIT_BYTES` (freeing kept ones if no other fetch runs, it holds less than
# half the fetch, and that lets it grow; one that holds half only grows to fit);
# fetches larger than their buffer go through it in slabs, one half read into while the other one
# is copied to the GPU (in parts of at most `_READ_PART_BYTES`, spread over the read threads).
# Slabs are slower when reading is faster than the copy (e.g. from the page cache): a fetch staged
# at once is copied while the decode is prepared.
_STAGING_STEP_BYTES = 64 << 20
_STAGING_LIMIT_BYTES = 4 << 30
_READ_PART_BYTES = 4 << 20
# The files reads keep open are limited, in file descriptors, to half the soft limit of open files
# per process (often 1024), shared by all threads (`_open_files`). A KvikIO file handle takes up to
# two; a contiguous read keeps every shard open, plus a KvikIO handle with `kvikio.fetch` "kvikio".
_CUFILE_FDS = 2
_CONTIGUOUS_FDS = {"host": 1, "kvikio": 1 + _CUFILE_FDS}
# Files a read opens at once, at most
_MAX_OPEN_FILES = 256

# (None, after `release_memory` dropped a pool: tells threads waiting on it to use the new one)
_pools: dict[int, queue.Queue[tuple[Any, Any] | None]] = {}
_pools_lock = threading.Lock()
# nvCOMP allocates and frees pinned host memory in every decode call, and freeing it
# (`cudaFreeHost`) synchronises the whole device, serialising concurrent reads: it gets the
# pinned memory from this pool instead.
_pinned_pool: list[Any] = []
# pinned host buffers for host fetches that no fetch uses now, each with the event after its last
# copy (if any), and the bytes of all buffers (also those in use)
_staging: list[tuple[np.ndarray, Any]] = []
_staging_bytes = 0
_staging_lock = threading.Lock()


def release_memory() -> None:
    """Release the scratch space held by the cached nvCOMP codecs and the pinned read buffers."""
    global _staging_bytes
    try:
        with _staging_lock:
            for buf, copied in _staging:
                if copied is not None:
                    copied.synchronize()
                _staging_bytes -= buf.nbytes
            _staging.clear()
    finally:
        with _pools_lock:
            try:
                for device_id in list(_pools):
                    pool = _pools.pop(device_id)
                    for _ in range(pool.maxsize):
                        # (waits for codecs in use)
                        codec, stream = cast("tuple[Any, Any]", pool.get())
                        stream.synchronize()
                        # the codec must go before the stream it runs on
                        # (a tuple would release its items the other way round)
                        del codec
                        del stream
                    # threads that got this pool but not yet a codec from it move on to the new one
                    pool.put(None)
            finally:
                # A Python allocator still set at exit crashes the process, so it is reset also
                # when a synchronize above failed (e.g. after a CUDA error). Otherwise no decode
                # runs now (new ones wait for the lock), and `_pool` sets it again.
                if _pinned_pool:
                    import nvidia.nvcomp as nv

                    nv.set_pinned_allocator(None)
                    _pinned_pool.pop().free_all_blocks()


# also at exit, so interpreter teardown does not release them in the wrong order
atexit.register(release_memory)


def _int_setting(key: str, default: int, minimum: int) -> int:
    """The integer setting `key` of zarr's config."""
    n = zarr.config.get(key, default)
    if type(n) is not int or n < minimum:
        msg = (
            f"zarr config {_setting_name(key)} must be an integer of at least {minimum}, got {n!r}"
        )
        raise ValueError(msg)
    return n


def _setting_name(key: str) -> str:
    """`key` and the environment variable that sets it (when zarr is imported)."""
    return f"{key!r} ({'ZARR_' + key.upper().replace('.', '__')})"


@cache
def _require_once() -> None:
    """:func:`zarr_kvikio.require`, until it passes (KvikIO and nvCOMP don't go away)."""
    require()


_kvikio_lock = threading.Lock()


def _kvikio() -> Any:
    """KvikIO, reading with ``kvikio.num_threads`` threads (unless that's None)."""
    import kvikio
    import kvikio.defaults

    if zarr.config.get(_NUM_THREADS_KEY, 0) is not None:
        default = int(os.environ.get("KVIKIO_NTHREADS", _DEFAULT_NUM_THREADS))
        n = _int_setting(_NUM_THREADS_KEY, default, 1)
        if kvikio.defaults.get("num_threads") != n:
            with _kvikio_lock:
                if kvikio.defaults.get("num_threads") != n:
                    # (KvikIO lets reads that run or are queued finish)
                    kvikio.defaults.set({"num_threads": n})
    return kvikio


class _FileBudget:
    """File descriptors that reads may keep open at once, shared by all threads."""

    def __init__(self, total: int | None = None) -> None:
        self._total = total  # (None: half the soft limit of open files, read when first needed)
        self._used = 0
        self._freed = threading.Condition()
        # reservations waiting, served in order: else small ones could keep a large one waiting
        self._queue: deque[object] = deque()

    @property
    def total(self) -> int:
        if self._total is None:
            import resource

            soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
            self._total = sys.maxsize if soft == resource.RLIM_INFINITY else max(soft // 2, 1)
        return self._total

    def files(self, fds_per_file: int) -> int:
        """How many files of `fds_per_file` file descriptors a read may open at once."""
        return max(1, min(_MAX_OPEN_FILES, self.total // fds_per_file))

    @contextmanager
    def reserve(self, fds: int) -> Iterator[None]:
        """Take `fds` file descriptors for the block, waiting until they are free.

        All are taken at once, and none is held while waiting, so reads can't deadlock.
        Reservations are served first come, first served.
        """
        fds = min(fds, self.total)  # (or a file taking more than all would wait forever)
        ticket = object()
        with self._freed:
            self._queue.append(ticket)
            try:
                self._freed.wait_for(
                    lambda: self._queue[0] is ticket and self._used + fds <= self.total
                )
                self._used += fds
            finally:
                self._queue.remove(ticket)
                self._freed.notify_all()  # (the next one may fit too)
        try:
            yield
        finally:
            with self._freed:
                self._used -= fds
                self._freed.notify_all()


_open_files = _FileBudget()


def _pool(device_id: int) -> queue.Queue[tuple[Any, Any] | None]:
    """The pool of nvCOMP zstd codecs for a device, each with the CUDA stream it runs on."""
    import cupy as cp
    import nvidia.nvcomp as nv

    with _pools_lock:
        if device_id not in _pools:
            # (with no codecs, decodes would wait for one forever)
            n = _int_setting(_DECODERS_KEY, _DEFAULT_DECODERS, 1)
            if not _pinned_pool:
                _pinned_pool.append(cp.cuda.PinnedMemoryPool())
                nv.set_pinned_allocator(lambda nbytes, *_: _pinned_pool[0].malloc(nbytes))
            pool: queue.Queue[tuple[Any, Any] | None] = queue.Queue(maxsize=n)
            for _ in range(n):
                stream = cp.cuda.Stream(non_blocking=True)
                codec = nv.Codec(
                    algorithm="Zstd",
                    bitstream_kind=nv.BitstreamKind.RAW,
                    device_id=device_id,
                    cuda_stream=stream.ptr,
                )
                pool.put((codec, stream))
            _pools[device_id] = pool
        return _pools[device_id]


def _stream(device_id: int) -> Any:
    """A per-thread, per-device CUDA stream, after the work queued on the default stream so far."""
    import cupy as cp

    streams: dict[int, Any] = _thread_local.__dict__.setdefault("streams", {})
    if device_id not in streams:
        streams[device_id] = cp.cuda.Stream(non_blocking=True)
    stream = streams[device_id]
    # zarr allocates the output from CuPy's pool for the default stream, so work queued there
    # (on an array freed since) may still use its memory
    stream.wait_event(cp.cuda.Stream.null.record())
    return stream


def _max_decode_bytes() -> int:
    return _int_setting(_MAX_DECODE_BYTES_KEY, _DEFAULT_MAX_DECODE_BYTES, 1)


def _decode(pieces: list[_Piece], stream: Any, device: Any) -> None:
    _decode_into(
        [p.data for p in pieces],
        [p.target for p in pieces],
        [p.target.nbytes for p in pieces],
        stream,
        device,
    )


def _decode_into(
    src: list[Any], dst: list[Any], sizes: list[int], stream: Any, device: Any
) -> None:
    """Decode the zstd frames `src` into `dst` (`sizes` decoded bytes each) on a pooled codec."""
    import nvidia.nvcomp as nv

    limit = _max_decode_bytes()
    groups: list[tuple[int, int]] = []
    start = size = 0
    for i, n in enumerate(sizes):
        if size and size + n > limit:
            groups.append((start, i))
            start, size = i, 0
        size += n
    groups.append((start, len(sizes)))
    pool = _pool(device.id)
    while (item := pool.get()) is None:  # the pool was released: pass that on, use the new one
        pool.put(None)
        pool = _pool(device.id)
    codec, codec_stream = item
    try:
        # the codec's stream waits for the reads/allocations queued on ours, and vice versa
        codec_stream.wait_event(stream.record())
        for a, b in groups:
            codec.decode(nv.as_arrays(src[a:b]), out=nv.as_arrays(dst[a:b]))
    finally:
        try:
            done = codec_stream.record()
        finally:
            # work queued later on the codec's stream runs after ours, so it can be reused right
            # away (and it must go back also if CUDA failed, or later decodes wait for it forever)
            pool.put((codec, codec_stream))
        stream.wait_event(done)
        # the buffers must stay alive until the decodes queued so far are done (also on errors)
        done.synchronize()


@dataclass(slots=True)
class _Piece:
    """One stored chunk (or inner chunk of a shard) to fetch and decode into ``target``."""

    path: Path
    offset: int
    nbytes: int
    target: Any  # contiguous device uint8 array receiving the decoded bytes
    data: Any = None  # device uint8 array of the stored bytes, set by `_fetch`


@dataclass(slots=True)
class _Entry:
    """How one entry of the batch gets into the output."""

    spec: ArraySpec
    chunk_selection: tuple[Any, ...]  # normalized (`_normalize`)
    out_selection: SelectorTuple
    missing: bool = False
    # Region mode: decoded bounding box of the selected part of the chunk, and its offset
    region: Any = None
    origin: tuple[int, ...] = ()
    # (destination, index, source) copies to do once everything is decoded
    copies: list[tuple[Any, Any, Any]] = field(default_factory=list)


def _read_gpu(
    layout: _Layout,
    batch: list[BatchInfo],
    paths: list[Path],
    out: Any,
    drop_axes: tuple[int, ...],
) -> tuple[GetResult, ...]:
    with out.device as device:
        stream = _stream(device.id)
        with stream:
            pieces: list[_Piece] = []
            entries = [
                _plan(layout, entry, path, out, drop_axes=drop_axes, pieces=pieces)
                for entry, path in zip(batch, paths, strict=True)
            ]
            _fetch(pieces)
            if layout.compressed and pieces:
                _decode(pieces, stream, device)
            else:
                for p in pieces:
                    if p.data.size != p.target.size:
                        msg = f"{p.path}: expected {p.target.size} bytes, found {p.data.size}"
                        raise ValueError(msg)
                    p.target[...] = p.data
            results = [_scatter(e, out, drop_axes) for e in entries]
            stream.synchronize()
    return tuple(results)


def _read_host(
    layout: _Layout,
    batch: list[BatchInfo],
    paths: list[Path],
    out: Any,
    drop_axes: tuple[int, ...],
) -> tuple[GetResult, ...]:
    """Read into host memory with plain file reads and CPU decoding, then copy to the GPU at once.

    For small reads, this is faster than a (fixed-cost) nvCOMP call, and avoids zarr's own
    pipeline, which copies every chunk to the GPU separately (and warns about each copy).
    """
    from numcodecs import Zstd

    host = np.empty(out.shape, dtype=out.dtype)
    pieces: list[_Piece] = []
    entries = [
        _plan(layout, entry, path, host, drop_axes=drop_axes, pieces=pieces)
        for entry, path in zip(batch, paths, strict=True)
    ]
    _fetch_host(pieces)
    codec = Zstd() if layout.compressed else None
    for p in pieces:
        if codec is not None:
            codec.decode(p.data, out=p.target)
        elif p.data.size != p.target.size:
            msg = f"{p.path}: expected {p.target.size} bytes, found {p.data.size}"
            raise ValueError(msg)
        else:
            p.target[...] = p.data
    results = tuple(_scatter(e, host, drop_axes) for e in entries)
    if out.flags.c_contiguous or out.flags.f_contiguous:
        out.set(host)
    else:  # (which `set` can't copy into)
        import cupy as cp

        with out.device:
            out[...] = cp.asarray(host)
    return results


# ---------------------------------------------------------------------------------------------
# Contiguous reads of sharded 1-D arrays
# ---------------------------------------------------------------------------------------------
# E.g. row ranges of a sparse matrix's `data` and `indices`, which data loaders read many of at
# once. All entries of a read are planned with numpy instead of per stored chunk, every shard
# index is read once, and the stored bytes are fetched and decoded together.


class _DevicePtr:
    """A device byte range for nvCOMP: much cheaper to make than a CuPy view."""

    __slots__ = ("__cuda_array_interface__",)

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }


def _contiguous_reads(
    layout: _Layout, batch: list[BatchInfo], out: Any, drop_axes: tuple[int, ...]
) -> bool:
    """Whether every entry is a contiguous range of a sharded, zstd-compressed 1-D array."""
    if (
        not layout.compressed
        or layout.inner_chunk_shape is None
        or len(layout.inner_chunk_shape) != 1
        or out.ndim != 1
        or drop_axes
        or not out.flags.c_contiguous
    ):
        return False
    for _, _, chunk_sel, out_sel, _ in batch:
        cs = chunk_sel[0] if isinstance(chunk_sel, tuple) else chunk_sel
        os_ = out_sel[0] if isinstance(out_sel, tuple) else out_sel
        if not (isinstance(cs, slice) and isinstance(os_, slice)):
            return False
        if cs.step not in (None, 1) or os_.step not in (None, 1):
            return False
    return True


def _read_gpu_contiguous(
    layout: _Layout, batch: list[BatchInfo], paths: list[Path], out: Any, fetch: str
) -> tuple[GetResult, ...] | None:
    """Read contiguous ranges of a sharded 1-D array.

    None if `_read_gpu` has to (missing chunks).
    """
    import cupy as cp

    assert layout.inner_chunk_shape is not None
    (n,) = layout.inner_chunk_shape
    chunk_bytes = n * out.itemsize
    fds: dict[Path, int] = {}
    # the shard files are open (and counted) only until their stored bytes are fetched
    # (closed before `_read_gpu` reads instead, if it has to)
    with _open_files.reserve(_CONTIGUOUS_FDS[fetch] * len(batch)):
        try:
            plan = _plan_contiguous(layout, batch, paths, out, fds)
            if plan is None:
                return None
            if plan.files.size == 0:
                return tuple(GetResult(status="present") for _ in batch)
            with out.device as device:
                stream = _stream(device.id)
                with stream:
                    stored, positions = _fetch_contiguous(plan, list(fds.values()), stream, fetch)
        finally:
            for fd in fds.values():
                os.close(fd)
    with out.device, stream:
        temp = cp.empty(max(len(plan.copies), 1) * n, out.dtype)
        targets = np.where(
            plan.targets >= 0,
            plan.targets,
            temp.data.ptr + (-1 - plan.targets) * chunk_bytes,
        )
        src_ptrs = (stored.data.ptr + positions).tolist()
        _decode_into(
            [_DevicePtr(p, s) for p, s in zip(src_ptrs, plan.sizes.tolist(), strict=True)],
            [_DevicePtr(int(t), chunk_bytes) for t in targets.tolist()],
            [chunk_bytes] * len(src_ptrs),
            stream,
            device,
        )
        for k, (a, b, out_a, out_b) in enumerate(plan.copies):
            out[out_a:out_b] = temp[k * n + a : k * n + b]
        stream.synchronize()
    return tuple(GetResult(status="present") for _ in batch)


@dataclass(slots=True)
class _ContiguousPlan:
    """The stored chunks to read, as arrays (one element per stored chunk)."""

    files: np.ndarray  # index into the list of open files
    offsets: np.ndarray  # byte offset in the file
    sizes: np.ndarray  # stored (compressed) bytes
    # device address to decode into: >= 0 into `out`, -1 - k into slot k of a temporary
    targets: np.ndarray
    # (start, stop) within temporary chunk k, copied to out[out_start:out_stop]
    copies: list[tuple[int, int, int, int]]


def _plan_contiguous(
    layout: _Layout,
    batch: list[BatchInfo],
    paths: list[Path],
    out: Any,
    fds: dict[Path, int],
) -> _ContiguousPlan | None:
    assert layout.inner_chunk_shape is not None
    (n,) = layout.inner_chunk_shape
    itemsize = out.itemsize
    indexes: dict[Path, np.ndarray] = {}
    file_ids: dict[Path, int] = {}
    files, offsets, sizes, targets = [], [], [], []
    copies: list[tuple[int, int, int, int]] = []
    for (_, spec, chunk_sel, out_sel, _), path in zip(batch, paths, strict=True):
        shard_len = spec.shape[0]
        cs = chunk_sel[0] if isinstance(chunk_sel, tuple) else chunk_sel
        os_ = out_sel[0] if isinstance(out_sel, tuple) else out_sel
        assert isinstance(cs, slice) and isinstance(os_, slice)  # see `_contiguous_reads`
        start, stop, _ = cs.indices(shard_len)
        out_start = os_.start or 0
        if stop <= start:
            continue
        if path not in indexes:
            try:
                fd = os.open(path, os.O_RDONLY)
            except FileNotFoundError:  # a missing shard: filled by `_read_gpu`
                return None
            fds[path] = fd
            file_ids[path] = len(file_ids)
            indexes[path] = _read_shard_index(layout, path, os.fstat(fd).st_size, (shard_len // n,))
        index = indexes[path]
        chunks = np.arange(start // n, (stop - 1) // n + 1)
        begin = chunks * n
        a = np.maximum(begin, start)
        b = np.minimum(begin + n, stop)
        stored = index[chunks]
        if (stored[:, 0] == _MAX_UINT_64).any():  # unwritten chunks: filled by `_read_gpu`
            return None
        whole = (a == begin) & (b == begin + n)
        target = np.where(whole, out.data.ptr + (out_start + a - start) * itemsize, 0)
        for k in np.flatnonzero(~whole).tolist():
            target[k] = -1 - len(copies)
            copies.append(
                (
                    int(a[k] - begin[k]),
                    int(b[k] - begin[k]),
                    int(out_start + a[k] - start),
                    int(out_start + b[k] - start),
                )
            )
        files.append(np.full(len(chunks), file_ids[path]))
        offsets.append(stored[:, 0].astype(np.int64))
        sizes.append(stored[:, 1].astype(np.int64))
        targets.append(target)
    if not files:
        empty = np.zeros(0, np.int64)
        return _ContiguousPlan(empty, empty, empty, empty, [])
    return _ContiguousPlan(
        np.concatenate(files),
        np.concatenate(offsets),
        np.concatenate(sizes),
        np.concatenate(targets),
        copies,
    )


def _fetch_contiguous(
    plan: _ContiguousPlan, fds: list[int], stream: Any, fetch: str
) -> tuple[Any, np.ndarray]:
    """The stored bytes of all planned chunks in one device buffer, and each chunk's position in it.

    Chunks that are stored next to each other are read together.
    """
    import cupy as cp

    order = np.lexsort((plan.offsets, plan.files))
    files, offsets, sizes = plan.files[order], plan.offsets[order], plan.sizes[order]
    first = np.ones(len(order), bool)
    first[1:] = (files[1:] != files[:-1]) | (offsets[1:] != offsets[:-1] + sizes[:-1])
    run = np.cumsum(first) - 1
    run_start = offsets[first]
    run_stop = np.zeros(len(run_start), np.int64)
    np.maximum.at(run_stop, run, offsets + sizes)
    run_len = run_stop - run_start
    run_pos = np.concatenate(([0], np.cumsum(run_len)[:-1]))
    total = int(run_len.sum())
    positions = np.empty(len(order), np.int64)
    positions[order] = run_pos[run] + offsets - run_start[run]
    reads = list(
        zip(
            [fds[f] for f in files[first].tolist()],
            run_start.tolist(),
            run_len.tolist(),
            run_pos.tolist(),
            strict=True,
        )
    )
    device_buf = cp.empty(total, cp.uint8)
    if fetch == "kvikio":
        _fetch_kvikio(reads, device_buf)
    else:
        _fetch_staged(reads, device_buf, stream)
    return device_buf, positions


def _fetch_mode() -> str:
    """How contiguous reads fetch the stored bytes (see `_FETCH_KEY`)."""
    if (fetch := zarr.config.get(_FETCH_KEY, _DEFAULT_FETCH)) not in _CONTIGUOUS_FDS:
        msg = f"zarr config {_setting_name(_FETCH_KEY)} must be 'host' or 'kvikio', got {fetch!r}"
        raise ValueError(msg)
    return fetch


type _Read = tuple[int, int, int, int]  # (fd, offset in the file, bytes, position in the buffer)


def _fetch_staged(reads: list[_Read], device_buf: Any, stream: Any) -> None:
    """Read into `device_buf` (asynchronously copied on `stream`) through a pinned staging buffer.

    All at once if it fits, else slab by slab: while one half of it is read into, the
    asynchronous copy out of the other half runs.
    """
    total = len(device_buf)
    staging = _take_staging(total)
    half = len(staging) // 2
    halves = staging[:half], staging[half:]
    slabs = _slabs(reads, total, half) if total > len(staging) else []
    copied: list[Any] = [None, None]  # the last copy out of each half
    waits: list[Callable[..., None]] = []

    def start(k: int) -> None:
        a, b, parts = slabs[k]
        waits.append(_fetch_host_reads(parts, halves[k % 2][: b - a], after=copied[k % 2]))

    try:
        if not slabs:
            _fetch_host_reads(reads, staging[:total])()
            device_buf.set(staging[:total], stream=stream)
        else:
            start(0)
        for k, (a, b, _) in enumerate(slabs):
            if k + 1 < len(slabs):
                start(k + 1)  # (queued behind this slab's reads, so the read threads stay busy)
            waits[k]()
            device_buf[a:b].set(halves[k % 2][: b - a], stream=stream)
            copied[k % 2] = stream.record()
    except BaseException:
        # no read may still use the buffer (or the files) when the next fetch does
        # (reads not started yet are dropped)
        for wait in waits:
            with suppress(Exception):
                wait(drop=True)
        raise
    finally:
        _give_staging(staging, stream.record())  # (for after the copies queued so far)


def _slabs(reads: list[_Read], total: int, size: int) -> list[tuple[int, int, list[_Read]]]:
    """Split `reads` covering [0, total) into slabs: (start, stop, reads at positions in the slab).

    The reads are split into parts of at most `_READ_PART_BYTES` too.
    """
    slabs: list[tuple[int, int, list[_Read]]] = [
        (a, min(a + size, total), []) for a in range(0, total, size)
    ]
    for fd, offset, length, pos in reads:
        while length:
            n = min(length, _READ_PART_BYTES - pos % _READ_PART_BYTES, size - pos % size)
            start, _, parts = slabs[pos // size]
            parts.append((fd, offset, n, pos - start))
            offset, length, pos = offset + n, length - n, pos + n
    return slabs


def _pread_into(fd: int, buf: memoryview, offset: int) -> None:
    """Fill `buf` from `fd` at `offset` (a single read returns at most ~2 GiB on Linux)."""
    while buf:
        n = os.preadv(fd, [buf], offset)
        if n == 0:
            path = Path(f"/proc/self/fd/{fd}").resolve()
            msg = f"{path}: unexpected end of file at offset {offset}"
            raise OSError(msg)
        buf, offset = buf[n:], offset + n


def _fetch_host_reads(
    reads: list[_Read], host: np.ndarray, after: Any = None
) -> Callable[..., None]:
    """Read `reads` into `host` once `after` (an event, if any) is done.

    Returns a function that waits for the reads and raises the first error once every read is
    done (`host` and the files are reused next). Large reads are queued now, spread over the read
    threads; others run in that function (the work `after` stands for runs until then), or never
    if it is called with ``drop=True``.
    """
    view = memoryview(host)

    def read(job: _Read) -> OSError | None:
        fd, start, length, pos = job
        try:
            _pread_into(fd, view[pos : pos + length], start)
        except OSError as e:
            return e
        return None

    threads = _int_setting(_READ_THREADS_KEY, _DEFAULT_READ_THREADS, 1)
    queued = None
    if threads > 1 and len(reads) > 1 and len(host) // len(reads) >= _PARALLEL_READ_BYTES:
        if after is not None:
            after.synchronize()
        queued = _read_in_threads(threads, read, reads)

    def wait(*, drop: bool = False) -> None:
        if queued is None and drop:
            return
        if queued is None and after is not None:
            after.synchronize()
        for error in list(map(read, reads) if queued is None else queued):
            if error is not None:
                raise error

    return wait


def _fetch_kvikio(reads: list[tuple[int, int, int, int]], device_buf: Any) -> None:
    kvikio = _kvikio()
    handles: dict[int, Any] = {}
    pending: list[tuple[Any, int, int, Path]] = []
    try:
        for fd, start, length, pos in reads:
            path = Path(f"/proc/self/fd/{fd}")
            if fd not in handles:
                handles[fd] = kvikio.CuFile(path, "r")
            read = handles[fd].pread(device_buf[pos : pos + length], length, file_offset=start)
            pending.append((read, length, start, path))
    finally:
        _wait_reads(pending, handles)


def _wait_reads(reads: list[tuple[Any, int, int, Path]], handles: dict[Any, Any]) -> None:
    """Wait for the KvikIO `reads` (future, bytes, offset, file), then close and forget `handles`.

    Every read is waited for, also after an error: none may outlive its file handle or buffer.
    """
    waited = 0
    try:
        for fut, nbytes, offset, path in reads:
            waited += 1
            # (GPUDirect Storage reads past the end of a file return fewer bytes without an error)
            if (got := fut.get()) != nbytes:
                msg = f"{path.resolve()}: read {got} of {nbytes} bytes at offset {offset}"
                raise OSError(msg)
    finally:
        for fut, *_ in reads[waited:]:
            with suppress(Exception):
                fut.get()
        reads.clear()
        for fh in handles.values():
            fh.close()
        handles.clear()


_executor: Any = None
_executor_threads = 0
_executor_lock = threading.Lock()


def _read_in_threads(
    threads: int, read: Callable[[_Read], OSError | None], reads: list[_Read]
) -> Iterator[OSError | None]:
    """``map(read, reads)`` in a pool of `threads` threads, queued now."""
    global _executor, _executor_threads
    with _executor_lock:
        if _executor_threads != threads:
            from concurrent.futures import ThreadPoolExecutor

            if _executor is not None:
                _executor.shutdown(wait=False)  # (its threads end once their reads are done)
            _executor = ThreadPoolExecutor(threads, thread_name_prefix="zarr-kvikio-read")
            _executor_threads = threads
        # (queued with the lock held: an executor replaced by another thread takes no more reads)
        return _executor.map(read, reads)


def _take_staging(nbytes: int) -> np.ndarray:
    """A pinned host buffer for a host fetch of `nbytes` bytes (smaller if the limit requires).

    The smallest kept one that fits, else the largest one, grown as far as the limit allows
    (counting the other kept ones if no other fetch runs and it holds less than half the fetch,
    which are then freed for it if needed, smallest first). One that holds half only grows to fit.
    """
    global _staging_bytes
    step = _STAGING_STEP_BYTES
    buf, copied = None, None
    with _staging_lock:
        _staging.sort(key=lambda kept: kept[0].nbytes)
        if _staging:
            fits = [i for i, (kept, _) in enumerate(_staging) if kept.nbytes >= nbytes]
            buf, copied = _staging.pop(fits[0] if fits else -1)
        held = 0 if buf is None else buf.nbytes
        room = _STAGING_LIMIT_BYTES - _staging_bytes + held
        # the other kept ones count as room too if no other fetch runs and the largest one holds
        # less than half the fetch: else, once fetches running at once filled the limit, later
        # ones would stream through it in many slabs (fetches running at once keep them, else
        # they would end up with the smallest buffers; and up to four slabs, through one that
        # holds half, cost less than pinning more, so that one only grows to fit)
        idle = sum(kept.nbytes for kept, _ in _staging)
        if _staging_bytes > held + idle or 2 * held >= nbytes:
            idle = 0
        # (in whole steps: the fetch rounded up, at most the room left, and at least one)
        size = max(min(-(-nbytes // step), (room + idle) // step), 1) * step
        if 2 * held >= nbytes and size < nbytes:
            size = held
        while held < size and room < size <= room + idle:  # (only as many as it takes)
            kept, kept_copied = _staging.pop(0)
            if kept_copied is not None:
                kept_copied.synchronize()  # (no copy may still read the buffer that is freed)
            _staging_bytes -= kept.nbytes
            room, idle = room + kept.nbytes, idle - kept.nbytes
            del kept, kept_copied
        if held < size:
            if copied is not None:
                copied.synchronize()  # (no copy may still read the buffer that is freed)
            _staging_bytes -= held
            buf = copied = None  # (freed before the larger one is allocated)
            buf = _empty_pinned(size)
            _staging_bytes += size
    if copied is not None:
        copied.synchronize()  # (its last copy to the GPU)
    assert buf is not None
    return buf


def _give_staging(buf: np.ndarray, copied: Any) -> None:
    """Keep a buffer from `_take_staging` for later fetches, once `copied` (an event) is done."""
    with _staging_lock:
        _staging.append((buf, copied))


def _empty_pinned(nbytes: int) -> np.ndarray:
    """Pinned host memory of exactly `nbytes` bytes, freed once dropped.

    (Not from CuPy's pinned memory pool, which rounds sizes up to powers of two and keeps them.)
    """
    import cupy as cp

    mem = cp.cuda.PinnedMemoryPointer(cp.cuda.PinnedMemory(nbytes), 0)
    return np.ndarray(nbytes, np.uint8, buffer=mem)


def _scatter(entry: _Entry, out: Any, drop_axes: tuple[int, ...]) -> GetResult:
    if entry.missing:
        out[_index(entry.out_selection, out)] = fill_value_or_default(entry.spec)
        return GetResult(status="missing")
    for dest, index, src in entry.copies:
        dest[index] = src
    if entry.region is not None:
        value = entry.region[_index(_shift(entry.chunk_selection, entry.origin), out)]
        if drop_axes:
            value = value.squeeze(axis=drop_axes)
        out[_index(entry.out_selection, out)] = value
    return GetResult(status="present")


def _plan(
    layout: _Layout,
    batch_entry: BatchInfo,
    path: Path,
    out: Any,
    *,
    drop_axes: tuple[int, ...],
    pieces: list[_Piece],
) -> _Entry:
    """Register the pieces to read for one batch entry and decide where they are decoded to.

    Contiguous selections of 1-D arrays are handled in *direct mode*: stored chunks that are
    selected as a whole are decoded straight into ``out``, the others into temporaries that are
    copied over. Everything else uses *region mode*: the stored chunks are decoded into a
    bounding box of the selection, which is then indexed into ``out``.
    """
    _, spec, chunk_sel, out_sel, _ = batch_entry
    chunk_shape = tuple(spec.shape)
    chunk_sel = _normalize(chunk_sel, chunk_shape)
    entry = _Entry(spec, chunk_sel, out_sel)
    try:
        file_size = path.stat().st_size
    except FileNotFoundError:
        entry.missing = True
        return entry

    if layout.inner_chunk_shape is None:  # the file is the one stored chunk
        stored = [((0,) * len(chunk_shape), 0, file_size)]
        stored_shape = chunk_shape
    else:
        stored_shape = layout.inner_chunk_shape
        per_shard = tuple(s // c for s, c in zip(chunk_shape, stored_shape, strict=True))
        index = _read_shard_index(layout, path, file_size, per_shard)
        touched = [
            _touched_chunks(s, c, n)
            for s, c, n in zip(chunk_sel, stored_shape, per_shard, strict=True)
        ]
        stored = [
            (
                tuple(i * c for i, c in zip(coords, stored_shape, strict=True)),
                int(index[coords][0]),
                int(index[coords][1]),
            )
            for coords in product(*touched)
        ]
    direct = _contiguous_1d(out, chunk_sel, out_sel, drop_axes)
    if direct is not None:
        _plan_direct(entry, path, stored, stored_shape, out=out, direct=direct, pieces=pieces)
    else:
        _plan_region(entry, path, stored, stored_shape, out=out, pieces=pieces)
    return entry


type _Stored = list[tuple[tuple[int, ...], int, int]]  # (start in chunk, offset, nbytes)


def _plan_direct(
    entry: _Entry,
    path: Path,
    stored: _Stored,
    stored_shape: tuple[int, ...],
    *,
    out: Any,
    direct: tuple[int, int, int],
    pieces: list[_Piece],
) -> None:
    xp = _array_module(out)
    sel_start, sel_stop, out_start = direct
    (n,) = stored_shape
    out_bytes = out.view(np.uint8)  # one byte view to slice targets from
    dtype = entry.spec.dtype.to_native_dtype()
    fill = fill_value_or_default(entry.spec)
    for (s,), offset, nbytes in stored:
        a, b = max(s, sel_start), min(s + n, sel_stop)
        if a >= b:
            continue
        dest = slice(out_start + a - sel_start, out_start + b - sel_start)
        if (offset, nbytes) == (_MAX_UINT_64, _MAX_UINT_64):  # inner chunk not written
            out[dest] = fill
            continue
        if (a, b) == (s, s + n):
            isz = out.itemsize
            target_bytes = out_bytes[dest.start * isz : dest.stop * isz]
        else:
            target = xp.empty(n, dtype)
            entry.copies.append((out, dest, target[a - s : b - s]))
            target_bytes = target.view(np.uint8)
        pieces.append(_Piece(path, offset, nbytes, target_bytes))


def _plan_region(
    entry: _Entry,
    path: Path,
    stored: _Stored,
    stored_shape: tuple[int, ...],
    *,
    out: Any,
    pieces: list[_Piece],
) -> None:
    xp = _array_module(out)
    dtype = entry.spec.dtype.to_native_dtype()
    fill = fill_value_or_default(entry.spec)
    starts = np.array([s for s, *_ in stored])
    entry.origin = tuple(int(v) for v in starts.min(axis=0))
    region_shape = tuple(
        int(hi) + c - o
        for hi, c, o in zip(starts.max(axis=0), stored_shape, entry.origin, strict=True)
    )
    entry.region = xp.empty(region_shape, dtype)
    for start, offset, nbytes in stored:
        local = tuple(
            slice(s - o, s - o + c)
            for s, o, c in zip(start, entry.origin, stored_shape, strict=True)
        )
        target = entry.region[local]
        if (offset, nbytes) == (_MAX_UINT_64, _MAX_UINT_64):  # inner chunk not written
            target[...] = fill
            continue
        if not target.flags.c_contiguous:
            tmp = xp.empty(stored_shape, dtype)
            entry.copies.append((entry.region, local, tmp))
            target = tmp
        pieces.append(_Piece(path, offset, nbytes, _as_bytes(target)))


def _as_bytes(arr: Any) -> Any:
    return arr.reshape(-1).view(np.uint8)


def _array_module(arr: Any) -> Any:
    if _is_device_array(arr):
        import cupy as cp

        return cp
    return np


def _contiguous_1d(
    out: Any,
    chunk_sel: tuple[Any, ...],
    out_sel: SelectorTuple,
    drop_axes: tuple[int, ...],
) -> tuple[int, int, int] | None:
    """(start, stop, output start) if the entry is a contiguous slice of a 1-D array."""
    if not isinstance(out_sel, tuple):
        out_sel = (out_sel,)
    if (
        len(chunk_sel) != 1
        or out.ndim != 1
        or drop_axes
        or not out.flags.c_contiguous
        or not isinstance(cs := chunk_sel[0], slice)
        or not isinstance(os_ := out_sel[0], slice)
        or cs.step != 1
        or os_.step not in (None, 1)
    ):
        return None
    return cs.start, cs.stop, os_.start or 0


def _normalize(sel: SelectorTuple, shape: tuple[int, ...]) -> tuple[Any, ...]:
    if not isinstance(sel, tuple):
        sel = (sel,)
    # boolean masks become the indices they select, which `_shift` can move into a region
    return tuple(
        slice(*s.indices(n))
        if isinstance(s, slice)
        else np.flatnonzero(s)
        if isinstance(s, np.ndarray) and s.dtype == bool
        else s
        for s, n in zip(sel, shape, strict=True)
    )


def _touched_chunks(sel: Any, chunk_len: int, n_chunks: int) -> list[int]:
    """Sorted indices of the inner chunks a per-dimension selection touches."""
    if isinstance(sel, int | np.integer):
        return [int(sel) // chunk_len]
    if isinstance(sel, slice):
        start, stop, step = sel.start, sel.stop, sel.step
        if start >= stop:
            return [min(start // chunk_len, n_chunks - 1)]
        if step == 1:
            return list(range(start // chunk_len, (stop - 1) // chunk_len + 1))
        idx = np.arange(start, stop, step)
    else:
        idx = np.asarray(sel).ravel()
        if idx.size == 0:
            return [0]
    return np.unique(idx // chunk_len).tolist()


def _shift(sel: tuple[Any, ...], origin: tuple[int, ...]) -> tuple[Any, ...]:
    """Express a selection within a chunk relative to a region starting at ``origin``."""
    shifted: list[Any] = []
    for s, o in zip(sel, origin, strict=True):
        if isinstance(s, slice):
            shifted.append(slice(s.start - o, s.stop - o, s.step))
        elif isinstance(s, int | np.integer):
            shifted.append(int(s) - o)
        else:
            shifted.append(np.asarray(s) - o)
    return tuple(shifted)


def _index(sel: Any, like: Any) -> Any:
    """An index with its arrays on the device (or host) that `like` is on."""
    if not _is_device_array(like):
        return sel
    import cupy as cp

    if isinstance(sel, tuple):
        return tuple(_index(s, like) for s in sel)
    return cp.asarray(sel) if isinstance(sel, np.ndarray) else sel


def _read_shard_index(
    layout: _Layout, path: Path, file_size: int, per_shard: tuple[int, ...]
) -> np.ndarray:
    n_bytes = 16 * prod(per_shard)
    size = n_bytes + (4 if layout.index_has_crc else 0)
    with path.open("rb") as f:
        # (a file shorter than the index reads short, rather than at a negative offset)
        raw = os.pread(f.fileno(), size, max(file_size - size, 0) if layout.index_at_end else 0)
    if len(raw) != size:
        msg = f"{path}: shard index is truncated"
        raise ValueError(msg)
    payload = raw[:n_bytes]
    if layout.index_has_crc and google_crc32c.value(payload) != int.from_bytes(
        raw[n_bytes:], "little"
    ):
        msg = f"{path}: shard index checksum mismatch"
        raise ValueError(msg)
    return np.frombuffer(payload, dtype="<u8").reshape((*per_shard, 2))


def _byte_ranges(pieces: list[_Piece]) -> Iterator[tuple[Path, int, int, list[_Piece]]]:
    """Group pieces by file and merge nearby ones into (path, start, stop, pieces) ranges."""
    by_path: dict[Path, list[_Piece]] = {}
    for p in pieces:
        by_path.setdefault(p.path, []).append(p)
    for path, group in by_path.items():
        group.sort(key=lambda p: p.offset)
        i = 0
        while i < len(group):
            j, start, stop = i + 1, group[i].offset, group[i].offset + group[i].nbytes
            while j < len(group) and group[j].offset <= stop + _MERGE_GAP:
                stop = max(stop, group[j].offset + group[j].nbytes)
                j += 1
            yield path, start, stop, group[i:j]
            i = j


def _fetch(pieces: list[_Piece]) -> None:
    """Read all pieces into device memory, one KvikIO read per contiguous byte range."""
    import cupy as cp

    kvikio = _kvikio()
    # (all ranges of a file in a row)
    files = [list(ranges) for _, ranges in groupby(_byte_ranges(pieces), key=lambda r: r[0])]
    for some_files in batched(files, _open_files.files(_CUFILE_FDS)):
        handles: dict[Path, Any] = {}
        reads: list[tuple[Any, int, int, Path]] = []
        with _open_files.reserve(_CUFILE_FDS * len(some_files)):
            try:
                for path, start, stop, group in chain.from_iterable(some_files):
                    if path not in handles:
                        handles[path] = kvikio.CuFile(path, "r")
                    buf = cp.empty(stop - start, cp.uint8)
                    read = handles[path].pread(buf, stop - start, file_offset=start)
                    reads.append((read, stop - start, start, path))
                    for p in group:
                        p.data = buf[p.offset - start : p.offset - start + p.nbytes]
            finally:
                _wait_reads(reads, handles)


def _fetch_host(pieces: list[_Piece]) -> None:
    """Read all pieces into host memory, one read per contiguous byte range."""
    for path, start, stop, group in _byte_ranges(pieces):
        buf = np.empty(stop - start, dtype=np.uint8)
        with path.open("rb") as fh:
            _pread_into(fh.fileno(), memoryview(buf), start)
        for p in group:
            p.data = buf[p.offset - start : p.offset - start + p.nbytes]
