"""nvCOMP's batched Zstd C API for reads planned as device pointer vectors."""

from __future__ import annotations

import ctypes as ct
import importlib
from functools import cache
from typing import Any

import numpy as np


class _Options(ct.Structure):
    # nvcompBatchedZstdDecompressOpts_t in nvcomp/zstd.h, using the default backend.
    _fields_ = (("backend", ct.c_int), ("reserved", ct.c_char * 60))


@cache
def batch_api(device_id: int) -> Any:
    """Return a compatible nvCOMP library for the current device, or None."""
    try:
        lib = importlib.import_module("nvidia.libnvcomp").load_library()
        ptr, size = ct.c_void_p, ct.c_size_t
        lib.nvcompGetProperties.argtypes = (ptr,)
        properties = (ct.c_uint32 * 2)()  # nvcompProperties_t (nvCOMP version, CUDA version)
        if lib.nvcompGetProperties(properties) or properties[0] // 1000 != 5:
            return None
        lib.nvcompBatchedZstdDecompressGetRequiredAlignments.argtypes = (_Options, ptr)
        alignment = (size * 3)()  # input, output, scratch
        if lib.nvcompBatchedZstdDecompressGetRequiredAlignments(_Options(), alignment):
            return None
        # Shard frames and destinations within a partially selected array can start at any byte.
        if list(alignment[:2]) != [1, 1] or not 0 < alignment[2] <= 256:
            return None
        query = (size, size, _Options, ct.POINTER(size), size)
        decompress = (ptr, ptr, ptr, ptr, size, ptr, size, ptr, _Options, ptr, ptr)
        lib.nvcompBatchedZstdDecompressGetTempSizeAsync.argtypes = query
        lib.nvcompBatchedZstdDecompressAsync.argtypes = decompress
    except (AttributeError, ImportError, OSError):
        return None
    return lib


def decode(
    lib: Any,
    sources: np.ndarray,
    sizes: np.ndarray,
    targets: np.ndarray,
    frame_bytes: int,
    limit: int,
    stream: Any,
) -> None:
    """Decode into caller-owned addresses on the current stream, reusing scratch across batches."""
    import cupy as cp

    batch = min(len(sizes), max(1, limit // frame_bytes))
    required = ct.c_size_t()
    while True:
        status = lib.nvcompBatchedZstdDecompressGetTempSizeAsync(
            batch, frame_bytes, _Options(), ct.byref(required), batch * frame_bytes
        )
        if status:
            raise RuntimeError(f"nvCOMP Zstd workspace query returned {status}")
        try:
            workspace = cp.empty(required.value + 255, cp.uint8)
            break
        except MemoryError:
            if batch == 1:
                raise
            batch //= 2
    scratch = (workspace.data.ptr + 255) & ~255
    try:
        for start in range(0, len(sizes), batch):
            part = slice(start, start + batch)
            n = len(sizes[part])
            rows = (sources[part], sizes[part], targets[part], np.full(n, frame_bytes))
            vectors = cp.asarray(np.stack(rows, dtype=np.uintp, casting="unsafe"))
            src, compressed, dst, capacities = (row.data.ptr for row in vectors)
            actual = cp.empty(n, cp.uintp)
            statuses = cp.empty(n, cp.int32)
            status = lib.nvcompBatchedZstdDecompressAsync(
                src,
                compressed,
                capacities,
                actual.data.ptr,
                n,
                scratch,
                required.value,
                dst,
                _Options(),
                statuses.data.ptr,
                stream.ptr,
            )
            if status:
                raise RuntimeError(f"nvCOMP Zstd launch returned {status}")
            if bool(cp.any((statuses != 0) | (actual != frame_bytes))):
                raise ValueError("a stored Zstd frame failed to decode to the expected size")
    finally:
        # All buffers above must outlive enqueued work, including when a later call fails.
        stream.synchronize()
