"""Reads under faults, misconfiguration and concurrency, and the resources they hold."""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numcodecs
import numpy as np
import pytest
import zarr
from zarr.codecs import BloscCodec, ZstdCodec
from zarr.core.buffer import gpu

import zarr_kvikio
from zarr_kvikio import _pipeline

try:
    import cupy as cp
except ImportError:
    cp = None

needs_gpu = pytest.mark.skipif(
    not (cp is not None and zarr_kvikio.available() and cp.is_available()),
    reason="needs KvikIO, nvCOMP and a GPU",
)

ZSTD = (ZstdCodec(level=3),)
rng = np.random.default_rng(0)

# keeps the GPU busy for `ns` nanoseconds
_SPIN = r"""
extern "C" __global__ void spin(unsigned long long ns) {
    unsigned long long start, now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - start < ns);
}
"""
# writes far outside its buffer: an error that every later CUDA call reports too
_FAULT = r"""
extern "C" __global__ void fault(int* p) { *(volatile int*)(p + (1ull << 40)) = 1; }
"""


def write(path: Path, data: np.ndarray, chunks, *, shards=None, compressors=ZSTD, fill_value=0):
    arr = zarr.create_array(
        path,
        shape=data.shape,
        dtype=data.dtype,
        chunks=chunks,
        shards=shards,
        compressors=compressors,
        fill_value=fill_value,
    )
    arr[...] = data
    return arr


def read_gpu(path: Path, selection=slice(None)) -> np.ndarray:
    with zarr_kvikio.enable():
        return cp.asnumpy(zarr.open_array(path, mode="r")[selection])


# ---------------------------------------------------------------------------------------------
# Host reads (no GPU needed)
# ---------------------------------------------------------------------------------------------


def test_host_reads_continue_partial_reads(tmp_path, monkeypatch):
    """A single read call returns at most ~2 GiB on Linux (here: at most 7 bytes)."""
    data = rng.integers(0, 256, 1000, dtype=np.uint8)
    path = tmp_path / "f"
    path.write_bytes(data.tobytes())
    pread, preadv = os.pread, os.preadv
    monkeypatch.setattr(os, "pread", lambda fd, n, offset: pread(fd, min(n, 7), offset))
    monkeypatch.setattr(os, "preadv", lambda fd, bufs, offset: preadv(fd, [bufs[0][:7]], offset))
    eof = re.escape(f"{path}: unexpected end of file")  # (e.g. of a truncated shard)

    host = np.zeros(900, np.uint8)
    fd = os.open(path, os.O_RDONLY)
    try:
        _pipeline._fetch_host_reads([(fd, 50, 500, 0), (fd, 600, 400, 500)], host)()
        np.testing.assert_array_equal(host, np.concatenate([data[50:550], data[600:]]))
        with pytest.raises(OSError, match=eof):
            _pipeline._fetch_host_reads([(fd, 900, 200, 0)], host)()
    finally:
        os.close(fd)

    piece = _pipeline._Piece(path, 100, 800, target=None)
    _pipeline._fetch_host([piece])
    np.testing.assert_array_equal(piece.data, data[100:900])
    with pytest.raises(OSError, match=eof):
        _pipeline._fetch_host([_pipeline._Piece(path, 900, 200, target=None)])


def test_host_reads_finish_before_an_error_is_raised(tmp_path, monkeypatch):
    """The staging buffer and the files are reused by the next read: no read may still run."""
    monkeypatch.setattr(_pipeline, "_PARALLEL_READ_BYTES", 0)
    path = tmp_path / "f"
    path.write_bytes(bytes(1000))
    preadv = os.preadv
    done = []

    def slow(fd, bufs, offset):
        if offset < 1000:  # (the read past the end of the file fails at once)
            time.sleep(0.3)
        n = preadv(fd, bufs, offset)
        done.append(offset)
        return n

    monkeypatch.setattr(os, "preadv", slow)
    host = np.zeros(400, np.uint8)
    fd = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(OSError, match="end of file"):
            _pipeline._fetch_host_reads(
                [(fd, 2000, 100, 0), (fd, 0, 100, 100), (fd, 100, 100, 200), (fd, 200, 100, 300)],
                host,
            )()
        assert sorted(done) == [0, 100, 200, 2000]
    finally:
        os.close(fd)


def test_truncated_shard_index_names_the_file(tmp_path):
    layout = _pipeline._Layout(compressed=True, inner_chunk_shape=(4,))
    path = tmp_path / "shard"
    path.write_bytes(bytes(10))  # shorter than the index at its end
    with pytest.raises(ValueError, match=re.escape(f"{path}: shard index is truncated")):
        _pipeline._read_shard_index(layout, path, 10, (4,))


@pytest.fixture
def nvcomp(monkeypatch):
    """A stand-in for `nvidia.nvcomp` (also without the cuda extra)."""
    nvcomp = ModuleType("nvidia.nvcomp")
    if (nvidia := sys.modules.get("nvidia")) is None:
        monkeypatch.setitem(sys.modules, "nvidia", nvidia := ModuleType("nvidia"))
    monkeypatch.setattr(nvidia, "nvcomp", nvcomp, raising=False)
    monkeypatch.setitem(sys.modules, "nvidia.nvcomp", nvcomp)
    return nvcomp


class FakeStream:
    """Records what happens on a CUDA stream, in `log`."""

    def __init__(self, name, log, *, fail_record=False):
        self.name, self.log, self.fail_record = name, log, fail_record

    def record(self):
        if self.fail_record:
            msg = "sticky CUDA error"
            raise RuntimeError(msg)
        self.log.append(f"{self.name}.record")
        return SimpleNamespace(synchronize=lambda: self.log.append(f"{self.name}.synchronize"))

    def wait_event(self, event):
        self.log.append(f"{self.name}.wait")


@pytest.mark.parametrize("fault", ["decode", "record"])
def test_decode_errors_return_the_codec(monkeypatch, nvcomp, fault):
    """Else every later decode on the device (and `release_memory`) waits for it forever."""
    nvcomp.as_arrays = list
    monkeypatch.setenv("ZARR_KVIKIO_MAX_DECODE_BYTES", "1")  # one decode call per frame
    log = []

    def decode(src, out):
        log.append("decode")
        if fault == "decode" and log.count("decode") == 2:
            msg = "decode failed"
            raise RuntimeError(msg)

    codec = SimpleNamespace(decode=decode)
    pool = queue.Queue()
    pool.put(item := (codec, FakeStream("codec", log, fail_record=fault == "record")))
    monkeypatch.setattr(_pipeline, "_pool", lambda device_id: pool)
    with pytest.raises(RuntimeError, match="failed" if fault == "decode" else "sticky"):
        _pipeline._decode_into(
            [b"a", b"b"], [b"c", b"d"], [1, 1], FakeStream("ours", log), SimpleNamespace(id=0)
        )
    assert pool.get_nowait() == item
    if fault == "decode":
        # the frames already queued are decoded before their buffers can be freed
        assert log[-3:] == ["codec.record", "ours.wait", "codec.synchronize"]


@pytest.mark.parametrize("fails", ["staging", "codec"])
def test_release_memory_resets_the_allocator_on_errors(monkeypatch, nvcomp, fails):
    """A Python allocator left set crashes the process at exit, also after a CUDA error."""

    def fail():
        msg = "sticky CUDA error"
        raise RuntimeError(msg)

    ok, broken = SimpleNamespace(synchronize=lambda: None), SimpleNamespace(synchronize=fail)
    allocators, freed = [], []
    nvcomp.set_pinned_allocator = allocators.append
    pool = queue.Queue(maxsize=1)
    pool.put((object(), broken if fails == "codec" else ok))
    monkeypatch.setattr(_pipeline, "_pools", {0: pool})
    staged = np.empty(8), broken if fails == "staging" else ok
    monkeypatch.setattr(_pipeline, "_staging", [staged])
    monkeypatch.setattr(_pipeline, "_staging_bytes", 64)
    pinned = SimpleNamespace(free_all_blocks=lambda: freed.append(True))
    monkeypatch.setattr(_pipeline, "_pinned_pool", [pinned])
    with pytest.raises(RuntimeError, match="sticky"):
        _pipeline.release_memory()
    assert (allocators, freed, _pipeline._pinned_pool) == ([None], [True], [])


def test_kept_staging_buffers_make_room(monkeypatch):
    """Once fetches running at once filled the limit, a later fetch alone still gets a buffer
    that fits.

    Kept buffers are freed for it (smallest first, once their copies are done), but not while
    other fetches run, nor if the largest one holds at least half of it (which only grows to fit).
    """
    mib = 1 << 20
    monkeypatch.setattr(_pipeline, "_STAGING_STEP_BYTES", mib)
    monkeypatch.setattr(_pipeline, "_STAGING_LIMIT_BYTES", 10 * mib)
    monkeypatch.setattr(_pipeline, "_empty_pinned", lambda n: np.empty(n, np.uint8))
    synced = []

    def kept(n):
        return np.empty(n * mib, np.uint8), SimpleNamespace(synchronize=lambda: synced.append(n))

    monkeypatch.setattr(_pipeline, "_staging", [kept(3), kept(1), kept(3), kept(3)])
    monkeypatch.setattr(_pipeline, "_staging_bytes", 10 * mib)
    buf = _pipeline._take_staging(7 * mib - 100)
    assert (buf.nbytes, synced) == (7 * mib, [1, 3, 3])  # (1 and 3 MiB freed, a 3 MiB one grown)
    assert [kept.nbytes for kept, _ in _pipeline._staging] == [3 * mib]
    _pipeline._give_staging(buf, None)
    buf = _pipeline._take_staging(14 * mib)  # (it holds half: none freed, it streams through)
    assert (buf.nbytes, synced) == (7 * mib, [1, 3, 3])
    _pipeline._give_staging(buf, None)
    buf = _pipeline._take_staging(15 * mib)  # (more than the limit)
    assert (buf.nbytes, synced) == (10 * mib, [1, 3, 3, 3])
    assert (_pipeline._staging, _pipeline._staging_bytes) == ([], 10 * mib)
    # none is freed while other fetches run (here with 6 MiB)
    monkeypatch.setattr(_pipeline, "_staging", [kept(2), kept(1), kept(1)])
    assert _pipeline._take_staging(5 * mib).nbytes == 2 * mib
    assert [kept.nbytes for kept, _ in _pipeline._staging] == [mib, mib]
    # one that holds half only grows to fit (here with 4 MiB of room)
    monkeypatch.setattr(_pipeline, "_staging", [kept(6)])
    monkeypatch.setattr(_pipeline, "_staging_bytes", 6 * mib)
    buf = _pipeline._take_staging(11 * mib)
    assert buf.nbytes == 6 * mib
    _pipeline._give_staging(buf, None)
    assert _pipeline._take_staging(10 * mib).nbytes == 10 * mib


def test_serial_host_reads_overlap_the_copies(tmp_path, monkeypatch):
    """With one read thread, a slab is read while the copy out of the one before runs.

    After an error, the reads of slabs not started yet are dropped.
    """
    slab = 64
    monkeypatch.setenv("ZARR_KVIKIO_READ_THREADS", "1")
    monkeypatch.setattr(_pipeline, "_STAGING_STEP_BYTES", 2 * slab)
    monkeypatch.setattr(_pipeline, "_STAGING_LIMIT_BYTES", 2 * slab)  # (a half for each slab)
    monkeypatch.setattr(_pipeline, "_READ_PART_BYTES", slab // 2)
    monkeypatch.setattr(_pipeline, "_staging", [])
    monkeypatch.setattr(_pipeline, "_staging_bytes", 0)
    monkeypatch.setattr(_pipeline, "_empty_pinned", lambda n: np.empty(n, np.uint8))
    log = []

    class Device(np.ndarray):  # (stands in for a CuPy array)
        def set(self, host, stream):
            self[...] = host
            log.append("copy")

    class Stream:
        def record(self):
            n = log.count("copy")
            return SimpleNamespace(synchronize=lambda: log.append(f"copied {n}"))

    preadv = os.preadv

    def logged(fd, bufs, offset):
        log.append(f"read {offset // slab}")
        return preadv(fd, bufs, offset)

    monkeypatch.setattr(os, "preadv", logged)
    data = rng.integers(0, 256, 4 * slab, dtype=np.uint8)
    path = tmp_path / "f"
    for size in (4 * slab, slab + 40):  # (then a file ending in slab 1)
        path.write_bytes(data[:size].tobytes())
        out = np.zeros(4 * slab, np.uint8).view(Device)
        log.clear()
        fd = os.open(path, os.O_RDONLY)
        try:
            if size < len(out):
                with pytest.raises(OSError, match="end of file"):
                    _pipeline._fetch_staged([(fd, 0, len(out), 0)], out, Stream())
                assert "read 2" not in log
            else:
                _pipeline._fetch_staged([(fd, 0, len(out), 0)], out, Stream())
                np.testing.assert_array_equal(out, data)
                parts = [[f"read {k}"] * 2 for k in range(4)]
                assert log == [
                    *parts[0], "copy", *parts[1], "copy",
                    "copied 1", *parts[2], "copy", "copied 2", *parts[3], "copy",
                ]  # fmt: skip
        finally:
            os.close(fd)
    assert len(_pipeline._staging) == 1  # (given back also after the error)


def test_file_budget_serves_in_order():
    """A large reservation waiting is not passed over by smaller ones that would fit."""
    budget, order = _pipeline._FileBudget(4), []

    def reserve(fds):
        with budget.reserve(fds):
            order.append(fds)

    with ThreadPoolExecutor(2) as pool, budget.reserve(3):
        pool.submit(reserve, 4)
        while len(budget._queue) < 1 and not order:  # (waiting now)
            time.sleep(0.01)
        pool.submit(reserve, 1)
        while len(budget._queue) < 2 and not order:  # (or it got through)
            time.sleep(0.01)
        assert order == []
    assert order == [4, 1]


def test_read_threads_setting_replaces_the_executor(tmp_path, monkeypatch):
    """The replaced executor is shut down, so its threads end."""
    monkeypatch.setattr(_pipeline, "_PARALLEL_READ_BYTES", 0)
    monkeypatch.setattr(_pipeline, "_executor", None)
    monkeypatch.setattr(_pipeline, "_executor_threads", 0)
    path = tmp_path / "f"
    path.write_bytes(bytes(100))
    executors = []
    fd = os.open(path, os.O_RDONLY)
    try:
        for threads in ("2", "3"):
            monkeypatch.setenv("ZARR_KVIKIO_READ_THREADS", threads)
            reads = [(fd, 0, 50, 0), (fd, 50, 50, 50)]
            _pipeline._fetch_host_reads(reads, np.zeros(100, np.uint8))()
            executors.append(_pipeline._executor)
    finally:
        os.close(fd)
    old, new = executors
    new.shutdown()
    assert old is not new
    with pytest.raises(RuntimeError, match="shutdown"):
        old.submit(int)


@pytest.mark.parametrize(
    ("value", "minimum", "match"),
    [("1MiB", None, "must be an integer, got '1MiB'"), ("0", 1, "must be at least 1, got 0")],
)
def test_bad_integer_settings_name_the_variable(monkeypatch, value, minimum, match):
    monkeypatch.setenv("ZARR_KVIKIO_SETTING", value)
    with pytest.raises(ValueError, match=f"ZARR_KVIKIO_SETTING {match}"):
        _pipeline._env_int("ZARR_KVIKIO_SETTING", 1, minimum)


# ---------------------------------------------------------------------------------------------
# GPU reads
# ---------------------------------------------------------------------------------------------


@needs_gpu
@pytest.mark.parametrize(
    ("name", "value"),
    [("ZARR_KVIKIO_DECODERS", "0"), ("ZARR_KVIKIO_MIN_BYTES", "1e6"), ("ZARR_KVIKIO_FETCH", "gds")],
)
def test_bad_settings_raise(tmp_path, monkeypatch, name, value):
    """Instead of e.g. waiting forever for one of no codecs."""
    zarr_kvikio.release_memory()  # (the codecs are made with the settings)
    write(tmp_path / "a.zarr", rng.random(1000).astype("float32"), (100,), shards=(500,))
    monkeypatch.setenv("ZARR_KVIKIO_MIN_BYTES", "0")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=f"{name} must be"):
        read_gpu(tmp_path / "a.zarr")


@needs_gpu
@pytest.mark.parametrize(
    ("shards", "unwritten"),
    [(None, False), ((1 << 15,), False), ((1 << 15,), True)],
    ids=["unsharded", "sharded", "sharded-unwritten-chunk"],
)
def test_reads_wait_for_the_default_stream(tmp_path, monkeypatch, shards, unwritten):
    """zarr allocates the output on the default stream, where queued work may still use it."""
    monkeypatch.setenv("ZARR_KVIKIO_MIN_BYTES", "0")
    data = rng.random(1 << 16).astype("float32")
    if unwritten:  # (`_read_gpu` fills it in `out` before reading anything)
        data[: 1 << 13] = 0
    write(tmp_path / "a.zarr", data, (1 << 13,), shards=shards)
    spin = cp.RawKernel(_SPIN, "spin")
    with zarr_kvikio.enable():
        arr = zarr.open_array(tmp_path / "a.zarr", mode="r")
        out = cp.empty(data.shape, data.dtype)
        arr.get_basic_selection(out=gpu.NDBuffer(out))  # (makes the codecs, fills memory pools)
        out.fill(-1)  # (loads the kernel now: loading it while the spin runs waits for that)
        spin((1,), (1,), (np.uint64(300_000_000),))  # 0.3 s
        out.fill(-1)  # still queued on the default stream when the read starts
        arr.get_basic_selection(out=gpu.NDBuffer(out))
    np.testing.assert_array_equal(cp.asnumpy(out), data)


@needs_gpu
@pytest.mark.parametrize("fetch", ["host", "kvikio"])
@pytest.mark.parametrize("shards", [None, (100,)], ids=["unsharded", "sharded"])
def test_open_files_are_bounded(tmp_path, monkeypatch, fetch, shards):
    """Reads running at once keep at most the process's budget of file descriptors open."""
    import kvikio

    budget = 6
    monkeypatch.setattr(_pipeline, "_open_files", _pipeline._FileBudget(budget))
    monkeypatch.setenv("ZARR_KVIKIO_MIN_BYTES", "0")
    monkeypatch.setenv("ZARR_KVIKIO_FETCH", fetch)
    lock, fds, opened = threading.Lock(), set(), [0, 0]  # (open now, at most)

    def count(n):
        with lock:
            opened[0] += n
            opened[1] = max(opened)

    class CuFile(kvikio.CuFile):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            count(_pipeline._CUFILE_FDS)

        def close(self):
            super().close()
            count(-_pipeline._CUFILE_FDS)

    os_open, os_close = os.open, os.close

    def open_(path, *args, **kwargs):
        fd = os_open(path, *args, **kwargs)
        if str(path).startswith(str(tmp_path)):
            fds.add(fd)
            count(1)
        return fd

    def close(fd):
        with lock:
            ours = fd in fds
            fds.discard(fd)
        os_close(fd)
        if ours:
            count(-1)

    monkeypatch.setattr(kvikio, "CuFile", CuFile)
    monkeypatch.setattr(os, "open", open_)
    monkeypatch.setattr(os, "close", close)
    data = rng.random(4000).astype("float32")
    data[500:600] = 7  # a shard (or chunks) not written: read by `_read_gpu`
    write(tmp_path / "a.zarr", data, (25 if shards else 50,), shards=shards, fill_value=7)
    sels = [slice(i * 100 + 10, i * 100 + 3000) for i in range(8)]
    with zarr_kvikio.enable(), ThreadPoolExecutor(len(sels)) as pool:
        arr = zarr.open_array(tmp_path / "a.zarr", mode="r")
        results = list(pool.map(lambda sel: cp.asnumpy(arr[sel]), sels))
    for sel, result in zip(sels, results, strict=True):
        np.testing.assert_array_equal(result, data[sel])
    assert 0 < opened[1] <= budget
    assert opened[0] == 0


@needs_gpu
@pytest.mark.parametrize("fetch", ["host", "kvikio"])
def test_contiguous_reads_close_files_before_decoding(tmp_path, monkeypatch, fetch):
    """The shard files (and their file descriptor budget) are needed only until they are read."""
    monkeypatch.setenv("ZARR_KVIKIO_MIN_BYTES", "0")
    monkeypatch.setenv("ZARR_KVIKIO_FETCH", fetch)
    monkeypatch.setattr(_pipeline, "_open_files", budget := _pipeline._FileBudget(64))
    decode, held = _pipeline._decode_into, []

    def spy(*args):
        files = 0
        for fd in Path("/proc/self/fd").iterdir():
            with suppress(OSError):  # (e.g. the one listing them, closed by now)
                files += str(fd.readlink()).startswith(str(tmp_path))
        held.append((budget._used, files))
        return decode(*args)

    monkeypatch.setattr(_pipeline, "_decode_into", spy)
    data = rng.random(4000).astype("float32")
    write(tmp_path / "a.zarr", data, (25,), shards=(1000,))
    np.testing.assert_array_equal(read_gpu(tmp_path / "a.zarr", slice(10, 3990)), data[10:3990])
    assert held
    assert set(held) == {(0, 0)}


@needs_gpu
@pytest.mark.parametrize("fault", ["short", "error"])
@pytest.mark.parametrize("shards", [None, (250,)], ids=["unsharded", "sharded"])
def test_kvikio_read_faults(tmp_path, monkeypatch, fault, shards):
    """A read failing or returning fewer bytes raises, once every other read is done."""
    import kvikio

    monkeypatch.setenv("ZARR_KVIKIO_MIN_BYTES", "0")
    monkeypatch.setenv("ZARR_KVIKIO_FETCH", "kvikio")
    events, reads = [], []

    class Future:
        def __init__(self, fut, first):
            self.fut, self.first = fut, first

        def get(self):
            n = self.fut.get()
            events.append("get")
            if self.first and fault == "error":
                msg = "injected"
                raise RuntimeError(msg)
            return n - 1 if self.first else n

    class CuFile(kvikio.CuFile):
        def pread(self, *args, **kwargs):
            reads.append(args)
            return Future(super().pread(*args, **kwargs), first=len(reads) == 1)

        def close(self):
            events.append("close")
            super().close()

    monkeypatch.setattr(kvikio, "CuFile", CuFile)
    data = rng.random(1000).astype("float32")
    write(tmp_path / "a.zarr", data, (50,), shards=shards)
    error, match = (RuntimeError, "injected") if fault == "error" else (OSError, r"read \d+ of")
    with pytest.raises(error, match=match):
        read_gpu(tmp_path / "a.zarr")
    assert events.count("get") == len(reads) == (4 if shards else 20)
    assert "get" not in events[events.index("close") :]


@needs_gpu
def test_release_memory_while_waiting_for_a_codec(monkeypatch):
    """A decode that got the pool `release_memory` drops, but no codec from it, uses the new one."""
    zarr_kvikio.release_memory()
    device = cp.cuda.Device(0)
    values = np.arange(256, dtype=np.uint8)
    src = cp.asarray(np.frombuffer(numcodecs.Zstd().encode(values), np.uint8))
    dst = cp.zeros(256, cp.uint8)
    old = _pipeline._pool(device.id)
    codec = old.get()  # (in use)
    release = threading.Thread(target=zarr_kvikio.release_memory)
    release.start()
    while device.id in _pipeline._pools:
        time.sleep(0.01)
    time.sleep(0.1)  # release_memory waits for the codec in use

    pool = _pipeline._pool
    pools = [old]
    monkeypatch.setattr(_pipeline, "_pool", lambda i: pools.pop() if pools else pool(i))
    with device:
        stream = _pipeline._stream(device.id)
    decode = threading.Thread(
        target=_pipeline._decode_into, args=([src], [dst], [256], stream, device), daemon=True
    )
    decode.start()
    time.sleep(0.1)  # waits for a codec too
    old.put(codec)  # which release_memory takes
    release.join(10)
    decode.join(10)
    assert not decode.is_alive()
    np.testing.assert_array_equal(cp.asnumpy(dst), values)
    zarr_kvikio.release_memory()


@needs_gpu
@pytest.mark.parametrize("compressors", [ZSTD, (BloscCodec(),)], ids=["zstd", "blosc"])
@pytest.mark.parametrize("device", [0, 1])
def test_small_reads_into_strided_out(tmp_path, monkeypatch, compressors, device):
    """Reads decoded on the host (here a column) into an output on any GPU."""
    if device >= cp.cuda.runtime.getDeviceCount():
        pytest.skip("needs two GPUs")
    data = rng.random((1000, 8)).astype("float32")
    write(tmp_path / "a.zarr", data, (100, 8), compressors=compressors)
    with cp.cuda.Device(device):
        buf = cp.zeros((1000, 8), "float32")
        out = gpu.NDBuffer(buf[:, 0])  # (copied if made on another device)
    # the host data goes to the output's device (also where GPUs could reach each other's memory)
    asarray, devices = cp.asarray, []
    monkeypatch.setattr(
        cp, "asarray", lambda *a, **k: devices.append(cp.cuda.Device().id) or asarray(*a, **k)
    )
    with zarr_kvikio.enable():
        arr = zarr.open_array(tmp_path / "a.zarr", mode="r")
        arr.get_basic_selection((slice(None), 3), out=out)
    np.testing.assert_array_equal(cp.asnumpy(buf[:, 0]), data[:, 3])
    assert set(devices) == {device}


@needs_gpu
@pytest.mark.parametrize("limit", [0, 10 << 20], ids=["streamed", "at-once"])
def test_host_fetch_staging(tmp_path, monkeypatch, limit):
    """The pinned staging buffers take at most the limit together.

    A fetch larger than its buffer streams through it, reusing each half of it once its copy to the
    GPU is done.
    """
    monkeypatch.setattr(_pipeline, "_STAGING_STEP_BYTES", 2 << 20)
    monkeypatch.setattr(_pipeline, "_STAGING_LIMIT_BYTES", limit)
    monkeypatch.setattr(_pipeline, "_READ_PART_BYTES", 256 << 10)
    monkeypatch.setattr(_pipeline, "_staging", [])
    monkeypatch.setattr(_pipeline, "_staging_bytes", 0)
    data = rng.integers(0, 256, 5 << 20, dtype=np.uint8)
    path = tmp_path / "f"
    path.write_bytes(data.tobytes())
    streams = cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)
    device_bufs = cp.zeros((2, (5 << 20) - 200), cp.uint8)
    with streams[0]:
        cp.RawKernel(_SPIN, "spin")((1,), (1,), (np.uint64(200_000_000),))  # delays the copies
    fd = os.open(path, os.O_RDONLY)
    try:
        # two runs (e.g. in two shards), then the other way round: the second fetch reuses the
        # buffer once the first one's copies are done
        a, b = (fd, 1 << 20, 3 << 20), (fd, 100, (2 << 20) - 200)
        _pipeline._fetch_staged([(*a, 0), (*b, 3 << 20)], device_bufs[0], streams[0])
        _pipeline._fetch_staged([(*b, 0), (*a, (2 << 20) - 200)], device_bufs[1], streams[1])
        for stream in streams:
            stream.synchronize()
    finally:
        os.close(fd)
    a_data, b_data = data[1 << 20 : 4 << 20], data[100 : (2 << 20) - 100]
    np.testing.assert_array_equal(cp.asnumpy(device_bufs[0]), np.concatenate([a_data, b_data]))
    np.testing.assert_array_equal(cp.asnumpy(device_bufs[1]), np.concatenate([b_data, a_data]))
    held = (6 << 20) if limit else (2 << 20)  # (grown for the fetch, or the smallest)
    [(staged, _)] = _pipeline._staging  # (kept for later fetches)
    assert staged.nbytes == _pipeline._staging_bytes == held
    assert cp.cuda.runtime.pointerGetAttributes(staged.ctypes.data).type == 1  # pinned
    # a fetch while that buffer is in use gets another one, as large as the limit allows
    taken, other = _pipeline._take_staging(5 << 20), _pipeline._take_staging(5 << 20)
    assert (taken.nbytes, other.nbytes) == (held, max(limit - held, 2 << 20))


@needs_gpu
@pytest.mark.parametrize("fault", [False, True], ids=["clean", "after-a-cuda-error"])
def test_exit_after_gpu_reads(tmp_path, fault):
    """nvCOMP's pinned allocator is reset at exit: one left set to a Python function crashed it.

    Also when releasing the read buffers at exit fails.
    """
    data = rng.random(1 << 16).astype("float32")
    write(tmp_path / "a.zarr", data, (1 << 13,), shards=(1 << 15,))
    script = f"""
import cupy as cp, zarr, zarr_kvikio
with zarr_kvikio.enable():
    assert float(zarr.open_array({str(tmp_path / "a.zarr")!r}, mode="r")[:].sum()) > 0
if {fault}:
    cp.RawKernel({_FAULT!r}, "fault")((1,), (1,), (cp.zeros(1, cp.int32),))
    try:
        cp.cuda.runtime.deviceSynchronize()
    except cp.cuda.runtime.CUDARuntimeError:
        pass
"""
    env = {**os.environ, "ZARR_KVIKIO_MIN_BYTES": "0"}
    proc = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    # (Python 3.12 puts a colon after "callback")
    assert bool(re.search("in atexit callback:? <function release_memory", proc.stderr)) == fault
