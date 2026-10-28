# zarr-kvikio

A [zarr-python](https://github.com/zarr-developers/zarr-python) codec pipeline that reads chunks from a local file system
straight into GPU memory with [KvikIO](https://github.com/rapidsai/kvikio) and decodes them with [nvCOMP](https://developer.nvidia.com/nvcomp).
Reads it can't do on the GPU fall back to zarr's own pipelines, so any array can be read.

Status: beta. Used by [rapids-singlecell](https://rapids-singlecell.scverse.org/)'s out-of-core reading (`rapids_singlecell.io`).

## Installation

```bash
pip install "zarr-kvikio[cuda13]"  # or [cuda12]; add ,dask for zarr_kvikio.dask
```

Without the `cuda12`/`cuda13` extra, zarr-kvikio installs without the GPU stack (KvikIO, nvCOMP, CuPy):
it can then check and convert stores, and `zarr_kvikio.enable()` explains what is missing.

## Quick start

```python
import zarr
import zarr_kvikio

zarr_kvikio.enable()  # or only within a block: with zarr_kvikio.enable(): ...
arr = zarr.open_array("data.zarr", mode="r")  # the pipeline is bound when the array is opened
x = arr[:]  # cupy.ndarray
```

With Dask, one call switches the whole cluster: the workers read into GPU memory,
while the client keeps reading into host memory (e.g. AnnData's `obs` and `var`):

```python
import zarr_kvikio.dask

# or only within a block (undone on the workers too): with zarr_kvikio.dask.enable(client): ...
zarr_kvikio.dask.enable(client)
# blocks arrive as cupyx matrices
X = anndata.experimental.read_elem_lazy(zarr.open_group("data.zarr")["X"])
```

## API

| | |
|---|---|
| `enable(num_threads=8)` | Read zarr arrays into GPU memory in this process (also as a context manager). |
| `dask.enable(client, num_threads=8)` | The same on every worker of a Dask cluster, including workers started later. |
| `available()`, `require()` | Whether KvikIO and nvCOMP are installed; raise with install instructions if not. |
| `check()` | KvikIO, nvCOMP, zarr and CuPy versions, and whether GPUDirect Storage is active. |
| `describe(array)` | An array's layout and what keeps it from being read fast on the GPU. |
| `convert_array(src, dst_group, name)` | Stream a 1-D or 2-D array into the layout below, shard by shard, and verify the copy. |
| `layout(shape, itemsize)` | The chunk and shard shape `convert_array` uses. |
| `cpu_pipeline()` | Open (source) arrays with host buffers and a fast CPU pipeline (zarrs if installed). |
| `release_memory()` | Free the GPU scratch space held by the cached nvCOMP codecs. |
| `KvikioCodecPipeline` | The codec pipeline, registered with zarr as `"zarr_kvikio.KvikioCodecPipeline"`. |

## Settings

- **`num_threads`** of `enable()` / `dask.enable()` (default 8): KvikIO itself reads with a single thread by default.
- **`ZARR_KVIKIO_MAX_DECODE_BYTES`** (default 1 GiB): the most decoded bytes per nvCOMP call.
  nvCOMP's batched zstd decode needs about **3.2× the decoded size as scratch space**,
  and its codec keeps that memory for reuse, so this bounds the device memory held (≈3.4 GB by default).
  Each call also costs a few milliseconds, so a larger limit speeds up very large reads.
- **`ZARR_KVIKIO_MIN_BYTES`** (default 1 MiB): smaller reads are read and zstd-decoded on the CPU by zarr-kvikio itself
  and copied to the GPU at once.
  Every nvCOMP call has a fixed cost of a few milliseconds, more than decoding e.g. a small slice of a sparse matrix's `indptr`.
- **`ZARR_KVIKIO_DECODERS`** (default 1): nvCOMP codecs per GPU, each with its own scratch space.
  More let several threads decode at once; for sparse-matrix reads from 10 threads per GPU, 2-4 were not faster.
  `zarr_kvikio.release_memory()` frees that scratch space.
- **`ZARR_KVIKIO_FETCH`** (default `host`): how contiguous reads of sharded 1-D arrays (e.g. a sparse matrix's `data` and
  `indices`) fetch the stored bytes. `host` reads them into pinned host memory and copies them to the GPU at once, which is
  fastest when the data comes from the page cache or GPUDirect Storage runs in compatibility mode; `kvikio` reads them
  straight into GPU memory with KvikIO (for GPUDirect Storage).
- **`ZARR_KVIKIO_READ_THREADS`** (default 16): threads for those host reads.
- If KvikIO isn't using GPUDirect Storage (`zarr_kvikio.check()`, or `gdscheck -p`), it reads through the page cache and
  a bounce buffer. That still works (and is fast for cached data), but isn't direct NVMe to GPU DMA.

## What takes the fast path

A read uses KvikIO + nvCOMP when all of these hold:

- the output is a GPU buffer (`zarr_kvikio.enable()`, or `zarr.config.enable_gpu()` with the pipeline selected),
- the array lives in a `zarr.storage.LocalStore`,
- the array is zarr v3 with a fixed-size numeric dtype, and
- its codecs are `bytes` (little endian), optionally followed by `zstd` (without checksum),
  either directly or as the inner codecs of the `sharding_indexed` codec
  (index codecs `bytes` + optional `crc32c`, index at the start or end).

Everything else, including reads into host buffers and all writes, is handled by the pipeline zarr was configured with
before `enable()` (e.g. zarrs), or else zarr's `FusedCodecPipeline`. Reads into GPU buffers that the GPU can't decode
(e.g. blosc) are decoded on the CPU and then copied.

How a fast read works:

1. The shard indices are read on the host (a few bytes per inner chunk; the `crc32c` is verified).
2. The stored bytes of every needed chunk are read with KvikIO (or into pinned host memory, see `ZARR_KVIKIO_FETCH`),
   one read per contiguous byte range (so typically one read per shard).
3. The chunks of the request are decoded in batched nvCOMP calls of up to `ZARR_KVIKIO_MAX_DECODE_BYTES` each (usually one call).
4. The decoded values are copied into the output buffer.
   For 1-D arrays, chunks that are selected as a whole are decoded directly into the output.

## Writing stores for the GPU

- **Keep zstd frames (= inner chunks) small, around 256 KiB.** nvCOMP decodes each zstd frame with a single thread block,
  so one large frame is slow no matter how fast the GPU is: a lone 1 MiB frame takes ~20 ms, an 8 MiB frame ~150 ms.
  Measured on an RTX PRO 6000 (decode time, real float32 data):

  | frame size | 16 MiB read | 256 MiB read | 2 GiB read |
  |---|---|---|---|
  | 64 KiB | 3.4 ms | 22 ms | 213 ms |
  | 256 KiB | 6.3 ms | 12 ms | 60 ms |
  | 1 MiB | 22 ms | 27 ms | 42 ms |

  This applies to *every* array a read touches, e.g. also the `indptr` of a sparse matrix.
- **Shard** so that a read opens a few files instead of one per chunk: without sharding, small reads are dominated by per-file overhead.
- **Store the chunks of every shard in order**, so a contiguous range is a single read. zarr's fused pipeline does;
  zarrs (faster at writing) stores them in whatever order they were compressed in.

`convert_array` writes this layout (zstd, ~256 KiB chunks: whole rows of a 2-D array or row pieces, 512 chunks per
shard, in order; optionally with a smaller dtype, e.g. uint16 indices), and `describe` reports what an existing array
lacks. For whole AnnData stores, use [`rapids_singlecell.io.convert_zarr`](https://rapids-singlecell.scverse.org/),
which converts every matrix this way and copies everything else.

## Benchmark

Row slices of a CSR matrix (the first 1M cells of a 11.4M × 45k single-cell atlas, float32 data, int32 indices)
read with `anndata.io.sparse_dataset(...)[start:stop]` in zarr GPU mode (`zarr.config.enable_gpu()`), producing a `cupyx` CSR matrix.
RTX PRO 6000 Blackwell, data in the page cache, KvikIO in compat mode, `KVIKIO_NTHREADS=16`.
Best of 5, see `benchmarks/bench_anndata.py`.

| rows | nnz | zarr default pipeline (CPU decode) | zarr-kvikio | zarr-kvikio, 4 GiB decode limit |
|---|---|---|---|---|
| 20k | 73M | 0.134 s | **0.062 s** | 0.062 s |
| 20k (offset 300k) | 33M | 0.071 s | **0.047 s** | 0.046 s |
| 200k | 624M | 1.170 s | **0.303 s** | 0.227 s |
| 500k | 1.1B | 2.109 s | **0.515 s** | 0.406 s |

Store layout: zstd level 3, sharded with 1 MiB inner chunks and 128 inner chunks per shard
(measured before the frame-size findings above). For comparison, zarr's default pipeline on the original layout
(blosc-lz4, int64 indices, one 609k-element chunk per file) took 0.10 s / 0.75 s / ~1.3 s for 20k / 200k / 500k rows.

## Limitations

- Local files only (`LocalStore`); remote and in-memory stores use the fallback.
- zstd (without checksum) or no compression only. Blosc, which many existing stores use, cannot be decoded by nvCOMP.
- nvCOMP's Python API costs a few tens of milliseconds per `decode` call and blocks the calling thread,
  so reads, decoding and copies do not yet overlap across CUDA streams the way a C-API implementation can
  (see Tom Augspurger's [GPU-accelerated Zarr](https://tomaugspurger.net/posts/gpu-accelerated-zarr/)).
- zarr's `vindex` (coordinate selection) does not support GPU buffers with any pipeline.

## Compatibility

The GPU tests pass with Python 3.13 and 3.14, CUDA 13, on NVIDIA Blackwell GPUs, with:

| KvikIO | nvCOMP | zarr-python |
|---|---|---|
| 26.04, 26.06, 26.08 | 5.2, 5.3 | 3.3, 3.4 |

The CPU tests (no GPU stack) also run on Python 3.12 in CI.

- **nvCOMP ≥ 5.2**: decoding standard zstd frames in batches into preallocated buffers needs its newer Python API.
- **zarr-python**: the pipeline subclasses zarr's `BatchedCodecPipeline` and uses internal helpers, so new zarr releases can break it.
- **CUDA 12** (`[cuda12]`) is untested.

## Development

```bash
uv sync --extra cuda13 --extra dask --group test --group dev --group typecheck
uv run pre-commit install
uv run pytest            # GPU tests skip without a CUDA GPU and the cuda extra
uv run mypy
```

The tests compare every read against zarr's default CPU pipeline. GitHub's runners have no GPU, so CI runs the
CPU tests, the type check, the lint hooks and a build; run the GPU tests locally before a release.

To release, publish a GitHub release with a tag `vX.Y.Z`: the version comes from the tag
(hatch-vcs), and `release.yml` uploads it to PyPI through trusted publishing.
Add the release's changes to [CHANGELOG.md](CHANGELOG.md) first.

## License

MIT, see [LICENSE](LICENSE).
