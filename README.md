# zarr-kvikio

A [zarr-python](https://github.com/zarr-developers/zarr-python) codec pipeline that reads chunks from a local file system
straight into GPU memory with [KvikIO](https://github.com/rapidsai/kvikio) and decodes them with [nvCOMP](https://developer.nvidia.com/nvcomp).

Status: experimental prototype.

## Usage

```python
import zarr
import zarr_kvikio

zarr_kvikio.enable()  # or: with zarr_kvikio.enable(): ...
arr = zarr.open_array("data.zarr", mode="r")  # the pipeline is bound when the array is opened
x = arr[:]  # cupy.ndarray
```

With Dask, one call switches the cluster: the workers read into GPU memory,
while the client keeps reading `obs`, `var` etc. into host memory as usual:

```python
import zarr_kvikio.dask

zarr_kvikio.dask.enable(client)  # or: with zarr_kvikio.dask.enable(client): ... (undone on the workers too)
X = anndata.experimental.read_elem_lazy(
    zarr.open_group("data.zarr")["X"]
)  # blocks arrive as cupyx matrices
```

`zarr_kvikio.available()` tells whether KvikIO and nvCOMP are installed; `zarr_kvikio.check()` also reports their
versions and whether GPUDirect Storage is active.

Install with `pip install -e ".[cuda13]"` (or `cuda12`; add `dask` for `zarr_kvikio.dask`).

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
- **`zarr_kvikio.release_memory()`** frees that scratch space.
- **`ZARR_KVIKIO_FETCH`** (default `host`): how contiguous reads of sharded 1-D arrays (e.g. a sparse matrix's `data` and
  `indices`) fetch the stored bytes. `host` reads them into pinned host memory and copies them to the GPU at once, which is
  fastest when the data comes from the page cache or GPUDirect Storage runs in compatibility mode; `kvikio` reads them
  straight into GPU memory with KvikIO (for GPUDirect Storage).
- **`ZARR_KVIKIO_READ_THREADS`** (default 16): threads for those host reads.
- If KvikIO isn't using GPUDirect Storage (check with `gdscheck -p`), it reads through the page cache and a bounce buffer.
  That still works, but isn't direct NVMe→GPU DMA.

## What takes the fast path

A read uses KvikIO + nvCOMP when all of these hold. Everything else, including reads into host buffers, is handled by the
pipeline zarr was configured with before `enable()` (e.g. zarrs), or else zarr's `FusedCodecPipeline`; reads into GPU
buffers that the GPU can't decode (e.g. blosc) are decoded on the CPU and then copied.

- the output is a GPU buffer (`zarr.config.enable_gpu()`),
- the array lives in a `zarr.storage.LocalStore`,
- the array is zarr v3 with a fixed-size numeric dtype, and
- its codecs are `bytes` (little endian), optionally followed by `zstd` (without checksum),
  either directly or as the inner codecs of the `sharding_indexed` codec
  (index codecs `bytes` + optional `crc32c`, index at the start or end).

Writes always use the fallback.

How a fast read works:

1. The shard indices are read on the host (a few bytes per inner chunk; the `crc32c` is verified).
2. The stored bytes of every needed chunk are read into device memory with KvikIO,
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

`zarr_kvikio.convert_array(src, dst_group, name)` streams a 1-D or 2-D array into this layout, shard by shard
(zstd, ~256 KiB chunks: whole rows of a 2-D array or row pieces, 512 chunks per shard, in order; optionally with a smaller
dtype, e.g. uint16 indices), and verifies the copy. `zarr_kvikio.layout(shape, itemsize)` gives its chunk and shard shape,
`zarr_kvikio.describe(array)` lists what keeps an existing array from being read fast, and
`zarr_kvikio.cpu_pipeline()` opens source arrays with a fast CPU pipeline.
For whole AnnData stores, use [`rapids_singlecell.io.convert_zarr`](https://rapids-singlecell.scverse.org/), which
converts every matrix this way and copies everything else; `scripts/convert_anndata_zarr.py` is an older standalone script.

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

Store layout: zstd level 3, sharded with 1 MiB inner chunks and 128 inner chunks per shard.
(Measured before the frame-size findings above; see `05_out-of-core_kvikio.ipynb` in the notebooks repository for the
full 11.4M-cell atlas in the recommended layout, read by a Dask cluster.)
For comparison, zarr's default pipeline on the original layout (blosc-lz4, int64 indices, one 609k-element chunk per file)
took 0.10 s / 0.75 s / ~1.3 s for 20k / 200k / 500k rows.

## Limitations

- Local files only (`LocalStore`); remote and in-memory stores use the fallback.
- zstd (without checksum) or no compression only. Blosc, which many existing stores use, cannot be decoded by nvCOMP.
- nvCOMP's Python API costs a few tens of milliseconds per `decode` call and blocks the calling thread,
  so reads, decoding and copies do not yet overlap across CUDA streams the way a C-API implementation can
  (see Tom Augspurger's [GPU-accelerated Zarr](https://tomaugspurger.net/posts/gpu-accelerated-zarr/)).
- zarr's `vindex` (coordinate selection) does not support GPU buffers with any pipeline.

## Compatibility

Tested (all tests passing) with Python 3.13 and 3.14, CUDA 13, on NVIDIA Blackwell GPUs:

| KvikIO | nvCOMP | zarr-python |
|---|---|---|
| 26.04, 26.06, 26.08 | 5.2, 5.3 | 3.3, 3.4 |

- **nvCOMP ≥ 5.2**: decoding standard zstd frames in batches into preallocated buffers needs its newer Python API.
- **zarr-python**: the pipeline subclasses zarr's `BatchedCodecPipeline` and uses internal helpers, so new zarr releases can break it.
- **CUDA 12** (`[cuda12]`) is untested.

## Tests

```bash
pytest
```

The tests compare every read against zarr's default CPU pipeline and need a CUDA GPU.
