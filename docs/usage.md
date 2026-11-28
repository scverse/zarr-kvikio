# User guide

## Select the pipeline

Select `zarr_kvikio.KvikioCodecPipeline` by name in `zarr.config` before opening the array, as in the
[configuration example](index.md). The {func}`zarr_kvikio.enable` helper selects the pipeline and GPU buffers together. Reads into host buffers use the configured [host pipeline](#configuration).

## Supported stores

Open the array with Zarr’s usual path, URL, or store object.
For GPU reads of fixed-size numeric arrays, the store determines how the pipeline reads chunks.

  | Zarr store                                         | How the pipeline reads into GPU memory                                                                                                                               |
  | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `LocalStore` (a directory on disk)                 | Eligible chunks use the [local fast path](#what-takes-the-fast-path). For other codec chains, the host pipeline decodes the chunks and copies the result to the GPU. |
  | `FsspecStore` (e.g. S3 or HTTPS) and `ObjectStore` | Zarr’s `BatchedCodecPipeline` reads through the store and fills the GPU buffer, including for sharded arrays.                                                        |
  | `ZipStore` and `MemoryStore`                       | These use the same Zarr GPU-buffer path as remote stores, even for a ZIP file on local disk.                                                                         |

Remote reads support the same codecs and data types as [Zarr’s GPU buffers](https://zarr.readthedocs.io/en/stable/user-guide/gpu/).
See the [remote read example](quickstart.md#read-a-remote-array) for backend setup.

## Indexing

GPU reads support integer indices, slices (including strides), orthogonal selections with `oindex`,
and chunk selections with `blocks`. For example, `array.oindex[[0, 3], :]` reads rows 0 and 3 of a 2-D array.
Zarr’s coordinate indexing (`vindex`) requires a host buffer.

## Read with Dask

{func}`zarr_kvikio.dask.enable` configures a Dask cluster’s workers to read into GPU memory.
The client can still read metadata such as AnnData’s `obs` and `var` into host memory.
For example, with a running cluster, read the first block of a sparse AnnData matrix on a worker.

```python
import anndata
import zarr
from distributed import get_client, wait

import zarr_kvikio.dask

client = get_client()
with zarr_kvikio.dask.enable(client):
    group = zarr.open_group("data.zarr", mode="r")
    matrix = anndata.experimental.read_elem_lazy(group["X"])
    first_block = client.persist(matrix.blocks[0, 0])
    wait(first_block)  # the cupyx sparse block stays on the worker
```

Call `zarr_kvikio.dask.enable(client)` without `with` to keep the configuration active.
Workers added later receive the same configuration.
Only the workers need KvikIO and nvCOMP: if one lacks them, `dask.enable()` raises and undoes itself. The workers get
this process’s `kvikio.*` settings (see [Configuration](#configuration)) as they are when it is called. In-process
workers (`Client(processes=False)`, `LocalCluster(processes=False)`) share zarr’s settings with the client, which then
reads into GPU memory too, until the call is undone (also after the cluster closes). Repeated `dask.enable()` calls can
be undone in any order.
Workers started some other way (e.g. `dask cuda worker`) can be set up through the environment zarr is imported in
instead, while the client selects the pipeline in its config:
`ZARR_CODEC_PIPELINE__PATH=zarr_kvikio.KvikioCodecPipeline` (for arrays opened on the workers, e.g. a sparse matrix’s
indices), `ZARR_BUFFER=zarr.buffer.gpu.Buffer`, `ZARR_NDBUFFER=zarr.buffer.gpu.NDBuffer`, and any `ZARR_KVIKIO__*`
settings.

## Configuration

Like zarrs’ options, zarr-kvikio’s are keys of zarr’s config, in its `kvikio` section:

```python
import zarr

zarr.config.set({"kvikio.host_pipeline": "zarrs.ZarrsCodecPipeline", "kvikio.num_threads": 16})
```

or environment variables, which zarr reads when it is imported (e.g. `ZARR_KVIKIO__NUM_THREADS=16`). Each read uses
the settings of the process it runs in as they are then, except for `kvikio.decoders` and `kvikio.host_pipeline`
(see below). Invalid values raise a `ValueError` naming the key.

- **`kvikio.host_pipeline`** handles host-buffer reads and writes, and local reads that need host decoding (e.g. Blosc).
  The default is Zarr’s `FusedCodecPipeline`. If you have selected a different pipeline such as
  [zarrs-python](https://zarrs-python.readthedocs.io/en/latest/), `enable()` and `dask.enable()`
  preserve it for host operations unless you set `kvikio.host_pipeline` explicitly.
  An opened array reads it when it first needs that pipeline, and keeps it (copies sent to Dask workers read it there).
- **`kvikio.num_threads`** (default 8, or `KVIKIO_NTHREADS` if that is set): the threads KvikIO reads with, a setting
  of the whole process that KvikIO sets to 1 by default. Before each read with KvikIO, KvikIO's thread count is set to
  this if it differs (also if it was set with `kvikio.defaults.set`); `None` leaves it alone.
- **`kvikio.max_decode_bytes`** (default 1 GiB): the most decoded bytes per nvCOMP call.
  nvCOMP's batched zstd decode needs about **3.2× the decoded size as scratch space**,
  and its codec keeps that memory for reuse, so this bounds the device memory held (≈3.4 GB by default).
  Each call also costs a few milliseconds, so a larger limit speeds up very large reads.
  Sharded 2-D reads also use this limit to group temporary tile buffers. A single shard selection can exceed it.
  Other reads can group up to twice the output size when that is larger.
- **`kvikio.min_bytes`** (default 1 MiB): reads that return fewer bytes are read and zstd-decoded on the CPU by
  zarr-kvikio itself and copied to the GPU at once, unless they decode at least that many bytes from 32 or more chunks
  (e.g. a column of a matrix).
  Every nvCOMP call has a fixed cost of a few milliseconds, more than decoding e.g. a small slice of a sparse matrix's `indptr`.
- **`kvikio.decoders`** (default 1): nvCOMP codecs per GPU, each with its own scratch space.
  More let several threads decode at once; for sparse-matrix reads from 10 threads per GPU, 2-4 were not faster.
  `zarr_kvikio.release_memory()` frees that scratch space. The setting is read when a GPU's codecs are created: at its
  first decode, and at the first one after `release_memory()`.
- **`kvikio.fetch`** (default `"host"`): how contiguous reads of zstd-compressed sharded 1-D arrays (e.g. a sparse
  matrix's `data` and `indices`) fetch the stored bytes. `"host"` reads them into pinned host memory and copies them to
  the GPU at once, which is fastest when the data comes from the page cache or GPUDirect Storage runs in compatibility
  mode; `"kvikio"` reads them straight into GPU memory with KvikIO (for GPUDirect Storage).
  `"host"` keeps its pinned buffers for later reads, shared by all threads: up to 4 GiB in all, but at least 64 MiB for
  each read running at once. A buffer grows for a larger read (freeing the other kept ones if no other read runs, it
  holds less than half the read, and that lets it grow; one that holds half only grows to fit), which pins new memory
  once; a read with more stored bytes than its buffer streams through it in parts, which can be slower.
  `zarr_kvikio.release_memory()` frees these buffers.
- **`kvikio.read_threads`** (default 16): threads for those host reads.

The files that reads keep open while they fetch the stored bytes use at most half the soft limit of open files
(`ulimit -n`, read at the first read), all threads together, and one read opens at most 256 of them at once; reads
wait for free file descriptors (first come, first served) instead of failing. Files opened only briefly, e.g. to read
shard indices or for reads decoded on the CPU, are not counted.

If KvikIO isn’t using GPUDirect Storage (`zarr_kvikio.check()`, or `gdscheck -p`), it reads through the page cache and
a bounce buffer. That still works (and is fast for cached data), but isn’t direct NVMe to GPU DMA. With the default
`kvikio.fetch`, contiguous reads of zstd-compressed sharded 1-D arrays go through pinned host memory instead of KvikIO,
so GPUDirect Storage applies to them only with `"kvikio"`.

## What takes the fast path

A read takes the fast path, which reads the stored bytes with KvikIO (or into pinned host memory, see
`kvikio.fetch`) and decodes them on the GPU with nvCOMP (uncompressed ones are copied), when all of these hold:

- the output is a GPU buffer of the array's data type (`zarr.config.enable_gpu()` with the pipeline selected, or
  `zarr_kvikio.enable()`) of at least `kvikio.min_bytes`, or the read decodes that many bytes from many chunks (others
  are decoded on the CPU by zarr-kvikio and then copied),
- the array lives in a `zarr.storage.LocalStore`,
- the array is zarr v3 with a fixed-size numeric dtype (not datetimes, which CuPy doesn't hold),
- its codecs are `bytes` (little endian), optionally followed by `zstd` (without checksum),
  either directly or as the inner codecs of the `sharding_indexed` codec
  (index codecs `bytes` + optional `crc32c`, index at the start or end), and
- its (inner) chunks are smaller than 2 GiB if zstd-compressed (nvCOMP's limit).

On a `LocalStore`, `ZstdCodec(checksum=False)` can use nvCOMP. `BloscCodec(cname="zstd")` uses the host pipeline
because its chunks are wrapped in the Blosc format. The host pipeline also handles data types outside the fixed-size
numeric types and local reads that cast to a different output dtype. GPU-buffer writes use Zarr’s `BatchedCodecPipeline`.

How a fast read works:

1. The shard indices are read on the host (a few bytes per inner chunk; the `crc32c` is verified).
2. The stored bytes of every needed chunk are read with KvikIO (or into pinned host memory, see `kvikio.fetch`),
   one read per contiguous byte range (so typically one read per shard).
3. The chunks of the request are decoded (if compressed) in batched nvCOMP calls of up to
   `kvikio.max_decode_bytes` each (usually one call).
4. The decoded values are copied into the output buffer.
   For 1-D arrays, chunks that are selected as a whole are decoded directly into the output.

## Writing stores for the GPU

- **Keep zstd frames (= inner chunks) small, around 256 KiB.** nvCOMP decodes each zstd frame with a single thread block,
  so one large frame is slow no matter how fast the GPU is: a lone 1 MiB frame takes ~20 ms, an 8 MiB frame ~150 ms.
  Measured on an RTX PRO 6000 (decode time, real float32 data):

  | frame size | 16 MiB read | 256 MiB read | 2 GiB read |
  | ---------- | ----------- | ------------ | ---------- |
  | 64 KiB     | 3.4 ms      | 22 ms        | 213 ms     |
  | 256 KiB    | 6.3 ms      | 12 ms        | 60 ms      |
  | 1 MiB      | 22 ms       | 27 ms        | 42 ms      |

  This applies to *every* array a read touches, e.g. also the `indptr` of a sparse matrix.

- **Shard** so that a read opens a few files instead of one per chunk: without sharding, small reads are dominated by per-file overhead.

- **Store the chunks of every shard in order**, so a contiguous range is a single read. zarr's fused pipeline does;
  zarrs (faster at writing) stores them in whatever order they were compressed in.

`convert_array` writes this layout (zstd, ~256 KiB chunks: whole rows of a 2-D array or row pieces, 512 chunks per
shard, in order; optionally with a smaller dtype, e.g. uint16 indices), and `describe` reports what an existing array
lacks. For example, copy a two-dimensional numeric array into a new group and verify its values.

```python
import zarr
import zarr_kvikio

with zarr_kvikio.cpu_pipeline():
    source = zarr.open_array("data.zarr", mode="r")
destination = zarr.open_group("converted.zarr", mode="a")
converted = zarr_kvikio.convert_array(source, destination, "data")
```

`convert_array` returns the new array, opened with the codec pipeline zarr is configured with (e.g. the GPU one after
`enable()`). It runs one conversion at a time, because zarr’s config is process-wide: while one runs, zarr reads in
other threads return NumPy arrays, and arrays opened in other threads while it creates the new array keep zarr’s fused
pipeline. `cpu_pipeline()` sets zarr’s config for the whole process too, so overlapping `cpu_pipeline()` blocks in
several threads are not supported.

## Benchmark

Row slices of a CSR matrix (the first 1M cells of a 11.4M × 45k single-cell atlas, float32 data, int32 indices)
read with `anndata.io.sparse_dataset(...)[start:stop]` in zarr GPU mode (`zarr.config.enable_gpu()`), producing a `cupyx` CSR matrix.
RTX PRO 6000 Blackwell, data in the page cache, KvikIO in compat mode, `KVIKIO_NTHREADS=16`.
The table shows the fastest of five runs of the [AnnData benchmark](https://github.com/scverse/zarr-kvikio/blob/main/benchmarks/bench_anndata.py).

  | rows              | nnz  | zarr default pipeline (CPU decode) | zarr-kvikio | zarr-kvikio, 4 GiB decode limit |
  | ----------------- | ---- | ---------------------------------- | ----------- | ------------------------------- |
  | 20k               | 73M  | 0.134 s                            | **0.062 s** | 0.062 s                         |
  | 20k (offset 300k) | 33M  | 0.071 s                            | **0.047 s** | 0.046 s                         |
  | 200k              | 624M | 1.170 s                            | **0.303 s** | 0.227 s                         |
  | 500k              | 1.1B | 2.109 s                            | **0.515 s** | 0.406 s                         |

Store layout: zstd level 3, sharded with 1 MiB inner chunks and 128 inner chunks per shard
(measured before the frame-size findings above). For comparison, zarr’s default pipeline on the original layout
(blosc-lz4, int64 indices, one 609k-element chunk per file) took 0.10 s / 0.75 s / ~1.3 s for 20k / 200k / 500k rows.

## Limitations

nvCOMP’s Python API costs a few milliseconds per `decode` call and blocks the calling thread,
so reads, decoding and copies do not yet overlap across CUDA streams the way a C-API implementation can
(see Tom Augspurger’s [GPU-accelerated Zarr](https://tomaugspurger.net/posts/gpu-accelerated-zarr/)).

## Compatibility

The GPU tests pass on NVIDIA Blackwell GPUs (RTX PRO 6000) with these versions, each in at least one tested combination:

  | CUDA              | KvikIO                     | nvCOMP        | zarr-python | Python     |
  | ----------------- | -------------------------- | ------------- | ----------- | ---------- |
  | 13 (`[cuda13]`)   | 26.04, 26.06, 26.08, 26.10 | 5.2, 5.3, 6.0 | 3.3, 3.4    | 3.13, 3.14 |
  | 12.9 (`[cuda12]`) | 26.04, 26.10               | 5.2, 6.0      | 3.3, 3.4    | 3.13       |

The CPU tests (no GPU stack) also run on Python 3.12 in CI.

- **nvCOMP ≥ 5.2**: decoding standard zstd frames in batches into preallocated buffers needs its newer Python API.
- **zarr-python**: the pipeline subclasses zarr's `BatchedCodecPipeline` and uses zarr internals, so zarr releases
  other than the tested ones above can break it. zarr then still works with its other pipelines: `import zarr_kvikio`
  (and so `enable()`) raises the original error, and selecting `zarr_kvikio.KvikioCodecPipeline` by name in zarr's
  config raises an `ImportError` chained to it.
