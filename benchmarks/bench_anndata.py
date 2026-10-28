"""Row-slice reads of a CSR matrix with anndata's `sparse_dataset` into GPU memory.

Usage: python bench_anndata.py STORE.zarr [PIPELINE ...]
PIPELINE is `default` (zarr's BatchedCodecPipeline, CPU decode) or `kvikio`.
Set KVIKIO_NTHREADS to control KvikIO's read parallelism.
"""

import sys
import time
import warnings

import anndata as ad
import cupy as cp
import cupyx.scipy.sparse as cpx
import zarr

import zarr_kvikio  # noqa: F401

warnings.filterwarnings("ignore")

ROWS = [(0, 20_000), (300_000, 320_000), (0, 200_000), (0, 500_000)]
PIPELINES = {
    "default": "zarr.core.codec_pipeline.BatchedCodecPipeline",
    "kvikio": "zarr_kvikio.KvikioCodecPipeline",
}


def bench(store: str, pipeline: str, reps: int = 5) -> None:
    with zarr.config.enable_gpu(), zarr.config.set({"codec_pipeline.path": PIPELINES[pipeline]}):
        X = ad.io.sparse_dataset(zarr.open_group(store, mode="r")["X"])
        for r0, r1 in ROWS:
            best = float("inf")
            for _ in range(reps):
                m = None
                t0 = time.perf_counter()
                m = X[r0:r1]
                cp.cuda.Device().synchronize()
                best = min(best, time.perf_counter() - t0)
            assert isinstance(m, cpx.csr_matrix), type(m)
            print(f"{pipeline:8s} rows {r0}:{r1:<8d} {best:7.3f}s  nnz={int(m.nnz):,}", flush=True)
            del m
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    store, *pipelines = sys.argv[1:]
    for p in pipelines or list(PIPELINES):
        bench(store, p)
