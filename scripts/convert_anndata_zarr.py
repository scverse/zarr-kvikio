"""Rewrite an AnnData zarr store with a CSR/CSC `X` into a GPU-friendly layout.

- `X/data` and `X/indices`: zstd, sharded (256 KiB inner chunks, 512 per shard).
  nvCOMP decodes each zstd frame (= inner chunk) with one thread block, so frames must be
  small enough to give the GPU parallelism for dask-sized reads (a lone 1 MiB frame: ~20 ms).
- `X/indices`: the smallest dtype that holds the minor-axis size (uint16 for < 65,536 genes)
- everything else is copied with anndata's `read_elem`/`write_elem`

The shards are written once each, in parallel;
`--verify` re-reads every shard and compares it with the source.

    python convert_anndata_zarr.py SRC.zarr DST.zarr [--n-rows N] [--workers 16] [--verify]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from math import ceil

import anndata as ad
import numpy as np
import zarr
from zarr.codecs import ZstdCodec


def index_dtype(n_minor: int) -> np.dtype:
    for dt in (np.uint16, np.int32):
        if n_minor - 1 <= np.iinfo(dt).max:
            return np.dtype(dt)
    return np.dtype(np.int64)


def copy_other_elems(src: zarr.Group, dst: zarr.Group, n_rows: int | None) -> None:
    for key in src:
        if key == "X":
            continue
        elem = ad.io.read_elem(src[key])
        if n_rows is not None:
            if key == "obs":
                elem = elem.iloc[:n_rows]
            elif key == "obsm":
                elem = {k: v[:n_rows] for k, v in elem.items()}
            elif key == "obsp":
                elem = {k: v[:n_rows, :n_rows] for k, v in elem.items()}
            elif key == "layers" and elem:
                msg = "subsetting non-empty layers is not supported"
                raise NotImplementedError(msg)
        ad.io.write_elem(dst, key, elem)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--n-rows", type=int, help="only convert the first N rows (for testing)")
    p.add_argument("--inner-bytes", type=int, default=256 << 10, help="inner chunk size of X/data")
    p.add_argument("--per-shard", type=int, default=512, help="inner chunks per shard")
    p.add_argument("--level", type=int, default=3, help="zstd level")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    src = zarr.open_group(args.src, mode="r")
    sx = src["X"]
    fmt = sx.attrs["encoding-type"]
    if fmt not in {"csr_matrix", "csc_matrix"}:
        msg = f"X must be a CSR or CSC matrix, got {fmt}"
        raise ValueError(msg)
    shape = tuple(sx.attrs["shape"])
    major, minor = (0, 1) if fmt == "csr_matrix" else (1, 0)
    n_major = shape[major] if args.n_rows is None else args.n_rows
    if args.n_rows is not None and fmt != "csr_matrix":
        msg = "--n-rows needs a CSR matrix"
        raise ValueError(msg)
    out_shape = [0, 0]
    out_shape[major], out_shape[minor] = n_major, shape[minor]

    indptr = sx["indptr"][: n_major + 1]
    nnz = int(indptr[-1])
    s_data, s_idx = sx["data"], sx["indices"]
    idx_dtype = index_dtype(shape[minor])
    inner = args.inner_bytes // s_data.dtype.itemsize
    shard = inner * args.per_shard
    n_shards = ceil(nnz / shard)
    print(
        f"X: {fmt} {tuple(out_shape)}, nnz={nnz:,}, data {s_data.dtype}, "
        f"indices {s_idx.dtype} -> {idx_dtype}"
    )
    print(
        f"   inner chunks of {inner:,} elements, shards of {shard:,} elements "
        f"-> {n_shards} shards per array"
    )

    dst = zarr.open_group(args.dst, mode="w-", zarr_format=3)
    dst.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
    t0 = time.perf_counter()
    copy_other_elems(src, dst, args.n_rows)
    print(f"copied non-X elements in {time.perf_counter() - t0:.1f}s", flush=True)

    dx = dst.create_group("X")
    dx.attrs.update({"encoding-type": fmt, "encoding-version": "0.1.0", "shape": out_shape})
    # indptr gets small chunks too: every read of a row range reads a piece of it
    dx.create_array(
        "indptr",
        data=indptr,
        chunks=(inner,),
        shards=(shard,),
        compressors=[ZstdCodec(level=args.level)],
    )
    compressors = [ZstdCodec(level=args.level)]
    d_data = dx.create_array(
        "data",
        shape=(nnz,),
        dtype=s_data.dtype,
        chunks=(inner,),
        shards=(shard,),
        compressors=compressors,
    )
    d_idx = dx.create_array(
        "indices",
        shape=(nnz,),
        dtype=idx_dtype,
        chunks=(inner,),
        shards=(shard,),
        compressors=compressors,
    )

    def convert(k: int) -> int:
        s, e = k * shard, min(nnz, (k + 1) * shard)
        idx = s_idx[s:e]
        if idx.size and (idx.min() < 0 or idx.max() >= shape[minor]):
            msg = f"indices out of range in [{s}, {e})"
            raise ValueError(msg)
        d_data[s:e] = s_data[s:e]
        d_idx[s:e] = idx.astype(idx_dtype)
        return e - s

    def verify(k: int) -> int:
        s, e = k * shard, min(nnz, (k + 1) * shard)
        if not np.array_equal(d_data[s:e], s_data[s:e]) or not np.array_equal(
            d_idx[s:e], s_idx[s:e]
        ):
            msg = f"mismatch in [{s}, {e})"
            raise AssertionError(msg)
        return e - s

    for name, fn in [("convert", convert), *([("verify", verify)] if args.verify else [])]:
        t0, done = time.perf_counter(), 0
        with ThreadPoolExecutor(args.workers) as ex:
            for i, n in enumerate(ex.map(fn, range(n_shards)), 1):
                done += n
                if i % max(1, n_shards // 20) == 0 or i == n_shards:
                    dt = time.perf_counter() - t0
                    print(
                        f"{name}: {i}/{n_shards} shards, {done / 1e9:.2f}B nnz, {dt:.0f}s "
                        f"({done / dt / 1e6:.0f}M nnz/s)",
                        flush=True,
                    )
    if args.verify:
        assert np.array_equal(dx["indptr"][:], indptr)
        print("verified: X is identical to the source")
    zarr.consolidate_metadata(dst.store)
    print("done")


if __name__ == "__main__":
    main()
