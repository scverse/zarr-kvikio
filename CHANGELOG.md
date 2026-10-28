# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - unreleased

### Added

- `KvikioCodecPipeline`: reads zstd-compressed or uncompressed (optionally sharded) zarr v3 arrays from local
  files into GPU memory with KvikIO and decodes them with nvCOMP; everything else is handled by the previously
  configured pipeline (e.g. zarrs) or zarr's fused pipeline. Contiguous ranges of sharded 1-D arrays are read
  in batches.
- `enable()` and `zarr_kvikio.dask.enable()` (undoable, also on Dask workers), `available()`, `require()`.
- `check()` and `describe()` to diagnose the installation and an array's layout.
- `convert_array()`, `layout()` and `cpu_pipeline()` to rewrite arrays in the layout the GPU reads fastest.
- `release_memory()` to free nvCOMP's scratch space.

Moved from [rapids-singlecell](https://github.com/scverse/rapids-singlecell), where it was developed as
`rapids_singlecell_zarr`.
