# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - unreleased

### Added

- `KvikioCodecPipeline`: reads zstd-compressed or uncompressed (optionally sharded) zarr v3 arrays from local
  files into GPU memory with KvikIO and decodes them with nvCOMP; everything else is handled by
  `kvikio.host_pipeline` (e.g. zarrs) or zarr's own pipelines. Contiguous ranges of sharded 1-D arrays are read
  in batches. Like zarrs, it is selected and set up through zarr's config alone: `codec_pipeline.path`, and
  `kvikio.*` settings (also as `ZARR_KVIKIO__*` environment variables).
- `enable()`, to select it with GPU buffers, and `zarr_kvikio.dask.enable()` (undoable, also on Dask workers, which
  get the `kvikio.*` settings; `dask.enable()` calls can be undone in any order, and only the workers need KvikIO
  and nvCOMP), `available()`, `require()`.
- `check()` and `describe()` to diagnose the installation and an array's layout.
- `convert_array()`, `layout()` and `cpu_pipeline()` to rewrite arrays in the layout the GPU reads fastest.
- `release_memory()` to free nvCOMP's scratch space and the pinned host read buffers.
- The `zarr.codec_pipeline` entry point goes through a small guard module: if `zarr_kvikio` fails to import
  (e.g. with an untested zarr release), zarr keeps working, and only selecting the pipeline raises.

Moved from [rapids-singlecell](https://github.com/scverse/rapids-singlecell), where it was developed as
`rapids_singlecell_zarr`.

[0.1.0]: https://github.com/scverse/zarr-kvikio/releases/tag/v0.1.0
