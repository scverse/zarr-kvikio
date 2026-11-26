# zarr-kvikio

zarr-kvikio reads local Zarr v3 arrays into CuPy arrays. It uses [KvikIO](https://github.com/rapidsai/kvikio) for I/O
and [nvCOMP](https://developer.nvidia.com/nvcomp) to decompress Zstandard chunks on the GPU. Other codecs fall back to CPU decoding.

[Install zarr-kvikio](installation.md) and [read your first array on the GPU](quickstart.md).
The [user guide](usage.md) covers Dask workers, store layouts and performance tuning.
The [API reference](api.md) documents the pipeline and its Python helpers.

```{toctree}
:caption: Getting started
:hidden: true
:maxdepth: 1

installation.md
quickstart.md
```

```{toctree}
:caption: Guides
:hidden: true
:maxdepth: 1

usage.md
```

```{toctree}
:caption: Reference
:hidden: true
:maxdepth: 1

api.md
```

```{toctree}
:caption: Project
:hidden: true
:maxdepth: 1

release-notes/index.md
contributing.md
references.md
```
