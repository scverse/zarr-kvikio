# zarr-kvikio

zarr-kvikio is a Zarr-Python codec pipeline that reads numeric arrays into CuPy arrays.
For local Zarr v3 arrays it accelerates uncompressed reads and Zstandard decompression with
[KvikIO](https://github.com/rapidsai/kvikio) and [nvCOMP](https://developer.nvidia.com/nvcomp).
For remote arrays it uses Zarr’s standard GPU-buffer reads.

After [installation](installation.md), select the pipeline and GPU buffers before opening an array.

```python
import zarr

zarr.config.set({"codec_pipeline.path": "zarr_kvikio.KvikioCodecPipeline"})
zarr.config.enable_gpu()
array = zarr.open_array("data.zarr", mode="r")
rows = array[:1024]  # cupy.ndarray
```

The [quick start](quickstart.md) shows how to enable GPU reads inside a `with` block and how to open a remote array.
The user guide covers [supported stores](usage.md#supported-stores), [indexing](usage.md#indexing),
[codec requirements](usage.md#what-takes-the-fast-path), and [configuration](usage.md#configuration).
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
