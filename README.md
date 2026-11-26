# zarr-kvikio

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/zarr-kvikio/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/zarr-kvikio/badge/
[tests]: https://github.com/scverse/zarr-kvikio/actions/workflows/test.yaml
[documentation]: https://zarr-kvikio.readthedocs.io/

zarr-kvikio is a [Zarr-Python](https://zarr.readthedocs.io/) codec pipeline for reading local and remote arrays into CuPy arrays.
It accelerates reads of local Zarr v3 arrays with uncompressed or Zstandard-compressed numeric chunks using
[KvikIO](https://github.com/rapidsai/kvikio) and [nvCOMP](https://developer.nvidia.com/nvcomp).
Remote stores use Zarr’s standard GPU-buffer reads.

## Installation

Choose the extra that matches your CUDA installation, either `cuda12` or `cuda13`.

```bash
pip install 'zarr-kvikio[cuda13]'
```

For Dask support, add the `dask` extra (e.g. `zarr-kvikio[cuda13,dask]`). To inspect or convert stores
on the CPU, install the base package with `pip install zarr-kvikio`.

## Quick start

Enable the pipeline before opening a local array. For example, read the first 1,024 rows of `data.zarr` into GPU memory.

```python
import zarr
import zarr_kvikio

with zarr_kvikio.enable():
    array = zarr.open_array("data.zarr", mode="r")
    rows = array[:1024]  # cupy.ndarray
```

See the [documentation][documentation] for Dask setup, supported store layouts, configuration, and the API reference.
