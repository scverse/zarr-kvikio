# zarr-kvikio

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/zarr-kvikio/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/zarr-kvikio/badge/
[tests]: https://github.com/scverse/zarr-kvikio/actions/workflows/test.yaml
[documentation]: https://zarr-kvikio.readthedocs.io/

zarr-kvikio reads local Zarr v3 arrays into CuPy arrays using [KvikIO](https://github.com/rapidsai/kvikio)
for I/O and [nvCOMP](https://developer.nvidia.com/nvcomp) for GPU decompression. It accelerates reads of
Zstandard-compressed and uncompressed numeric chunks. Other codecs fall back to CPU decoding.

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
