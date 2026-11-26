# Installation

zarr-kvikio requires Python 3.12 or later. To read into GPU memory, install the extra for your CUDA version.

```bash
pip install 'zarr-kvikio[cuda13]'  # CUDA 13
# or
pip install 'zarr-kvikio[cuda12]'  # CUDA 12
```

For [Dask workers](usage.md#read-with-dask), combine the extras, for example `pip install 'zarr-kvikio[cuda13,dask]'`.
A client that only schedules GPU workers can use `pip install 'zarr-kvikio[dask]'`.
To inspect or convert stores on the CPU, install the base package with `pip install zarr-kvikio`.

To check the installed versions of KvikIO, nvCOMP, zarr and CuPy and the status of GPUDirect Storage, run

```bash
python -c 'import zarr_kvikio; print(zarr_kvikio.check())'
```

See the [quick start](quickstart.md) for a GPU read and [compatibility](usage.md#compatibility) for tested versions.
