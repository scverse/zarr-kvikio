# Quick start

With the [GPU dependencies installed](installation.md), enable the pipeline before opening the array.
This reads the first 1,024 rows of `data.zarr` into a CuPy array.

```python
import zarr
import zarr_kvikio

with zarr_kvikio.enable():
    array = zarr.open_array("data.zarr", mode="r")
    rows = array[:1024]
```

Zarr selects the pipeline when it opens the array. {func}`zarr_kvikio.enable` also selects GPU buffers for reads.
The `with` block restores the previous settings when it ends. To keep GPU reads enabled for the process, call
`zarr_kvikio.enable()` directly.

To see whether your array’s chunks can be decoded on the GPU, inspect its layout.

```python
print(zarr_kvikio.describe(array))
```

The result lists the codecs, dtype and chunk size, along with any layout issues.
See [writing stores for the GPU](usage.md#writing-stores-for-the-gpu) for an example of rewriting an array.

## Read a remote array

Use a URL to open a remote array. This example reads two rows from a small Zarr v3 array published by zarrs.
Install the HTTP backend with `pip install 'fsspec[http]'`.

```python
import zarr
import zarr_kvikio

url = (
    "https://raw.githubusercontent.com/zarrs/zarrs/zarrs-v0.23.14/"
    "zarrs/tests/data/v3_zarr_python/array_zstd.zarr"
)
with zarr_kvikio.enable():
    array = zarr.open_array(url, mode="r")
    rows = array[:2]  # CuPy float32 array, shape (2, 10)
```

Zarr reads through its `FsspecStore`. For an S3 array, install `s3fs` and use an `s3://` URL.
See the [storage guide](https://zarr.readthedocs.io/en/stable/user-guide/storage/) for credentials and other backend options.
