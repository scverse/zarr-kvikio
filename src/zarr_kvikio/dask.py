"""Dask integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import zarr
from distributed.diagnostics.plugin import WorkerPlugin

from ._config import PIPELINE_PATH, _enable, _host_pipeline, require
from ._pipeline import HOST_PIPELINE_KEY

if TYPE_CHECKING:
    from distributed import Client
    from donfig.config_obj import ConfigSet

__all__ = ["KvikioPlugin", "enable"]


def enable(client: Client, *, num_threads: int = 8) -> _ClusterConfigSet:
    """Read zarr arrays with KvikIO and nvCOMP in a Dask cluster.

    Every worker of the cluster (including ones started later) reads into GPU memory, so every
    block of e.g. :func:`anndata.experimental.read_elem_lazy` arrives as a GPU array.
    This process keeps reading into host memory (e.g. ``obs`` and ``var``), but arrays it opens
    and hands to the workers are set up to use KvikIO there.

    Takes effect right away; used as a context manager, it is undone at the end of the
    ``with`` block, here and on the workers::

        zarr_kvikio.dask.enable(client)
        with zarr_kvikio.dask.enable(client):
            ...

    Parameters
    ----------
    client
        A :class:`~distributed.Client` of a (dask-cuda) cluster.
    num_threads
        Number of threads KvikIO reads with on every worker.
    """
    require()
    host = _host_pipeline()
    client.register_plugin(KvikioPlugin(num_threads=num_threads, host_pipeline=host))
    # zarr fixes an array's codec pipeline when the array is opened, and arrays opened here
    # are shipped to the workers with it.
    local = zarr.config.set({"codec_pipeline.path": PIPELINE_PATH, HOST_PIPELINE_KEY: host})
    return _ClusterConfigSet(local, client, KvikioPlugin.name)


class KvikioPlugin(WorkerPlugin):
    """Make a Dask worker read zarr arrays into GPU memory with KvikIO and nvCOMP.

    Usually used through :func:`zarr_kvikio.dask.enable`.
    """

    name = "zarr-kvikio"

    def __init__(self, *, num_threads: int = 8, host_pipeline: str | None = None) -> None:
        self.num_threads = num_threads
        # the client's host pipeline (e.g. zarrs), for what the GPU can't decode
        self.host_pipeline = host_pipeline

    def setup(self, worker) -> None:
        """Switch GPU reads on in the worker."""
        self._config = _enable(self.num_threads, self.host_pipeline)

    def teardown(self, worker) -> None:
        """Undo :meth:`setup` and free the decoders' GPU memory."""
        from ._pipeline import release_memory

        if (config := getattr(self, "_config", None)) is not None:  # set up
            config.__exit__(None, None, None)
        release_memory()


class _ClusterConfigSet:
    """Undoes :func:`enable` here and on the workers."""

    def __init__(self, local: ConfigSet, client: Client, plugin: str) -> None:
        self._local, self._client, self._plugin = local, client, plugin

    def __enter__(self) -> _ClusterConfigSet:
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.unregister_worker_plugin(self._plugin)
        self._local.__exit__(*exc)
