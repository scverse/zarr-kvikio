"""Dask integration."""

from __future__ import annotations

import contextlib
import copy
import threading
import uuid
from typing import TYPE_CHECKING, Any

import zarr
from distributed.diagnostics.plugin import WorkerPlugin

from ._config import GPU_BUFFERS, PIPELINE_PATH, _host_pipeline, require

if TYPE_CHECKING:
    from collections.abc import Callable

    from distributed import Client

__all__ = ["KvikioPlugin", "enable"]

# zarr's settings are shared by all workers in a process (and with the client, for in-process
# workers), so what enable() and the plugins change in a process is undone in reverse
# order of the changes, whatever order the calls are undone or the workers stop in
_lock = threading.Lock()
_changes: list[_Change] = []


def enable(client: Client) -> _ClusterConfigSet:
    """Read zarr arrays with KvikIO and nvCOMP in a Dask cluster.

    Every worker of the cluster (including ones started later) reads into GPU memory, so every
    block of e.g. :func:`anndata.experimental.read_elem_lazy` arrives as a GPU array.
    This process keeps reading into host memory (e.g. ``obs`` and ``var``) and needs neither
    KvikIO nor nvCOMP, but arrays it opens and hands to the workers are set up to use KvikIO
    there. A worker without KvikIO or nvCOMP makes this raise an :class:`ImportError` (and undo
    itself). The workers get this process's ``kvikio.*`` settings in zarr's config (e.g.
    ``kvikio.num_threads``), as they are now. In-process workers (e.g. ``Client(processes=False)``)
    share zarr's settings with this process, which then reads into GPU memory too, until the call
    is undone (also after the cluster closes).

    Takes effect right away; used as a context manager, it is undone at the end of the
    ``with`` block, here and on the workers. Calls can be undone
    in any order: the ones not undone yet stay in effect, such as an enclosing ``with`` block::

        zarr_kvikio.dask.enable(client)
        with zarr_kvikio.dask.enable(client):
            ...

    Undo it in reverse order relative to :func:`zarr_kvikio.enable` and ``zarr.config.set`` calls
    in this process, though, as those need.

    Parameters
    ----------
    client
        A :class:`~distributed.Client` of a (dask-cuda) cluster.
    """
    host = _host_pipeline()
    # (dotted keys: they only replace the same keys on the workers)
    settings = {f"kvikio.{k}": v for k, v in copy.deepcopy(zarr.config.get("kvikio", {})).items()}
    settings |= {"codec_pipeline.path": PIPELINE_PATH, **GPU_BUFFERS, **host}
    # a name of its own, so that undoing this call leaves other calls in effect
    name = f"{KvikioPlugin.name}-{uuid.uuid4().hex}"
    try:
        client.register_plugin(KvikioPlugin(settings), name=name)
    except Exception:
        # distributed keeps a plugin that a worker fails to set up (on the scheduler and on the
        # workers that did set it up). Not on e.g. Ctrl-C, which must not wait for the workers
        # (the plugin may then stay on the cluster).
        with contextlib.suppress(Exception):
            client.unregister_worker_plugin(name)
        raise
    # zarr fixes an array's codec pipeline when the array is opened, and arrays opened here
    # are shipped to the workers with it. Set after setting up the workers, which can fail.
    local = _Change(lambda: zarr.config.set({"codec_pipeline.path": PIPELINE_PATH, **host}))
    return _ClusterConfigSet(local, client, name)


class KvikioPlugin(WorkerPlugin):
    """Make a Dask worker read zarr arrays into GPU memory with KvikIO and nvCOMP.

    Usually used through :func:`zarr_kvikio.dask.enable`.
    """

    name = "zarr-kvikio"
    _change: _Change | None = None

    def __init__(self, settings: dict[str, Any]) -> None:
        # zarr config: the pipeline, GPU buffers and the client's `kvikio.*` settings
        self.settings = settings

    def setup(self, worker) -> None:
        """Switch GPU reads on in the worker, which needs KvikIO and nvCOMP."""
        require()
        self._change = _Change(lambda: zarr.config.set(self.settings))

    def teardown(self, worker) -> None:
        """Undo :meth:`setup` and free the decoders' GPU memory."""
        from ._pipeline import release_memory

        if self._change is not None:  # set up
            self._change.undo()
        release_memory()


class _ClusterConfigSet:
    """Undoes :func:`enable` here and on the workers."""

    def __init__(self, local: _Change, client: Client, plugin: str) -> None:
        self._local, self._client, self._plugin = local, client, plugin

    def __enter__(self) -> _ClusterConfigSet:
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self._client.unregister_worker_plugin(self._plugin)
        finally:
            self._local.undo()


class _Change:
    """A change to this process's settings, made right away and reverted in reverse order."""

    _undone = False

    def __init__(self, change: Callable[[], contextlib.AbstractContextManager[object]]) -> None:
        with _lock:
            self._config = change()
            _changes.append(self)

    def undo(self) -> None:
        """Revert this change as soon as all later ones are undone too."""
        with _lock:
            self._undone = True
            while _changes and _changes[-1]._undone:
                _changes.pop()._config.__exit__(None, None, None)
