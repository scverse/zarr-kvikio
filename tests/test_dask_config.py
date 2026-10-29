"""How `zarr_kvikio.dask.enable` changes and restores the config, without reading on the GPU."""

from __future__ import annotations

import re
from itertools import chain, permutations
from types import SimpleNamespace

import distributed
import pytest
import zarr

import zarr_kvikio
import zarr_kvikio.dask
from zarr_kvikio._config import INSTALL_HINT
from zarr_kvikio._pipeline import HOST_PIPELINE_KEY

PIPELINE = "zarr_kvikio.KvikioCodecPipeline"
CLUSTERS = [
    pytest.param((False, 1), id="in-process"),
    pytest.param((False, 2), id="in-process-2-workers"),
    pytest.param((True, 1), id="processes"),
]


def plugins(client: distributed.Client) -> list[str]:
    names = client.run_on_scheduler(lambda dask_scheduler: list(dask_scheduler.worker_plugins))
    return [n for n in names if n.startswith(zarr_kvikio.dask.KvikioPlugin.name)]


@pytest.fixture
def state():
    # defined in here to be sent to the workers by value: worker processes can only import
    # this module when pytest runs from the repository root
    def state() -> tuple[str | None, ...]:
        return tuple(
            zarr.config.get(k, None) for k in ("codec_pipeline.path", HOST_PIPELINE_KEY, "buffer")
        )

    return state


@pytest.fixture
def client(request, monkeypatch, state):
    processes, n_workers = request.param
    before = state()
    with (
        distributed.LocalCluster(
            n_workers=n_workers, threads_per_worker=1, processes=processes, dashboard_address=":0"
        ) as cluster,
        distributed.Client(cluster) as client,
    ):
        # the workers check for KvikIO and nvCOMP, which they don't need with num_threads=None
        # (in-process workers share this process's modules)
        monkeypatch.setattr(zarr_kvikio._config, "available", lambda: True)
        if processes:
            client.run(setattr, zarr_kvikio._config, "available", lambda: True)
        yield client
    # closing workers tear down the plugins they still have, which must not change this process
    assert state() == before


@pytest.mark.parametrize("client", CLUSTERS, indirect=True)
def test_enable_nested(client, state):
    before, on_workers = state(), set(client.run(state).values())
    # undone in any order, the calls not undone yet stay in effect, here and on the workers
    for order in chain(permutations(range(2)), permutations(range(3))):
        calls = [zarr_kvikio.dask.enable(client, num_threads=None) for _ in order]
        enabled, enabled_on_workers = state(), set(client.run(state).values())
        assert enabled[0] == PIPELINE
        assert {(path, buf) for path, _, buf in enabled_on_workers} == {
            (PIPELINE, "zarr.buffer.gpu.Buffer")
        }
        assert len(plugins(client)) == len(calls)
        for i in order[:-1]:
            calls[i].__exit__(None, None, None)
            assert (state(), set(client.run(state).values())) == (enabled, enabled_on_workers)
        calls[order[-1]].__exit__(None, None, None)
        assert (state(), set(client.run(state).values())) == (before, on_workers), order
        assert plugins(client) == []


@pytest.mark.parametrize("client", CLUSTERS[-1:], indirect=True)
def test_enable_undo_errors(client, state, monkeypatch):
    def unreachable(name: str) -> None:
        raise OSError(name)

    before = state()
    with pytest.raises(KeyError), zarr_kvikio.dask.enable(client, num_threads=None):
        raise KeyError
    assert state() == before
    assert plugins(client) == []
    # this process is restored even when the workers can't be reached
    enabled = zarr_kvikio.dask.enable(client, num_threads=None)
    monkeypatch.setattr(client, "unregister_worker_plugin", unreachable)
    with pytest.raises(OSError, match=zarr_kvikio.dask.KvikioPlugin.name):
        enabled.__exit__(None, None, None)
    assert state() == before


@pytest.mark.parametrize("client", CLUSTERS[1:2], indirect=True)
def test_enable_setup_fails(client, state, monkeypatch):
    setup = zarr_kvikio.dask.KvikioPlugin.setup

    def setup_fails_on_second_worker(self, worker) -> None:
        if worker.name == 1:
            raise RuntimeError("no GPU")
        setup(self, worker)

    monkeypatch.setattr(zarr_kvikio.dask.KvikioPlugin, "setup", setup_fails_on_second_worker)
    before = state()
    with pytest.raises(RuntimeError, match="no GPU"):
        zarr_kvikio.dask.enable(client, num_threads=None)
    # undone on the worker that was set up too, which shares this process's config
    assert set(client.run(state).values()) == {before}
    assert state() == before
    assert plugins(client) == []


def test_enable_interrupted(state):
    def interrupt(*args, **kwargs) -> None:
        raise KeyboardInterrupt

    def fail(*args, **kwargs) -> None:
        raise RuntimeError("no GPU")

    before, removed = state(), []
    # a Ctrl-C doesn't wait for the workers to remove the plugin
    client = SimpleNamespace(register_plugin=interrupt, unregister_worker_plugin=removed.append)
    with pytest.raises(KeyboardInterrupt):
        zarr_kvikio.dask.enable(client, num_threads=None)
    assert (state(), removed) == (before, [])
    # this process is left as it was, also after a Ctrl-C while removing the plugin
    client = SimpleNamespace(register_plugin=fail, unregister_worker_plugin=interrupt)
    with pytest.raises(KeyboardInterrupt):
        zarr_kvikio.dask.enable(client, num_threads=None)
    assert state() == before


@pytest.mark.parametrize("client", CLUSTERS[-1:], indirect=True)
def test_enable_needs_gpu_stack_on_workers(client, state, monkeypatch):
    # this process keeps reading into host memory
    monkeypatch.setattr(zarr_kvikio._config, "available", lambda: False)
    with zarr_kvikio.dask.enable(client, num_threads=None):
        pass
    # the workers need them
    client.run(setattr, zarr_kvikio._config, "available", lambda: False)
    before = state()
    with pytest.raises(ImportError, match=re.escape(INSTALL_HINT)):
        zarr_kvikio.dask.enable(client, num_threads=None)
    assert state() == before
    assert plugins(client) == []


@pytest.mark.skipif(not zarr_kvikio.available(), reason="needs KvikIO and nvCOMP")
@pytest.mark.parametrize("client", CLUSTERS[-1:], indirect=True)
def test_num_threads_restored(client):
    import kvikio.defaults

    def num_threads() -> int:
        return kvikio.defaults.get("num_threads")

    before = num_threads()
    with zarr_kvikio.enable(num_threads=before + 3):
        assert num_threads() == before + 3
    assert num_threads() == before
    on_workers = client.run(num_threads)
    with zarr_kvikio.dask.enable(client, num_threads=max(on_workers.values()) + 3):
        assert client.run(num_threads) != on_workers
    assert client.run(num_threads) == on_workers
