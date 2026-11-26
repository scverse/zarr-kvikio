"""Sphinx configuration for zarr-kvikio."""

from datetime import datetime
from importlib.metadata import metadata

from docutils import nodes
from docutils.parsers.rst import roles
from scverse_doc.config import DEFAULTS
from scverse_doc.registry import intersphinx

info = metadata("zarr-kvikio")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, scverse®"
version = release = info["Version"]
urls = dict(url.split(", ", 1) for url in info.get_all("Project-URL"))
source_repository = urls["Source"]

extensions = [
    "scverse_doc",
    "sphinx.ext.extlinks",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
    "IPython.sphinxext.ipython_console_highlighting",
]
bibtex_bibfiles = ["references.bib"]
typehints_defaults = "braces"
always_use_bars_union = True
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

intersphinx_mapping = intersphinx() | {
    "distributed": ("https://distributed.dask.org/en/stable/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
}
exclude_patterns = [*DEFAULTS.exclude_patterns, "release-notes/[0-9]*.md", "release-notes/blank.md"]
extlinks = {
    "issue": (f"{source_repository}/issues/%s", "issue%s"),
    "pr": (f"{source_repository}/pull/%s", "pr%s"),
}
nitpick_ignore = [
    # zarr's codec pipelines and donfig's config are documented without these
    ("py:class", "zarr.core.codec_pipeline.BatchedCodecPipeline"),
    ("py:class", "zarr.core.codec_pipeline.FusedCodecPipeline"),
    ("py:class", "donfig.config_obj.ConfigSet"),
    # private: `dask.enable` documents how it's used
    ("py:class", "zarr_kvikio.dask._ClusterConfigSet"),
]


def setup(app):
    """Register inline roles for release dates and contributors."""
    app.add_role("small", roles.GenericRole("small", nodes.inline))
    app.add_role("smaller", roles.GenericRole("smaller", nodes.inline))
