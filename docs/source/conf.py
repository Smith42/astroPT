# Configuration file for the Sphinx documentation builder.

import os
import shutil
import sys
from pathlib import Path

# --- AstroPTv3 mount -------------------------------------------------
# Copy the AstroPTv3 docs corpus out of the git submodule into this
# source tree at build time (single source of truth: the AstroPTv3 repo,
# pinned by the submodule; update the pin to refresh). Code and env
# directories are ignored -- only the prose trees are mounted.
_here = Path(__file__).resolve().parent
_v3_src = _here.parent.parent / "astroPTv3" / "astro"
_v3_dst = _here / "astropt3"
if _v3_src.exists():
    shutil.copytree(
        _v3_src,
        _v3_dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            ".venv", ".pytest_cache", "src", "tests", "configs",
            "scripts", "wandb", "conf.py", "index.md",
        ),
    )
# ----------------------------------------------------------------------

sys.path.insert(0, os.path.abspath('../../src'))

# Project information
project = 'AstroPT'
copyright = '2025 Michael J. Smith'
author = 'Michael J. Smith'

# The full version, including alpha/beta/rc tags
try:
    from astropt._version import __version__
    release = __version__
except ImportError:
    release = '0.1.0'  # Default fallback version

# Add any Sphinx extension module names here
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    'myst_parser',  # For markdown support
]

# Add any paths that contain templates here
templates_path = ['_templates']

# List of patterns to exclude from source
exclude_patterns = []

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'
html_static_path = []

# Theme options
html_theme_options = {
    'logo_only': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'style_nav_header_background': '#2b2b2b',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Logo
html_static_path = []  # Empty list since we're not using _static
html_logo = "https://github.com/Smith42/astroPT/raw/main/assets/shoggoth_telescope_sticker_2.png"
html_favicon = "https://github.com/Smith42/astroPT/raw/main/assets/shoggoth_telescope_sticker_2.png"

# Cross-project references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# Configure autodoc
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Support for markdown
source_suffix = ['.rst', '.md']

# The mounted AstroPTv3 corpus uses GitHub-style repo-relative links
# that are not Sphinx document targets
suppress_warnings = ['myst.xref_missing']
