# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib.metadata
import os

project = "wcosmo"
copyright = "2024, Colm Talbot, Amanda Farah"
author = "Colm Talbot, Amanda Farah"
# release = importlib.metadata.version("wcosmo").split("+")[0]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "numpydoc",
    "nbsphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
html_static_path = ["_static"]
exclude_patterns = []

root_doc = "index"

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

autosummary_generate = True
numpydoc_show_class_members = False

# Define the json_url for our version switcher.
json_url = "https://wcosmo.readthedocs.io/en/latest/_static/switcher.json"

# Define the version we use for matching in the version switcher.
version_match = os.environ.get("READTHEDOCS_VERSION")
release = importlib.metadata.version("wcosmo").split("+")[0]
# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
# If it's "latest" → change to "dev" (that's what we want the switcher to call it)
if not version_match or version_match.isdigit() or version_match == "latest":
    # For local development, infer the version to match from the package.
    if "dev" in release or "rc" in release:
        version_match = "latest"
        # We want to keep the relative reference if we are in dev mode
        # but we want the whole url if we are effectively in a released version
        json_url = "_static/switcher.json"
    else:
        version_match = f"v{release}"
elif version_match == "stable":
    version_match = f"v{release}"


html_theme_options = {
    "logo": {"text": "wcosmo"},
    "navbar_center": ["navbar-nav"],
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "use_edit_page_button": True,
    "show_version_warning_banner": True,
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ColmTalbot/wcosmo",
            "icon": "fa-brands fa-github",
        },
    ],
}

html_context = {
    "github_user": "ColmTalbot",
    "github_repo": "wcosmo",
    "github_version": "main",
    "doc_path": "doc/source",
}
