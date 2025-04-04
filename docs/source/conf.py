# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'sispca'
copyright = '2024, Jiayu Su'
author = 'Jiayu Su'

release = 'v1.1.1'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'sphinx.ext.autosectionlabel',
    'autoapi.extension',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx_gallery.load_style',
]

autoapi_dirs = ['../../sispca']
autoapi_add_toctree_entry = False
autoapi_python_class_content = 'both'


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'