import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

project = 'GARCH BTC-SP500'
author = 'Reproducible Research 2026 Team'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'autoapi.extension',
]

autoapi_dirs = ['../src']
autoapi_type = 'python'

html_theme = 'furo'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
