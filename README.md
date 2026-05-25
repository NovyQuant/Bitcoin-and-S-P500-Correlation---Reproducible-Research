# GARCH Project - Bitcoin and S&P 500 Correlation

Reproducible Research 2026. Port from R (Kaggle) to Python, extended with Oil,
Gold and the VIX volatility index, plus a longer date range.

## Quick Start

```bash
docker pull novyquant/garch-project:latest

# macOS / Linux:
docker run -v "$(pwd)/outputs:/app/outputs" novyquant/garch-project:latest

# Windows (PowerShell):
docker run -v "${PWD}/outputs:/app/outputs" novyquant/garch-project:latest
```

The HTML report appears in the `outputs/` directory on the host. Keep the mount
in quotes — it protects against spaces in the path (e.g. `Quant Projects`).

## What the project does

Reproduction of a GARCH-family correlation analysis between Bitcoin (BTC) and
the S&P 500 index:

- Univariate GARCH(1,1), EGARCH, GJR-GARCH, APARCH with normal / Student-t /
  skewed-t error distributions.
- DCC-GARCH(1,1) per Engle (2002) for time-varying correlation.
- Extended assets: BTC, S&P 500, Oil, Gold, VIX.
- Date range: ~2014/2015 to today.

Original R source: https://www.kaggle.com/code/linhanphm/garch-family-bitcoin-and-s-p-500-correlation
Data source: https://www.kaggle.com/datasets/linhanphm/bitcoin-and-s-and-p-500-historical-prices

## Sphinx Documentation

Auto-generated API docs are built from docstrings in `src/garch_btc_sp/`.

From the published image (no local setup needed) — mount a host directory for
the build output:

```bash
# macOS / Linux:
docker run --rm -v "$(pwd)/docs_html:/app/docs/_build" novyquant/garch-project:latest make docs

# Windows (PowerShell):
docker run --rm -v "${PWD}/docs_html:/app/docs/_build" novyquant/garch-project:latest make docs

# then open docs_html/html/index.html
```

Locally (after the dev install below) just run `make docs` and open
`docs/_build/html/index.html`.

## Local Development

```bash
pip install -e '.[dev]'
pre-commit install

make fit       # run the pipeline
make report    # render the Quarto report into outputs/
make docs      # build Sphinx HTML docs
make lint      # ruff check
make format    # ruff format
make clean     # remove generated outputs
```

## Project Structure

```
src/garch_btc_sp/       Python package (data loading, models, DCC)
data/                   CSV files baked into the Docker image
docs/                   Sphinx documentation source
report.qmd              Quarto report (rendered to outputs/)
Dockerfile              Reproducible build environment
Makefile                Automation entry points
docker-compose.yml      Volume mapping for outputs/
```
