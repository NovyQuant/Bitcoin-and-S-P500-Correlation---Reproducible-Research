"""Smoke tests for univariate GARCH model wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from garch_btc_sp.models import GARCH_VARIANTS, GarchModel, compare_models, fit_model_grid


def test_garch_model_fit_exposes_expected_properties() -> None:
    """A fitted model exposes criteria, residuals, and conditional volatility."""

    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    returns = pd.Series(rng.normal(0, 0.01, size=len(idx)), index=idx, name="BTC")

    model = GarchModel(variant="GARCH", distribution="normal").fit(returns)

    assert model.asset == "BTC"
    assert np.isfinite(model.aic)
    assert np.isfinite(model.bic)
    assert len(model.standardized_residuals) > 0
    assert len(model.conditional_volatility) > 0


def test_fit_model_grid_and_comparison_table() -> None:
    """The helper fits the requested model grid and returns a sortable table."""

    rng = np.random.default_rng(123)
    idx = pd.date_range("2021-01-01", periods=300, freq="B")
    returns = pd.Series(rng.normal(0, 0.01, size=len(idx)), index=idx, name="SP500")

    models = fit_model_grid(returns, variants=("GARCH", "EGARCH"), distributions=("normal",))
    comparison = compare_models(models, lags=5)

    assert len(models) == 2
    assert set(comparison["model"]) == {"GARCH", "EGARCH"}
    assert {"aic", "bic", "log_likelihood", "resid_lb_pvalue"}.issubset(comparison.columns)


def test_expected_variants_are_available() -> None:
    """The required GARCH-family variants are exposed by the package."""

    assert set(GARCH_VARIANTS) == {"GARCH", "EGARCH", "GJR_GARCH", "APARCH"}
