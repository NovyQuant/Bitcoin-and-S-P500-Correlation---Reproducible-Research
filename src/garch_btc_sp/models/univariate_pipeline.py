"""Pipeline for fitting univariate GARCH models and extracting standardized residuals."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from garch_btc_sp.data.preprocessing import build_yahoo_returns
from garch_btc_sp.models import compare_models, fit_model_grid, select_best_by_bic

ASSETS = ("BTC", "SP500", "VIX", "OIL", "GOLD")


def load_returns(root: Path | None = None) -> pd.DataFrame:
    """Load log-returns from preprocessed data or build fresh."""

    if root is None:
        root = Path.cwd()

    returns_path = root / "data" / "processed" / "returns_yahoo.parquet"

    if returns_path.exists():
        returns = pd.read_parquet(returns_path)
    else:
        returns = build_yahoo_returns()

    return returns.dropna()


def fit_univariate_garch(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit univariate GARCH models for all assets.

    Returns
    -------
    standardized_residuals : pd.DataFrame
        Shape (N_dates, 5), columns = BTC, SP500, VIX, OIL, GOLD
    conditional_volatility : pd.DataFrame
        Shape (N_dates, 5), same columns
    """

    if root is None:
        root = Path.cwd()

    # Load returns
    returns = load_returns(root)

    # Fit models for each asset
    all_comparisons = []
    best_residuals = {}
    best_volatility = {}

    for asset in ASSETS:
        if asset not in returns.columns:
            print(f"Warning: {asset} not found in returns, skipping")
            continue

        print(f"Fitting univariate GARCH for {asset}...")
        fitted = fit_model_grid(returns[asset])
        comparison = compare_models(fitted)
        all_comparisons.append(comparison)

        # Select best model by BIC
        best_row = select_best_by_bic(comparison).iloc[0]
        best_model = next(
            model
            for model in fitted
            if model.variant == best_row["model"] and model.distribution == best_row["distribution"]
        )

        best_residuals[asset] = best_model.standardized_residuals.rename(asset)
        best_volatility[asset] = best_model.conditional_volatility.rename(asset)

    # Combine results
    standardized_residuals = pd.concat(best_residuals.values(), axis=1).dropna()
    conditional_volatility = pd.concat(best_volatility.values(), axis=1).dropna()

    print(f"Standardized residuals shape: {standardized_residuals.shape}")
    print(f"Conditional volatility shape: {conditional_volatility.shape}")

    return standardized_residuals, conditional_volatility