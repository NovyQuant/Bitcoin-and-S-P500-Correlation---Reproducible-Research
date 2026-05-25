"""Command-line entry point for fitting the model pipeline."""

from __future__ import annotations

from garch_btc_sp.models.dcc import DCCModel
from garch_btc_sp.models.univariate_pipeline import fit_univariate_garch


def main() -> None:
    """Fit univariate GARCH models and DCC on the processed Yahoo returns."""

    standardized_residuals, conditional_volatility = fit_univariate_garch()
    dcc = DCCModel(standardized_residuals).fit()

    print("Univariate GARCH fitted.")
    print(f"  standardized_residuals: {standardized_residuals.shape}")
    print(f"  conditional_volatility: {conditional_volatility.shape}")
    print("DCC fitted.")
    print(f"  summary: {dcc.summary()}")


if __name__ == "__main__":
    main()
