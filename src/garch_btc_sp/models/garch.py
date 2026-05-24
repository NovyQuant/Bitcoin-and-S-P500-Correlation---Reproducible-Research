"""Thin wrappers around :func:`arch.arch_model` for univariate volatility models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox

GARCH_VARIANTS = ("GARCH", "EGARCH", "GJR_GARCH", "APARCH")
DISTRIBUTIONS = ("normal", "t", "skewt")

_VARIANT_KWARGS: dict[str, dict[str, Any]] = {
    "GARCH": {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "power": 2.0},
    "EGARCH": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1, "power": 2.0},
    "GJR_GARCH": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "power": 2.0},
    "APARCH": {"vol": "APARCH", "p": 1, "o": 1, "q": 1, "power": 2.0},
}


@dataclass
class GarchModel:
    """Common interface for the univariate GARCH-family models.

    Parameters
    ----------
    variant : {"GARCH", "EGARCH", "GJR_GARCH", "APARCH"}
        Volatility model family. All variants use p=1 and q=1.
    distribution : {"normal", "t", "skewt"}
        Innovation distribution passed to ``arch.arch_model``.
    scale : float
        Return multiplier used before fitting. We use percentage returns and set
        ``rescale=False`` explicitly, avoiding the silent scaling behaviour of ``arch``.
    """

    variant: str = "GARCH"
    distribution: str = "normal"
    scale: float = 100.0
    result: Any | None = field(default=None, init=False, repr=False)
    asset: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate requested model configuration."""

        if self.variant not in GARCH_VARIANTS:
            raise ValueError(f"Unsupported GARCH variant: {self.variant}")
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    @property
    def model_id(self) -> str:
        """Stable model identifier used in tables and plots."""

        return f"{self.variant}_{self.distribution}"

    def fit(self, returns: pd.Series) -> GarchModel:
        """Fit the model to a single log-return series."""

        clean = returns.dropna().astype(float) * self.scale
        clean.name = returns.name
        self.asset = returns.name

        kwargs = _VARIANT_KWARGS[self.variant]
        model = arch_model(
            clean,
            mean="Constant",
            dist=self.distribution,
            rescale=False,
            **kwargs,
        )
        self.result = model.fit(disp="off")
        return self

    def forecast(self, horizon: int = 1):
        """Forecast conditional variance for the fitted model."""

        self._require_result()
        return self.result.forecast(horizon=horizon, reindex=False)

    def summary(self) -> str:
        """Return the fitted model summary as text."""

        self._require_result()
        return str(self.result.summary())

    @property
    def aic(self) -> float:
        """Akaike information criterion."""

        self._require_result()
        return float(self.result.aic)

    @property
    def bic(self) -> float:
        """Bayesian information criterion."""

        self._require_result()
        return float(self.result.bic)

    @property
    def log_likelihood(self) -> float:
        """Fitted model log-likelihood."""

        self._require_result()
        return float(self.result.loglikelihood)

    @property
    def standardized_residuals(self) -> pd.Series:
        """Standardized residuals used as input to the DCC step."""

        self._require_result()
        residuals = self.result.std_resid.dropna()
        residuals.name = self.asset
        return residuals

    @property
    def conditional_volatility(self) -> pd.Series:
        """Fitted conditional volatility in percentage-return units."""

        self._require_result()
        volatility = self.result.conditional_volatility.dropna()
        volatility.name = self.asset
        return volatility

    def _require_result(self) -> None:
        if self.result is None:
            raise RuntimeError(f"{self.model_id} must be fitted first.")


def fit_model_grid(
    returns: pd.Series,
    variants: tuple[str, ...] = GARCH_VARIANTS,
    distributions: tuple[str, ...] = DISTRIBUTIONS,
) -> list[GarchModel]:
    """Fit every variant/distribution combination for one asset."""

    fitted: list[GarchModel] = []
    for variant in variants:
        for distribution in distributions:
            fitted.append(GarchModel(variant=variant, distribution=distribution).fit(returns))
    return fitted


def compare_models(models: list[GarchModel], lags: int = 20) -> pd.DataFrame:
    """Compare fitted models using information criteria and residual diagnostics."""

    rows: list[dict[str, object]] = []
    for model in models:
        residuals = model.standardized_residuals.dropna()
        lb_resid = acorr_ljungbox(residuals, lags=[lags], return_df=True)
        lb_squared = acorr_ljungbox(residuals**2, lags=[lags], return_df=True)

        rows.append(
            {
                "asset": model.asset,
                "model": model.variant,
                "distribution": model.distribution,
                "aic": model.aic,
                "bic": model.bic,
                "log_likelihood": model.log_likelihood,
                "resid_lb_pvalue": float(lb_resid["lb_pvalue"].iloc[0]),
                "squared_resid_lb_pvalue": float(lb_squared["lb_pvalue"].iloc[0]),
            }
        )

    return pd.DataFrame(rows).sort_values(["asset", "bic"]).reset_index(drop=True)


def select_best_by_bic(comparison: pd.DataFrame) -> pd.DataFrame:
    """Select the lowest-BIC model for each asset."""

    idx = comparison.groupby("asset")["bic"].idxmin()
    return comparison.loc[idx].sort_values("asset").reset_index(drop=True)
