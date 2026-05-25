"""DCC-GARCH model implementation after Engle (2002)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class DCCModel:
    """
    Dynamic Conditional Correlation GARCH(1,1) model.

    Implements the two-step procedure by Engle (2002):
    1. Fit univariate GARCH models (external, provides standardized residuals)
    2. Estimate DCC parameters (alpha, beta) via ML on correlation dynamics

    Parameters
    ----------
    std_residuals : pd.DataFrame
        Standardized residuals from univariate GARCH, shape (T, N)
    """

    std_residuals: pd.DataFrame
    alpha: float = field(default=None, init=False)
    beta: float = field(default=None, init=False)
    time_varying_corr: pd.DataFrame = field(default=None, init=False, repr=False)
    result: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Validate input."""
        if self.std_residuals.empty or self.std_residuals.shape[1] < 2:
            raise ValueError("Need at least 2 assets for DCC")

    def fit(self) -> DCCModel:
        """
        Fit DCC parameters using maximum likelihood.

        Procedure:
        1. Compute unconditional correlation (Q_bar)
        2. Initialize Q_t recursively
        3. Optimize alpha, beta to maximize log-likelihood
        4. Store time-varying correlations
        """

        residuals = self.std_residuals.values  # (T, N)
        T, N = residuals.shape

        # Step 1: Unconditional correlation matrix (sample average)
        Q_bar = np.corrcoef(residuals.T)

        # Step 2: Initialize Q recursively and get series for likelihood
        def get_Q_series(alpha: float, beta: float) -> np.ndarray:
            """Recursive Q_t evolution."""
            Q = np.zeros((T, N, N))
            Q[0] = Q_bar

            for t in range(1, T):
                # Q_t recursion from Engle (2002), driven by lagged residual outer products.
                outer_prod = np.outer(residuals[t - 1], residuals[t - 1])
                Q[t] = (1 - alpha - beta) * Q_bar + alpha * outer_prod + beta * Q[t - 1]

            return Q

        # Step 3: Log-likelihood function
        def neg_log_likelihood(params: np.ndarray) -> float:
            alpha, beta = params

            # Constraints: alpha >= 0, beta >= 0, alpha + beta < 1
            if alpha < 0 or beta < 0 or (alpha + beta) >= 1.0:
                return 1e10

            Q = get_Q_series(alpha, beta)
            ll = 0.0

            for t in range(T):
                # Standardize Q to correlation matrix
                D_t = np.diag(np.sqrt(np.diag(Q[t])))
                D_t_inv = np.linalg.pinv(D_t)

                if np.linalg.cond(D_t_inv) > 1e10:
                    return 1e10

                rho_t = D_t_inv @ Q[t] @ D_t_inv

                # Ensure valid correlation (diagonal = 1)
                np.fill_diagonal(rho_t, 1.0)

                try:
                    det_rho = np.linalg.det(rho_t)
                    if det_rho <= 0:
                        return 1e10

                    # Log-likelihood: -0.5 * (log|rho_t| + eps_t^T * rho_t^{-1} * eps_t)
                    ll -= 0.5 * (np.log(det_rho) +
                                 residuals[t] @ np.linalg.inv(rho_t) @ residuals[t].T)
                except np.linalg.LinAlgError:
                    return 1e10

            return -ll

        # Step 4: Optimize
        x0 = np.array([0.01, 0.95])  # Initial guess
        bounds = [(0, 0.3), (0.5, 0.99)]

        result = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500}
        )

        if not result.success:
            print(f"Warning: DCC optimization did not fully converge: {result.message}")

        self.alpha, self.beta = result.x
        self.result = result

        # Check stationarity
        if self.alpha + self.beta >= 1.0:
            print(f"Warning: alpha + beta = {self.alpha + self.beta:.4f} >= 1.0 (non-stationary)")

        # Step 5: Compute final time-varying correlations
        self._compute_time_varying_corr()

        return self

    def _compute_time_varying_corr(self) -> None:
        """Compute and store time-varying correlation matrix."""

        residuals = self.std_residuals.values
        T, N = residuals.shape

        # Recompute Q with fitted parameters
        Q_bar = np.corrcoef(residuals.T)
        Q = np.zeros((T, N, N))
        Q[0] = Q_bar

        for t in range(1, T):
            outer_prod = np.outer(residuals[t - 1], residuals[t - 1])
            Q[t] = (1 - self.alpha - self.beta) * Q_bar + \
                   self.alpha * outer_prod + \
                   self.beta * Q[t - 1]

        # Store correlations (tensor form)
        self.Q_tensor = Q

        # Extract diagonal correlations (e.g., for BTC-SP500)
        # Convert to DataFrame for easier handling
        dates = self.std_residuals.index
        asset_names = self.std_residuals.columns

        # For now, store the full tensor
        self.time_varying_corr = self._extract_pairwise_corr(asset_names, dates)

    def _extract_pairwise_corr(
            self, asset_names: list[str] | tuple[str, ...], dates: pd.DatetimeIndex
    ) -> dict[tuple[str, str], pd.Series]:
        """Extract pairwise correlations from Q tensor."""

        Q = self.Q_tensor
        T, N, _ = Q.shape

        corr_dict = {}

        for i in range(N):
            for j in range(i + 1, N):
                asset_i, asset_j = asset_names[i], asset_names[j]

                # Extract correlation between assets i and j
                corr_series = np.zeros(T)
                for t in range(T):
                    D_t = np.diag(np.sqrt(np.diag(Q[t])))
                    D_t_inv = np.linalg.pinv(D_t)
                    rho_t = D_t_inv @ Q[t] @ D_t_inv
                    corr_series[t] = rho_t[i, j]

                corr_dict[(asset_i, asset_j)] = pd.Series(
                    corr_series, index=dates, name=f"{asset_i}-{asset_j}"
                )

        return corr_dict

    def get_correlation(self, asset1: str, asset2: str) -> pd.Series:
        """Get time-varying correlation between two specific assets."""

        key = tuple(sorted([asset1, asset2]))
        if key not in self.time_varying_corr:
            available = list(self.time_varying_corr.keys())
            raise ValueError(f"Correlation {key} not found. Available: {available}")

        return self.time_varying_corr[key]

    def summary(self) -> dict[str, Any]:
        """Return fitted DCC parameters and diagnostics."""

        if self.alpha is None:
            raise RuntimeError("Model must be fitted first")

        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "alpha_plus_beta": self.alpha + self.beta,
            "num_assets": self.std_residuals.shape[1],
            "num_obs": self.std_residuals.shape[0],
            "optimization_success": self.result.success,
            "optimization_message": self.result.message,
        }
