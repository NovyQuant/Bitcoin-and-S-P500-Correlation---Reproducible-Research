"""Visualization functions for DCC-GARCH analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_time_varying_correlation(
        dcc_model,
        asset1: str = "BTC",
        asset2: str = "SP500",
        figsize: tuple[int, int] = (14, 5),
        save_path: str | None = None,
) -> None:
    """Plot time-varying correlation between two assets."""

    corr_series = dcc_model.get_correlation(asset1, asset2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(corr_series.index, corr_series.values, linewidth=1.5, color="steelblue")
    ax.fill_between(corr_series.index, corr_series.values, alpha=0.3, color="steelblue")

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Correlation", fontsize=11)
    ax.set_title(
        f"Dynamic Conditional Correlation: {asset1} - {asset2}",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_all_correlations(
        dcc_model,
        figsize: tuple[int, int] = (16, 10),
        save_path: str | None = None,
) -> None:
    """Plot all pairwise correlations in subplots."""

    corr_dict = dcc_model.time_varying_corr
    num_pairs = len(corr_dict)

    n_cols = 3
    n_rows = int(np.ceil(num_pairs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if num_pairs > 1 else [axes]

    for idx, (pair, corr_series) in enumerate(corr_dict.items()):
        ax = axes[idx]
        asset1, asset2 = pair

        ax.plot(corr_series.index, corr_series.values, linewidth=1, color="steelblue")
        ax.fill_between(corr_series.index, corr_series.values, alpha=0.3, color="steelblue")
        ax.set_title(f"{asset1} - {asset2}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Correlation", fontsize=9)
        ax.grid(True, alpha=0.3)

    for idx in range(num_pairs, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(
        dcc_model,
        period: str = "all",
        figsize: tuple[int, int] = (8, 6),
        save_path: str | None = None,
) -> None:
    """Plot correlation heatmap for a specific period."""

    Q = dcc_model.Q_tensor
    asset_names = list(dcc_model.std_residuals.columns)
    dates = dcc_model.std_residuals.index

    if period == "all":
        idx = len(dates) - 1
    elif period == "recent":
        idx = max(0, len(dates) - 252)
    else:
        idx = 0

    D_t = np.diag(np.sqrt(np.diag(Q[idx])))
    D_t_inv = np.linalg.pinv(D_t)
    rho_t = D_t_inv @ Q[idx] @ D_t_inv
    np.fill_diagonal(rho_t, 1.0)

    corr_df = pd.DataFrame(rho_t, index=asset_names, columns=asset_names)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )

    date_str = dates[idx].strftime("%Y-%m-%d")
    ax.set_title(f"Correlation Matrix ({date_str})", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_rolling_pearson_comparison(
        std_residuals: pd.DataFrame,
        dcc_model,
        asset1: str = "BTC",
        asset2: str = "SP500",
        window: int = 252,
        figsize: tuple[int, int] = (14, 5),
        save_path: str | None = None,
) -> None:
    """Compare DCC correlation with rolling Pearson correlation."""

    dcc_corr = dcc_model.get_correlation(asset1, asset2)

    pearson_corr = std_residuals[[asset1, asset2]].rolling(window).corr().unstack().iloc[:, 0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dcc_corr.index, dcc_corr.values, label="DCC-GARCH", linewidth=2, color="steelblue")
    ax.plot(pearson_corr.index, pearson_corr.values, label=f"Rolling Pearson ({window}d)",
            linewidth=1.5, color="orange", alpha=0.7)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Correlation", fontsize=11)
    ax.set_title(f"DCC vs Pearson Correlation: {asset1} - {asset2}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
