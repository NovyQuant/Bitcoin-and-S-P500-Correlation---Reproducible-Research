"""Analiza wstępna (EDA): tabele i wykresy z log-zwrotów.

Uruchomienie z katalogu głównego repo::

    python notebooks/01_eda.py

Tabele wypisują się na konsolę, a wykresy zapisują jako pliki PNG
w katalogu ``notebooks/figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from garch_btc_sp.stats.descriptive import descriptive_stats
from garch_btc_sp.stats.tests import run_all_tests

sns.set_theme(style="whitegrid")
FIG_DIR = Path("notebooks/figures")


def fig_returns_over_time(returns: pd.DataFrame, out: Path) -> None:
    """Zapisz wykres log-zwrotów w czasie (jeden panel na aktywo)."""
    n = returns.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.6 * n + 1), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, returns.columns, strict=True):
        ax.plot(returns.index, returns[col], linewidth=0.6)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel(col)
    axes[0].set_title("Dzienne log-zwroty")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_hist_vs_normal(returns: pd.DataFrame, out: Path) -> None:
    """Zapisz histogramy zwrotów z nałożonym rozkładem normalnym."""
    n = returns.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, returns.columns, strict=True):
        s = returns[col].dropna()
        ax.hist(s, bins=120, density=True, alpha=0.6)
        x = np.linspace(s.min(), s.max(), 300)
        ax.plot(x, stats.norm.pdf(x, s.mean(), s.std()), "r-", linewidth=1.2)
        ax.set_title(col)
    fig.suptitle("Histogram log-zwrotów vs rozkład normalny (czerwony)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_rolling_vol(returns: pd.DataFrame, out: Path, window: int = 30) -> None:
    """Zapisz wykres kroczącej zmienności (odchylenie standardowe)."""
    fig, ax = plt.subplots(figsize=(11, 4))
    returns.rolling(window).std().plot(ax=ax)
    ax.set_title(f"Krocząca zmienność (odchylenie std, okno {window} dni)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_acf(returns: pd.DataFrame, out: Path) -> None:
    """Zapisz wykresy ACF zwrotów oraz ich kwadratów (dowód efektów ARCH)."""
    n = returns.shape[1]
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.5 * n))
    axes = np.atleast_2d(axes)
    for i, col in enumerate(returns.columns):
        plot_acf(returns[col].dropna(), lags=30, ax=axes[i, 0], title=f"ACF zwrotów: {col}")
        plot_acf(
            (returns[col] ** 2).dropna(), lags=30, ax=axes[i, 1], title=f"ACF kwadratów: {col}"
        )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    """Wczytaj dane, wypisz tabele i zapisz wszystkie wykresy."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    kaggle = pd.read_parquet("data/processed/returns_kaggle.parquet")
    yahoo = pd.read_parquet("data/processed/returns_yahoo.parquet")

    print("=== Statystyki opisowe: Kaggle ===")
    print(descriptive_stats(kaggle).round(4).to_string())
    print("\n=== Testy diagnostyczne: Kaggle ===")
    print(run_all_tests(kaggle).round(4).to_string())
    print("\n=== Statystyki opisowe: Yahoo ===")
    print(descriptive_stats(yahoo).round(4).to_string())
    print("\n=== Testy diagnostyczne: Yahoo ===")
    print(run_all_tests(yahoo).round(4).to_string())

    fig_returns_over_time(kaggle, FIG_DIR / "kaggle_returns.png")
    fig_hist_vs_normal(kaggle, FIG_DIR / "kaggle_hist.png")
    fig_rolling_vol(kaggle, FIG_DIR / "kaggle_rolling_vol.png")
    fig_acf(kaggle, FIG_DIR / "kaggle_acf.png")
    fig_returns_over_time(yahoo, FIG_DIR / "yahoo_returns.png")

    print(f"\nWykresy zapisane w: {FIG_DIR}/")


if __name__ == "__main__":
    main()
