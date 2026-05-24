"""Statystyki opisowe szeregów log-zwrotów."""

from __future__ import annotations

import pandas as pd


def descriptive_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """Policz podstawowe statystyki opisowe dla każdej kolumny log-zwrotów.

    Parameters
    ----------
    returns : pd.DataFrame
        Log-zwroty, jedna kolumna na aktywo.

    Returns
    -------
    pd.DataFrame
        Wiersze = aktywa, kolumny = liczność, średnia, odchylenie
        standardowe, minimum, maksimum, skośność oraz kurtoza nadwyżkowa.
    """
    return pd.DataFrame(
        {
            "n": returns.count(),
            "mean": returns.mean(),
            "std": returns.std(),
            "min": returns.min(),
            "max": returns.max(),
            "skew": returns.skew(),
            "kurtosis": returns.kurtosis(),  # nadwyżkowa (Fishera)
        }
    )
