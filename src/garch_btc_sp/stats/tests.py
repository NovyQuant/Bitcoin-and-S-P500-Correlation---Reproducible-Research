"""Testy diagnostyczne szeregów log-zwrotów.

Zestaw testów uzasadniających modelowanie typu GARCH:
stacjonarność (ADF), normalność (Jarque-Bera), autokorelacja
(Ljung-Box) oraz efekty ARCH (test Engle'a).
"""

from __future__ import annotations

import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series) -> dict[str, float]:
    """Rozszerzony test Dickeya-Fullera na stacjonarność.

    Hipoteza zerowa: szereg ma pierwiastek jednostkowy (jest niestacjonarny).
    Niskie p-value (< 0.05) oznacza odrzucenie H0, czyli stacjonarność.

    Parameters
    ----------
    series : pd.Series
        Szereg log-zwrotów.

    Returns
    -------
    dict[str, float]
        Statystyka testu i p-value.
    """
    result = adfuller(series.dropna())
    return {"statistic": result[0], "pvalue": result[1]}


def jarque_bera_test(series: pd.Series) -> dict[str, float]:
    """Test Jarque-Bera na normalność rozkładu.

    Hipoteza zerowa: dane pochodzą z rozkładu normalnego.
    Niskie p-value oznacza odrzucenie normalności (grube ogony / skośność).

    Returns
    -------
    dict[str, float]
        Statystyka testu i p-value.
    """
    result = stats.jarque_bera(series.dropna())
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}


def ljung_box_test(series: pd.Series, lags: int = 10) -> dict[str, float]:
    """Test Ljung-Box na autokorelację.

    Hipoteza zerowa: brak autokorelacji do zadanego opóźnienia.
    Stosowany na zwrotach (zwykle brak autokorelacji) oraz na ich
    kwadratach (autokorelacja = sygnał efektów ARCH).

    Parameters
    ----------
    series : pd.Series
        Badany szereg.
    lags : int
        Liczba opóźnień.

    Returns
    -------
    dict[str, float]
        Statystyka testu i p-value dla zadanego opóźnienia.
    """
    result = acorr_ljungbox(series.dropna(), lags=[lags], return_df=True)
    return {
        "statistic": float(result["lb_stat"].iloc[0]),
        "pvalue": float(result["lb_pvalue"].iloc[0]),
    }


def arch_lm_test(series: pd.Series, lags: int = 10) -> dict[str, float]:
    """Test mnożnika Lagrange'a Engle'a na efekty ARCH.

    Hipoteza zerowa: brak heteroskedastyczności warunkowej (brak efektów ARCH).
    Niskie p-value oznacza obecność efektów ARCH, czyli uzasadnia GARCH.

    Returns
    -------
    dict[str, float]
        Statystyka LM i p-value.
    """
    stat, pvalue, _, _ = het_arch(series.dropna(), nlags=lags)
    return {"statistic": float(stat), "pvalue": float(pvalue)}


def run_all_tests(returns: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    """Uruchom wszystkie testy dla każdej kolumny ramki log-zwrotów.

    Parameters
    ----------
    returns : pd.DataFrame
        Log-zwroty, jedna kolumna na aktywo.
    lags : int
        Liczba opóźnień dla testów Ljung-Box i ARCH-LM.

    Returns
    -------
    pd.DataFrame
        Wiersze = aktywa, kolumny = wyniki testów (statystyki i p-value).
    """
    rows = {}
    for col in returns.columns:
        s = returns[col]
        adf = adf_test(s)
        jb = jarque_bera_test(s)
        lb_ret = ljung_box_test(s, lags=lags)
        lb_sq = ljung_box_test(s**2, lags=lags)
        arch = arch_lm_test(s, lags=lags)
        rows[col] = {
            "ADF_stat": adf["statistic"],
            "ADF_pval": adf["pvalue"],
            "JB_stat": jb["statistic"],
            "JB_pval": jb["pvalue"],
            "LB_returns_pval": lb_ret["pvalue"],
            "LB_squared_pval": lb_sq["pvalue"],
            "ARCH_LM_pval": arch["pvalue"],
        }
    return pd.DataFrame(rows).T
