"""Testy warstwy danych: preprocessing i statystyki."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from garch_btc_sp.data.preprocessing import build_kaggle_returns, compute_log_returns
from garch_btc_sp.stats.descriptive import descriptive_stats
from garch_btc_sp.stats.tests import run_all_tests


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Mała sztuczna ramka cen do szybkich testów."""
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.DataFrame({"A": [100, 110, 105, 120, 115]}, index=idx)


def test_log_returns_length(sample_prices: pd.DataFrame) -> None:
    """Log-zwroty są o jeden wiersz krótsze niż ceny (pierwszy NaN znika)."""
    returns = compute_log_returns(sample_prices)
    assert len(returns) == len(sample_prices) - 1


def test_log_returns_values(sample_prices: pd.DataFrame) -> None:
    """Pierwszy log-zwrot zgadza się z ręcznym wyliczeniem log(110/100)."""
    returns = compute_log_returns(sample_prices)
    expected = np.log(110 / 100)
    assert returns["A"].iloc[0] == pytest.approx(expected)


def test_nonpositive_prices_no_inf() -> None:
    """Ceny niedodatnie (np. ujemna ropa) nie tworzą nieskończoności w zwrotach."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    prices = pd.DataFrame(
        {"OIL": [50.0, -10.0, 40.0, 45.0], "GOLD": [100.0, 101.0, 99.0, 102.0]},
        index=idx,
    )
    returns = compute_log_returns(prices)
    # Kluczowe: zaden zwrot nie moze byc nieskonczonoscia (log z ceny <= 0).
    assert not np.isinf(returns.to_numpy()).any()
    # Zloto liczy sie normalnie, bo ma wylacznie dodatnie ceny.
    assert returns["GOLD"].notna().any()


def test_kaggle_returns_no_missing() -> None:
    """Gotowe log-zwroty Kaggle nie zawierają braków ani nieskończoności."""
    returns = build_kaggle_returns()
    assert not returns.isna().any().any()
    assert not np.isinf(returns.to_numpy()).any()
    assert list(returns.columns) == ["BTC", "SP500"]


def test_descriptive_stats_columns() -> None:
    """Tabela statystyk opisowych ma oczekiwane kolumny i wiersz na aktywo."""
    returns = build_kaggle_returns()
    stats = descriptive_stats(returns)
    assert set(stats.columns) == {"n", "mean", "std", "min", "max", "skew", "kurtosis"}
    assert len(stats) == returns.shape[1]


def test_run_all_tests_shape() -> None:
    """run_all_tests zwraca jeden wiersz na aktywo i komplet kolumn z p-value."""
    returns = build_kaggle_returns()
    result = run_all_tests(returns)
    assert len(result) == returns.shape[1]
    assert "ARCH_LM_pval" in result.columns
