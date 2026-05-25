"""Przetwarzanie cen na log-zwroty oraz zapis przygotowanych danych.

Moduł sprowadza oba źródła (Kaggle i Yahoo) do wspólnego formatu:
ramki log-zwrotów z indeksem dat i jedną kolumną na aktywo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from garch_btc_sp.data.loader import download_yahoo, load_kaggle_csv

PROCESSED_DIR = Path("data/processed")


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Policz dzienne log-zwroty z ramki cen.

    Log-zwrot to ``log(P_t) - log(P_{t-1})``. Pierwszy wiersz (NaN po
    różnicowaniu) jest usuwany.

    Parameters
    ----------
    prices : pd.DataFrame
        Ceny, jedna kolumna na aktywo, indeks dat.

    Returns
    -------
    pd.DataFrame
        Log-zwroty o tych samych kolumnach, o jeden wiersz krótsze.
    """
    valid = prices.where(prices > 0)  # ceny <= 0 (np. ujemna ropa 2020) -> braki
    returns = np.log(valid).diff()
    return returns.dropna(how="all")


def build_kaggle_returns(path: str | Path = "data.csv") -> pd.DataFrame:
    """Zbuduj log-zwroty z oryginalnego zbioru Kaggle (replikacja).

    Zachowuje metodologię oryginału: dzienna częstotliwość z wypełnionym
    S&P (bez wycinania weekendów).

    Returns
    -------
    pd.DataFrame
        Log-zwroty z kolumnami ``BTC`` i ``SP500``.
    """
    prices = load_kaggle_csv(path)
    prices = prices.rename(columns={"BtcPrice": "BTC", "SpPrice": "SP500"})
    returns = compute_log_returns(prices)
    return returns.dropna()


def build_yahoo_returns(
    columns: list[str] | None = None,
    start: str = "2014-09-17",
    end: str | None = None,
    align: bool = True,
) -> pd.DataFrame:
    """Zbuduj log-zwroty z danych Yahoo (rozszerzenie).

    Parameters
    ----------
    columns : list[str] | None
        Które aktywa zostawić (np. ``["BTC", "SP500"]``). ``None`` = wszystkie.
    start, end : str | None
        Zakres dat przekazywany do pobierania.
    align : bool
        Jeśli ``True`` (domyślnie), najpierw usuwa z CEN dni z brakami
        (weekendy/święta), żeby zwroty liczyć między kolejnymi sesjami,
        a nie względem pustych weekendów.

    Returns
    -------
    pd.DataFrame
        Log-zwroty z wybranych kolumn.
    """
    prices = download_yahoo(start=start, end=end)
    if columns is not None:
        prices = prices[columns]
    if align:
        prices = prices.dropna()  # usuń luki PRZED liczeniem zwrotów
    returns = compute_log_returns(prices)
    return returns.dropna()


def save_processed(
    df: pd.DataFrame,
    name: str,
    processed_dir: str | Path = PROCESSED_DIR,
) -> Path:
    """Zapisz przygotowane log-zwroty do pliku Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Dane do zapisania.
    name : str
        Nazwa pliku bez rozszerzenia (np. ``"returns_kaggle"``).
    processed_dir : str | Path
        Katalog docelowy. Domyślnie :data:`PROCESSED_DIR`.

    Returns
    -------
    Path
        Pełna ścieżka zapisanego pliku.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{name}.parquet"
    df.to_parquet(out_path)
    return out_path
