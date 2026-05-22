"""Wczytywanie danych: lokalny zbiór z Kaggle oraz pobieranie z Yahoo Finance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

# Mapowanie czytelnych nazw na tickery Yahoo Finance.
YAHOO_TICKERS: dict[str, str] = {
    "BTC": "BTC-USD",
    "SP500": "^GSPC",
    "VIX": "^VIX",
    "OIL": "CL=F",
    "GOLD": "GC=F",
}

# Katalogi na dane (względem korzenia repozytorium).
RAW_DIR = Path("data/raw")


def load_kaggle_csv(path: str | Path = "data.csv") -> pd.DataFrame:
    """Wczytaj oryginalny zbiór z Kaggle (BTC i S&P 500).

    Parameters
    ----------
    path : str | Path
        Ścieżka do pliku CSV. Domyślnie ``data.csv`` w korzeniu repo.

    Returns
    -------
    pd.DataFrame
        Ramka z indeksem typu daty (posortowanym rosnąco) oraz kolumnami
        cen ``BtcPrice`` i ``SpPrice``.
    """
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.set_index("Date").sort_index()
    return df


def download_yahoo(
    tickers: dict[str, str] | None = None,
    start: str = "2014-09-17",
    end: str | None = None,
) -> pd.DataFrame:
    """Pobierz dzienne ceny zamknięcia z Yahoo Finance.

    Parameters
    ----------
    tickers : dict[str, str] | None
        Mapowanie nazwa -> ticker Yahoo. Domyślnie :data:`YAHOO_TICKERS`.
    start : str
        Data początkowa (format ``RRRR-MM-DD``). Domyślnie 2014-09-17,
        bo wtedy zaczyna się historia BTC-USD na Yahoo.
    end : str | None
        Data końcowa. ``None`` oznacza dzień dzisiejszy.

    Returns
    -------
    pd.DataFrame
        Ceny zamknięcia, jedna kolumna na aktywo (nazwy z kluczy ``tickers``).
    """
    if tickers is None:
        tickers = YAHOO_TICKERS

    raw = yf.download(
        list(tickers.values()),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    close = raw["Close"].copy()
    # Zamień tickery Yahoo z powrotem na czytelne nazwy.
    ticker_to_name = {v: k for k, v in tickers.items()}
    close = close.rename(columns=ticker_to_name)
    close.index.name = "Date"
    return close.sort_index()


def save_raw(df: pd.DataFrame, name: str, raw_dir: str | Path = RAW_DIR) -> Path:
    """Zapisz ramkę do pliku CSV w katalogu surowych danych.

    Parameters
    ----------
    df : pd.DataFrame
        Dane do zapisania.
    name : str
        Nazwa pliku bez rozszerzenia (np. ``"yahoo_prices"``).
    raw_dir : str | Path
        Katalog docelowy. Domyślnie :data:`RAW_DIR`.

    Returns
    -------
    Path
        Pełna ścieżka zapisanego pliku.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{name}.csv"
    df.to_csv(out_path)
    return out_path
