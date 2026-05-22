"""Punkt wejścia: zbuduj cały zbiór danych od surowych cen do log-zwrotów.

Użycie::

    python -m garch_btc_sp.data
"""

from __future__ import annotations

from garch_btc_sp.data.loader import download_yahoo, save_raw
from garch_btc_sp.data.preprocessing import (
    build_kaggle_returns,
    compute_log_returns,
    save_processed,
)


def main() -> None:
    """Pobierz dane, policz log-zwroty i zapisz przygotowane pliki Parquet."""
    print("1/3  Pobieranie cen z Yahoo Finance...")
    prices = download_yahoo()
    save_raw(prices, name="yahoo_prices")

    print("2/3  Log-zwroty z danych Kaggle (replikacja)...")
    kaggle = build_kaggle_returns()
    save_processed(kaggle, name="returns_kaggle")
    print(f"      returns_kaggle: {kaggle.shape[0]} x {kaggle.shape[1]}")

    print("3/3  Log-zwroty z danych Yahoo (rozszerzenie)...")
    yahoo = compute_log_returns(prices).dropna()
    save_processed(yahoo, name="returns_yahoo")
    print(f"      returns_yahoo: {yahoo.shape[0]} x {yahoo.shape[1]}")

    print("Gotowe. Pliki w data/processed/")


if __name__ == "__main__":
    main()
