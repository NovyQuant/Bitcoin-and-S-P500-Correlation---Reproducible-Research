"""Punkt wejścia: pobierz dane z Yahoo i zapisz surowe pliki na dysk.

Użycie::

    python -m garch_btc_sp.data
"""

from __future__ import annotations

from garch_btc_sp.data.loader import download_yahoo, save_raw


def main() -> None:
    """Pobierz ceny z Yahoo Finance i zapisz do ``data/raw/yahoo_prices.csv``."""
    print("Pobieranie danych z Yahoo Finance...")
    prices = download_yahoo()
    out_path = save_raw(prices, name="yahoo_prices")
    print(f"Zapisano {prices.shape[0]} wierszy x {prices.shape[1]} kolumn -> {out_path}")


if __name__ == "__main__":
    main()
