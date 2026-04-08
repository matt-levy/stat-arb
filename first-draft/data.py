import itertools

import pandas as pd
import yfinance as yf

from config import LOOKBACK_PERIOD


def download_and_prepare(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=LOOKBACK_PERIOD,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        return pd.DataFrame(columns=[ticker])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].copy()
    df = df.rename(columns={"Close": ticker})
    return df


def generate_pair_candidates(universe_dict):
    pairs = []
    for bucket, tickers in universe_dict.items():
        for y_ticker, x_ticker in itertools.combinations(tickers, 2):
            pairs.append((bucket, y_ticker, x_ticker))
    return pairs
