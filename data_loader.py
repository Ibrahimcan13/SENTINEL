from collections import OrderedDict
from datetime import datetime, timedelta
import os
import pandas as pd
import yfinance as yf

MAX_CACHE_SIZE = 32
_DATA_CACHE: OrderedDict[str, pd.DataFrame] = OrderedDict()


def _normalize_yfinance_columns(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        if ticker and ticker in df.columns.levels[1]:
            df = df.xs(ticker, axis=1, level=1)
        elif ticker and ticker in df.columns.levels[0]:
            df = df.xs(ticker, axis=1, level=0)
        else:
            df.columns = df.columns.get_level_values(0)

    clean_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            col_name = str(col[0])
        else:
            col_name = str(col)
        clean_cols.append(col_name.strip())

    df.columns = clean_cols
    return df


def clean_market_data(df: pd.DataFrame, keep_weekends: bool = False) -> pd.DataFrame:
    if df.empty:
        return df

    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    else:
        df.index = df.index.tz_localize(None)

    df = df.sort_index()

    if not keep_weekends:
        df = df[df.index.dayofweek < 5]

    df = df.dropna(how="all")

    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')

    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0).astype('int64').astype('float32')

    return df


def save_data_to_parquet(
        df: pd.DataFrame, ticker: str, filename: str = None, folder: str = "data") -> str:
    if df.empty:
        print("[Warning] DataFrame is empty. Aborting Parquet save operation.")
        return ""

    try:
        os.makedirs(folder, exist_ok=True)

        if not filename:
            filename = f"{ticker}_cache.parquet"
        elif not filename.endswith(".parquet"):
            filename = f"{os.path.splitext(filename)[0]}.parquet"

        filepath = os.path.join(folder, filename)

        df.to_parquet(filepath, compression="zstd", engine="pyarrow")
        print(f"[Sentinel] Data successfully cached with ZSTD compression: {filepath}")
        return filepath

    except Exception as e:
        print(f"[Error] Failed to save Parquet file ({type(e).__name__}): {e}")
        return ""


def load_local_data(filename: str, folder: str = "data") -> pd.DataFrame:
    if not filename.endswith(".parquet"):
        filename = f"{os.path.splitext(filename)[0]}.parquet"

    filepath = os.path.join(folder, filename)

    try:
        if os.path.exists(filepath):
            df = pd.read_parquet(filepath, engine="pyarrow")
            return clean_market_data(df)
        else:
            return pd.DataFrame()

    except Exception as e:
        print(f"[Error] Unexpected error while reading Parquet file: {e}")
        return pd.DataFrame()


def _manage_memory_cache(key: str, df: pd.DataFrame):
    if key in _DATA_CACHE:
        _DATA_CACHE.move_to_end(key)
    _DATA_CACHE[key] = df.copy()

    if len(_DATA_CACHE) > MAX_CACHE_SIZE:
        _DATA_CACHE.popitem(last=False)


def _fetch_single_ticker(ticker: str, start_date: str, end_date: str, folder: str = "data") -> pd.DataFrame:
    cache_key = f"{ticker}_{start_date}_{end_date}"

    if cache_key in _DATA_CACHE:
        print(f"[Sentinel] [Memory Cache Hit] Returning cached data for {ticker}...")
        _DATA_CACHE.move_to_end(cache_key)
        return _DATA_CACHE[cache_key].copy()

    filename = f"{cache_key}.parquet"
    local_df = load_local_data(filename, folder=folder)
    if not local_df.empty:
        print(f"[Sentinel] [Disk Cache Hit] Loaded {ticker} from local ZSTD Parquet storage.")
        _manage_memory_cache(cache_key, local_df)
        return local_df

    print(f"[Sentinel] Fetching data from API for ticker: {ticker} ({start_date} to {end_date})...")

    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if df.empty:
            print(f"[Error] No data returned for ticker '{ticker}'. Please check symbol or date range.")
            return pd.DataFrame()

        df = _normalize_yfinance_columns(df, ticker=ticker)
        df = clean_market_data(df)

        print(f"[Sentinel] Successfully downloaded {len(df)} rows for {ticker} (UTC Normalized).")

        save_data_to_parquet(df, ticker=ticker, filename=filename, folder=folder)
        _manage_memory_cache(cache_key, df)

        return df

    except Exception as e:
        print(f"[Error] An unexpected error occurred while fetching {ticker} ({type(e).__name__}): {e}")
        return pd.DataFrame()


def fetch_market_data(
        tickers: str | list[str],
        start_date: str = None,
        end_date: str = None,
        days_back: int = 365,
        folder: str = "data",
        combine_into_single_df: bool = False
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    if end_date is None:
        end_dt = datetime.now()
        end_date = end_dt.strftime("%Y-%m-%d")
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if start_date is None:
        start_date = (end_dt - timedelta(days=days_back)).strftime("%Y-%m-%d")

    if isinstance(tickers, str):
        ticker_list = [t.strip().upper() for t in tickers.replace(",", " ").split() if t.strip()]
    else:
        ticker_list = [t.strip().upper() for t in tickers if t.strip()]

    if not ticker_list:
        print("[Error] No valid ticker symbol provided.")
        return pd.DataFrame()

    if len(ticker_list) == 1:
        return _fetch_single_ticker(ticker_list[0], start_date, end_date, folder=folder)

    results = {}
    for ticker in ticker_list:
        df = _fetch_single_ticker(ticker, start_date, end_date, folder=folder)
        if not df.empty:
            results[ticker] = df

    if combine_into_single_df and results:
        close_series = [df['Close'].rename(ticker) for ticker, df in results.items()]
        combined_df = pd.concat(close_series, axis=1)
        return combined_df

    return results