import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import find_peaks

# ---------- LOAD GOLD FUTURES ----------
def load_futures(ticker="GC=F", start="2000-01-01"):
    df = yf.download(ticker, start=start, progress=False)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.reset_index(inplace=True)
    df.rename(columns={"Close": "Price"}, inplace=True)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df.dropna(subset=["Price"], inplace=True)
    return df

# ---------- RSI ----------
def compute_RSI(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ---------- MACD ----------
def compute_MACD(price, short=12, long=26, signal=9):
    ema_short = price.ewm(span=short, adjust=False).mean()
    ema_long = price.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

# ---------- PEAKS / TROUGHS ----------
def detect_peaks_troughs(series, distance=5, prominence=5):
    peaks, _ = find_peaks(series, distance=distance, prominence=prominence)
    troughs, _ = find_peaks(-series, distance=distance, prominence=prominence)
    return peaks, troughs

# ---------- TECHNICAL INDICATORS PIPE ----------
def add_indicators(df):
    df["MA_7"] = df["Price"].rolling(7).mean()
    df["MA_15"] = df["Price"].rolling(15).mean()
    df["Volatility_30"] = df["Price"].pct_change().rolling(30).std() * 100
    df["Pct_Change"] = df["Price"].pct_change() * 100
    df["RSI_14"] = compute_RSI(df["Price"])
    df["MACD"], df["Signal"] = compute_MACD(df["Price"])
    df["MA_corr"] = df["MA_7"].rolling(30).corr(df["MA_15"])
    peaks, troughs = detect_peaks_troughs(df["Price"])
    df["is_peak"], df["is_trough"] = False, False
    df.loc[peaks, "is_peak"] = True
    df.loc[troughs, "is_trough"] = True
    return df
