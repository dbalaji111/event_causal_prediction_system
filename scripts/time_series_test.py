
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks


def load_gold_futures():
    ticker = "GC=F"  # Gold Futures symbol
    df = yf.download(ticker, start="2020-01-01", progress=False)
    df.reset_index(inplace=True)
    df.rename(columns={"Close": "Price"}, inplace=True)
    return df

y = load_gold_futures()
y.head()