#!/usr/bin/env python3
"""
Prepares historical candlestick data for Indian Markets (NSE / NIFTY)
Formatted specifically for Kronos finetune_csv pipeline:
Columns: timestamps, open, high, low, close, volume, amount
"""

import os
import sys
import pandas as pd
import yfinance as yf

# Default target symbols representing core Indian market dynamics
DEFAULT_TICKERS = [
    "^NSEI",           # NIFTY 50 Benchmark Index
    "RELIANCE.NS",     # Energy & Retail Heavyweight
    "TCS.NS",          # IT Sector Leader
    "HDFCBANK.NS",     # Private Banking Leader
    "ICICIBANK.NS",    # Private Banking
    "INFY.NS",         # IT
    "SBIN.NS",         # PSU Banking
    "TATAMOTORS.NS",   # Auto Sector
    "KALYANKJIL.NS"    # Retail / Jewellery High-Beta
]

def fetch_and_format_symbol(ticker, period="5y", interval="1d"):
    print(f"📡 Ingesting {ticker} ({period}, {interval})...")
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        print(f"⚠️ Warning: No data returned for {ticker}")
        return None
        
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df.reset_index()
    
    # Identify timestamp column
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "timestamps"})
    
    # Format timestamps cleanly
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
    df["timestamps"] = df["timestamps"].dt.strftime("%Y/%m/%d %H:%M")
    
    # Calculate amount = close * volume
    if "amount" not in df.columns:
        if "volume" in df.columns and "close" in df.columns:
            df["amount"] = (df["volume"] * df["close"]).round(2)
        else:
            df["amount"] = 0.0
            
    required_cols = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = 0.0
            
    df = df[required_cols].dropna().drop_duplicates(subset=["timestamps"]).reset_index(drop=True)
    print(f"✅ Formatted {len(df)} candles for {ticker}")
    return df

def generate_datasets(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Pure NIFTY 50 Benchmark Dataset
    nifty_df = fetch_and_format_symbol("^NSEI", period="7y", interval="1d")
    if nifty_df is not None and not nifty_df.empty:
        nifty_path = os.path.join(output_dir, "nifty50_daily.csv")
        nifty_df.to_csv(nifty_path, index=False)
        print(f"🎉 Saved NIFTY 50 dataset to: {nifty_path} ({len(nifty_df)} candles)")

    # 2. Combined Core Indian Equities Dataset
    combined_dfs = []
    for sym in DEFAULT_TICKERS:
        df_sym = fetch_and_format_symbol(sym, period="5y", interval="1d")
        if df_sym is not None and not df_sym.empty:
            combined_dfs.append(df_sym)
            
    if combined_dfs:
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        # Sort sequentially so chronological order is maintained
        combined_path = os.path.join(output_dir, "indian_equities_daily.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"🎉 Saved Multi-Asset Indian Equities dataset to: {combined_path} ({len(combined_df)} candles)")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    generate_datasets(target)
