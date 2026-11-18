#!/usr/bin/env python3
"""
Groww Instrument Master Synchronizer for Kronos AI + DayTrader
Fetches official live instrument list from Groww public master endpoint:
https://growwapi-assets.groww.in/instruments/instrument.csv

Extracts all active NSE Equities, Major Indices, MCX Commodities, and F&O symbols.
Requires zero authentication or private API keys.
"""

import os
import sys
import csv
import json
import urllib.request

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

GROWW_INSTRUMENTS_URL = "https://growwapi-assets.groww.in/instruments/instrument.csv"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "assets.json")

BENCHMARK_INDICES = [
    {"s": "^NSEI", "n": "NIFTY 50 Index", "ex": "Groww NSE", "cat": "INDEX"},
    {"s": "^NSEBANK", "n": "NIFTY Bank Index", "ex": "Groww NSE", "cat": "INDEX"},
    {"s": "^CNXIT", "n": "NIFTY IT Index", "ex": "Groww NSE", "cat": "INDEX"},
    {"s": "^BSESN", "n": "BSE SENSEX Index", "ex": "Groww BSE", "cat": "INDEX"},
    {"s": "NIFTY_FIN_SERVICE.NS", "n": "NIFTY Financial Services", "ex": "Groww NSE", "cat": "INDEX"},
    {"s": "NIFTYMID50.NS", "n": "NIFTY Midcap 50", "ex": "Groww NSE", "cat": "INDEX"},
]

BENCHMARK_COMMODITIES = [
    {"s": "CL=F", "n": "Crude Oil (MCX / WTI)", "ex": "MCX / NYMEX", "cat": "COMMODITY"},
    {"s": "NG=F", "n": "Natural Gas (MCX / Henry Hub)", "ex": "MCX / NYMEX", "cat": "COMMODITY"},
    {"s": "GC=F", "n": "Gold (MCX / COMEX)", "ex": "MCX / COMEX", "cat": "COMMODITY"},
    {"s": "SI=F", "n": "Silver (MCX / COMEX)", "ex": "MCX / COMEX", "cat": "COMMODITY"},
    {"s": "HG=F", "n": "Copper (MCX / COMEX)", "ex": "MCX / COMEX", "cat": "COMMODITY"},
    {"s": "BTC-USD", "n": "Bitcoin", "ex": "Crypto (USD)", "cat": "CRYPTO"},
    {"s": "ETH-USD", "n": "Ethereum", "ex": "Crypto (USD)", "cat": "CRYPTO"}
]

PRIORITY_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL",
    "ITC", "KALYANKJIL", "TATAMOTORS", "LT", "HINDUNILVR", "BAJFINANCE", "MARUTI",
    "AXISBANK", "SUNPHARMA", "TITAN", "HEXAGON", "ZOMATO", "SWIGGY", "OLAELEC",
    "JIOFIN", "TRENT", "BEL", "HAL", "COALINDIA", "NTPC", "POWERGRID"
]

def sync_groww_instruments(target_file=OUTPUT_PATH):
    print(f"📡 Fetching Groww official instrument master from {GROWW_INSTRUMENTS_URL}...")
    req = urllib.request.Request(GROWW_INSTRUMENTS_URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        reader = csv.DictReader(line.decode('utf-8', errors='ignore') for line in resp)
        
        fno_underlyings = set()
        equities_map = {}
        
        for row in reader:
            exchange = row.get("exchange", "").strip()
            segment = row.get("segment", "").strip()
            itype = row.get("instrument_type", "").strip()
            sym = row.get("trading_symbol", "").strip()
            name = row.get("name", "").strip()
            series = row.get("series", "").strip()
            
            # Record FNO underlyings
            if exchange == "NSE" and segment == "FNO":
                underlying = row.get("underlying_symbol", "").strip()
                if underlying:
                    fno_underlyings.add(underlying)
                    
            # Collect NSE Cash equities (EQ: Regular, BE: Trade-to-trade, SM: SME)
            if exchange == "NSE" and segment == "CASH" and series in ["EQ", "BE", "SM", "ST"]:
                if sym and sym not in equities_map:
                    clean_name = name if name else sym
                    equities_map[sym] = {
                        "s": f"{sym}.NS",
                        "n": clean_name,
                        "ex": "Groww NSE",
                        "cat": "EQUITY",
                        "fno": False
                    }
                    
    print(f"✅ Download complete. Discovered {len(equities_map)} NSE Cash stocks and {len(fno_underlyings)} F&O underlyings.")
    
    # Mark F&O eligible stocks
    for sym, item in equities_map.items():
        if sym in fno_underlyings:
            item["fno"] = True
            item["ex"] = "Groww NSE (F&O)"

    # Build assembled catalog
    catalog = []
    
    # 1. Benchmarks
    for b in BENCHMARK_INDICES:
        catalog.append(b)
        
    for c in BENCHMARK_COMMODITIES:
        catalog.append(c)
        
    # 2. Priority Stocks
    seen = {item["s"] for item in catalog}
    for p in PRIORITY_STOCKS:
        if p in equities_map:
            eq = equities_map[p]
            if eq["s"] not in seen:
                catalog.append(eq)
                seen.add(eq["s"])
                
    # 3. Remaining F&O stocks sorted alphabetically
    fno_stocks = [v for k, v in sorted(equities_map.items()) if v.get("fno") and v["s"] not in seen]
    for fs in fno_stocks:
        catalog.append(fs)
        seen.add(fs["s"])
        
    # 4. Remaining NSE cash equities sorted alphabetically
    other_stocks = [v for k, v in sorted(equities_map.items()) if v["s"] not in seen]
    for os_item in other_stocks:
        catalog.append(os_item)
        seen.add(os_item["s"])

    # Assign indexed IDs for fast client lookups
    for idx, item in enumerate(catalog):
        item["idx"] = idx

    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=None, separators=(',', ':'))

    size_kb = os.path.getsize(target_file) / 1024
    print(f"🎉 Generated {len(catalog)} instruments in {target_file} ({size_kb:.1f} KB)")
    return catalog

if __name__ == "__main__":
    out = OUTPUT_PATH
    if len(sys.argv) > 1:
        out = sys.argv[1]
    sync_groww_instruments(out)
