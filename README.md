# Kronos AI + DayTrader Intelligence

An open-source, full-stack financial market forecasting and risk intelligence system. It combines the **Kronos Financial Foundation Model** (transformer-based autoregressive time-series foundation model) with **DayTrader Intelligence** (10,000 Monte Carlo bootstrap session replays, 14-period ATR brackets, and Kelly Criterion position sizing).

Includes a modern, responsive web dashboard with real-time Server-Sent Events (SSE) streaming, dark mode UI, and an embedded searchable catalog of 180+ global and Indian market instruments (NSE, BSE, MCX Commodities, Global Indices, Crypto).

---

## Architecture Overview

```
[ Market Data ] (yfinance)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│                   Execution Pipeline                   │
├──────────────────────────┬─────────────────────────────┤
│   Kronos Foundation AI   │    DayTrader Intelligence   │
│  - Autoregressive Steps  │  - 14-Period ATR Brackets   │
│  - 10-Path Monte Carlo   │  - 10k Bootstrap Replays    │
│  - 5th/50th/95th Bands   │  - Mathematical Expectancy  │
│  - Dual Tokenizer VQ-VAE │  - Kelly Criterion Sizing   │
└──────────────────────────┴─────────────────────────────┘
       │
       ▼
[ FastAPI + SSE Stream ] ──► [ Modern Responsive Dashboard ]
```

### Core Features

1. **Kronos Foundation Transformer:**
   - Evaluates multi-channel price & volume time-series using discrete latent token quantization.
   - Generates multi-trajectory forward paths with 5th, 50th (median), and 95th percentile confidence envelopes.
   - Supports both `Kronos-small` (fast CPU/edge inference) and `Kronos-base` (deep 102M parameter model).

2. **DayTrader TimeMachine & Risk Engine:**
   - **10,000 Bootstrap Replays:** Simulates 10,000 potential trading sessions using empirical return distributions to calculate Target First % vs. Stop First % odds.
   - **14-Period ATR Brackets:** Dynamic stop-loss and multi-R target levels based on recent volatility.
   - **R-Multiple Position Sizing:** Automatically calculates recommended unit sizing and total capital exposure based on your account capital and risk tolerance.

3. **Real-Time Web Dashboard:**
   - **Live Instrument Search:** Search by symbol or company name across 180+ instruments (NIFTY 50, Bank Nifty, MCX Crude Oil, Natural Gas, Gold, Silver, top equities).
   - **Live Activity Feed:** Step-by-step progress streaming via Server-Sent Events with automatic keep-alive.
   - **Forecast History:** Slide-over drawer tracking past analyses with instant recall.

---

## Quickstart (Linux / Cloud VM)

### 1. Clone the Repository
```bash
git clone https://github.com/davidrhis/kronos-daytrader.git
cd kronos-daytrader
```

### 2. Run the Installer
The installer creates a Python 3.10+ virtual environment, installs dependencies, and pre-caches the foundation models from Hugging Face:
```bash
bash install.sh
```

### 3. Start the Dashboard
```bash
bash start.sh
```
Open your browser and navigate to:
```
http://localhost:8080
```

*(Optional) If you have `cloudflared` installed on your server, `start.sh` automatically prints a public tunnel URL to access the dashboard from anywhere.*

---

## Manual Installation

If you prefer installing manually or running on Windows / macOS:

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch application
uvicorn app:app --host 0.0.0.0 --port 8080
```

---

## Hardware & System Requirements

- **Operating System:** Linux (Ubuntu 22.04+, Debian 12+, RHEL/Oracle Linux), macOS, or Windows 11.
- **Architecture:** x86_64 or ARM64 (Apple Silicon, AWS Graviton, Neoverse, etc.).
- **CPU / RAM:** Minimum 2 cores, 4 GB RAM. Recommended 4+ cores, 8 GB RAM.
- **GPU:** Optional. Uses CUDA automatically if an NVIDIA GPU is available; falls back to optimized CPU vectorization seamlessly.

---

## Project Structure

```
kronos-daytrader/
├── app.py                # FastAPI web server & SSE streaming router
├── engine.py             # Inference pipeline & DayTrader calculation engine
├── model/                # Kronos transformer architecture & tokenizer
│   ├── __init__.py
│   ├── kronos.py
│   └── module.py
├── static/               # Frontend assets
│   ├── index.html        # Single-page dashboard UI
│   ├── assets.json       # Search catalog of 180+ tradeable instruments
│   └── charts/           # Output directory for generated chart plots
├── history/              # Local storage for saved historical forecasts
├── requirements.txt      # Python package requirements
├── install.sh            # Automated server setup script
├── start.sh              # Service startup script
└── README.md
```

---

## Configuration & Parameters

The dashboard supports full customization per run:

- **Horizon:** Daily (20 trading days) or Hourly (24 hours).
- **Model Engine:** `Kronos-small` (~10-15s CPU inference) or `Kronos-base` (~45-60s deep transformer).
- **Account Capital:** Define your trading capital (in ₹ or $).
- **Max Risk per Trade:** Set risk threshold (0.5% conservative, 1.0% standard, 1.5% aggressive, 2.0% high risk).

---

## Disclaimer

This software is for educational, analytical, and research purposes only. It is not financial advice. Past statistical performance and foundation model trajectories do not guarantee future market outcomes. Always manage your risk responsibly.

---

## License

MIT License. Open source and free for personal and commercial research.
