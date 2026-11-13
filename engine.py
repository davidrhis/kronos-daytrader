import os
import sys
import time
import math
import json
import uuid
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model.kronos import Kronos, KronosTokenizer, KronosPredictor, sample_from_logits, calc_time_stamps

PREDICTORS = {}

def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

def get_predictor(model_name="small", logger=None):
    model_id = "NeoQuasar/Kronos-base" if model_name == "base" else "NeoQuasar/Kronos-small"
    if model_id not in PREDICTORS:
        if logger:
            logger(f"Loading {model_id} model weights into memory...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained(model_id)
        dev = get_device()
        PREDICTORS[model_id] = KronosPredictor(model, tokenizer, device=dev, max_context=512)
        if logger:
            logger(f"Initialized {model_id} on {dev}.")
    return PREDICTORS[model_id]

def fast_auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len, clip=5, T=1.0, top_k=0, top_p=0.9, sample_count=10, logger=None):
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        device = x.device
        
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)

        x_token = tokenizer.encode(x, half=True)
        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)
            if logger and (i + 1) % 5 == 0:
                logger(f"⚡ Kronos transformer step {i + 1}/{pred_len} synthesized...")

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous()
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        return preds[0, :, -pred_len:, :]

def predict_paths(predictor, df, x_timestamp, y_timestamp, pred_len, paths=10, logger=None):
    df = df.copy()
    if predictor.vol_col not in df.columns:
        df[predictor.vol_col] = 0.0
        df[predictor.amt_vol] = 0.0
    if predictor.amt_vol not in df.columns and predictor.vol_col in df.columns:
        df[predictor.amt_vol] = df[predictor.vol_col] * df[predictor.price_cols].mean(axis=1)

    x_time_df = calc_time_stamps(x_timestamp)
    y_time_df = calc_time_stamps(y_timestamp)

    x = df[predictor.price_cols + [predictor.vol_col, predictor.amt_vol]].values.astype(np.float32)
    x_stamp = x_time_df.values.astype(np.float32)
    y_stamp = y_time_df.values.astype(np.float32)

    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    x = (x - x_mean) / (x_std + 1e-5)
    x = np.clip(x, -predictor.clip, predictor.clip)

    x = torch.from_numpy(x[np.newaxis, :]).to(predictor.device)
    x_stamp = torch.from_numpy(x_stamp[np.newaxis, :]).to(predictor.device)
    y_stamp = torch.from_numpy(y_stamp[np.newaxis, :]).to(predictor.device)

    preds = fast_auto_regressive_inference(
        predictor.tokenizer,
        predictor.model,
        x,
        x_stamp,
        y_stamp,
        predictor.max_context,
        pred_len,
        predictor.clip,
        T=1.0,
        top_p=0.9,
        sample_count=paths,
        logger=logger
    )
    preds = preds * (x_std + 1e-5) + x_mean
    closes = preds[:, :, 3]
    return closes

def calculate_atr(df, period=14):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) > 0 else float(close[-1] * 0.015)
    atr = np.mean(tr[-period:])
    return float(atr)

def calculate_brackets_and_risk(df, capital=100000.0, risk_pct=1.0):
    last_close = float(df['close'].iloc[-1])
    atr = calculate_atr(df, period=14)
    
    stop_distance = max(atr * 1.5, last_close * 0.008)
    stop_loss = round(last_close - stop_distance, 2)
    risk_per_share = round(last_close - stop_loss, 2)
    
    target_1r = round(last_close + 1.0 * risk_per_share, 2)
    target_1_5r = round(last_close + 1.5 * risk_per_share, 2)
    target_2r = round(last_close + 2.0 * risk_per_share, 2)
    
    max_risk_amount = round(capital * (risk_pct / 100.0), 2)
    shares = math.floor(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
    exposure = round(shares * last_close, 2)
    
    return {
        "entry": round(last_close, 2),
        "atr": round(atr, 2),
        "stop_loss": stop_loss,
        "risk_per_share": risk_per_share,
        "target_1r": target_1r,
        "target_1_5r": target_1_5r,
        "target_2r": target_2r,
        "reward_risk_ratio": 2.0,
        "capital": capital,
        "risk_pct": risk_pct,
        "max_risk_amount": max_risk_amount,
        "recommended_shares": shares,
        "capital_exposure": exposure
    }

def run_timemachine_simulation(df, entry, stop, target, replays=10000, steps=20):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    returns_high = highs[1:] / closes[:-1]
    returns_low = lows[1:] / closes[:-1]
    returns_close = closes[1:] / closes[:-1]
    n_history = len(returns_close)
    
    if n_history < 10:
        return {
            "replays": replays,
            "target_first_pct": 50.0,
            "stop_first_pct": 50.0,
            "neither_pct": 0.0,
            "expectancy_r": 0.0,
            "kelly_pct": 0.0,
            "verdict": "Insufficient historical candles for deep bootstrap"
        }
        
    indices = np.random.choice(n_history, size=(replays, steps), replace=True)
    target_hit = np.zeros(replays, dtype=bool)
    stop_hit = np.zeros(replays, dtype=bool)
    current_prices = np.full(replays, entry, dtype=np.float64)
    active = np.ones(replays, dtype=bool)
    
    for s in range(steps):
        step_idx = indices[:, s]
        bar_high = current_prices * returns_high[step_idx]
        bar_low = current_prices * returns_low[step_idx]
        bar_close = current_prices * returns_close[step_idx]
        
        hit_stop_now = active & (bar_low <= stop)
        hit_target_now = active & ~hit_stop_now & (bar_high >= target)
        
        stop_hit |= hit_stop_now
        target_hit |= hit_target_now
        active &= ~(hit_stop_now | hit_target_now)
        current_prices = bar_close
        
    target_count = int(np.sum(target_hit))
    stop_count = int(np.sum(stop_hit))
    neither_count = replays - target_count - stop_count
    
    target_pct = round((target_count / replays) * 100, 1)
    stop_pct = round((stop_count / replays) * 100, 1)
    neither_pct = round((neither_count / replays) * 100, 1)
    
    risk_val = entry - stop
    reward_val = target - entry
    r_mult = reward_val / risk_val if risk_val > 0 else 2.0
    
    expectancy = round(((target_pct / 100.0) * r_mult) - ((stop_pct / 100.0) * 1.0), 2)
    win_rate = target_pct / 100.0
    kelly = round((win_rate - ((1.0 - win_rate) / r_mult)) * 100, 1) if r_mult > 0 else 0.0
    
    if target_pct >= 55.0 and expectancy > 0.3:
        verdict = "STRONG EDGE (High Probability Setup)"
    elif target_pct >= 45.0 and expectancy >= 0.0:
        verdict = "MODERATE EDGE (Positive Mathematical Expectancy)"
    else:
        verdict = "NEGATIVE EXPECTANCY (Caution / Low Win Rate)"
        
    return {
        "replays": replays,
        "target_first_pct": target_pct,
        "stop_first_pct": stop_pct,
        "neither_pct": neither_pct,
        "expectancy_r": expectancy,
        "kelly_pct": max(0.0, kelly),
        "verdict": verdict
    }

def run_full_analysis(ticker, timeframe="daily", model_name="small", capital=100000.0, risk_pct=1.0, paths=10, charts_dir="static/charts", logger=None):
    if logger is None:
        logger = lambda msg: print(msg)
        
    clean_ticker = ticker.strip().upper()
    daily = (timeframe.lower() in ["daily", "d", "day"])
    lookback = 120 if daily else 96
    pred_len = 20 if daily else 24
    interval, period = ("1d", "1y") if daily else ("1h", "60d")
    
    logger(f"📡 Step 1/4: Ingesting candles for {clean_ticker} ({interval} interval)...")
    raw = yf.download(clean_ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"No market data returned for '{clean_ticker}'. For Indian stocks, ensure '.NS' is appended (e.g. RELIANCE.NS, HEXAGON.NS, ^NSEI).")
        
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    df = raw[["open", "high", "low", "close", "volume"]].dropna().tail(lookback).reset_index()
    df = df.rename(columns={df.columns[0]: "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
    
    last_close = float(df["close"].iloc[-1])
    last_time = str(df["timestamps"].iloc[-1])[:10] if daily else str(df["timestamps"].iloc[-1])
    logger(f"✅ Ingested {len(df)} candles. Latest price: {last_close:.2f} ({last_time})")
    
    # 1. Day Trader Bracket and Risk Sizing
    logger("🎯 Step 2/4: DayTrader Engine calculating 14-bar ATR brackets & position sizing...")
    brackets = calculate_brackets_and_risk(df, capital=capital, risk_pct=risk_pct)
    
    # 2. Day Trader TimeMachine Monte Carlo Replay
    logger("⏳ Step 3/4: DayTrader TimeMachine simulating 10,000 bootstrap session replays...")
    timemachine = run_timemachine_simulation(
        df,
        entry=brackets["entry"],
        stop=brackets["stop_loss"],
        target=brackets["target_2r"],
        replays=10000,
        steps=pred_len
    )
    
    # 3. Kronos Deep Learning Forecasting
    logger(f"🤖 Step 4/4: Kronos Foundation Model ({model_name}) generating {paths} trajectories...")
    predictor = get_predictor(model_name, logger=logger)
    x_ts = df["timestamps"]
    last_ts_val = x_ts.iloc[-1]
    if daily:
        y_ts = pd.Series([last_ts_val + pd.offsets.BDay(i + 1) for i in range(pred_len)])
    else:
        y_ts = pd.Series([last_ts_val + pd.Timedelta(hours=i + 1) for i in range(pred_len)])
        
    t0 = time.time()
    closes = predict_paths(predictor, df, x_ts, y_ts, pred_len, paths=paths, logger=logger)
    logger(f"⚡ Kronos inference finished in {time.time() - t0:.1f}s.")
    
    df_closes = pd.DataFrame(closes, columns=y_ts).astype(float)
    band_df = pd.DataFrame({
        "timestamp": [str(t)[:10] if daily else str(t) for t in y_ts],
        "low_5%": df_closes.quantile(0.05).round(2).values,
        "likely_50%": df_closes.median().round(2).values,
        "high_95%": df_closes.quantile(0.95).round(2).values
    })
    
    # 4. Generate Plot
    logger("📊 Rendering high-resolution analytics chart...")
    os.makedirs(charts_dir, exist_ok=True)
    job_id = str(uuid.uuid4())[:8]
    safe_ticker = clean_ticker.replace("^", "").replace("=", "_").replace("/", "_").replace(":", "_")
    chart_filename = f"{safe_ticker}_{job_id}.png"
    chart_filepath = os.path.join(charts_dir, chart_filename)
    
    plt.figure(figsize=(12, 5.8), facecolor='#0f172a')
    ax = plt.subplot(111)
    ax.set_facecolor('#0f172a')
    
    hist_subset = df.tail(80)
    ax.plot(hist_subset["timestamps"], hist_subset["close"], color="#cbd5e1", linewidth=1.8, label="Historical Close")
    
    ax.fill_between(y_ts, band_df["low_5%"], band_df["high_95%"], color="#38bdf8", alpha=0.22, label="Kronos 90% Confidence Band")
    ax.plot(y_ts, band_df["likely_50%"], color="#38bdf8", linewidth=2.2, label="Kronos Median Forecast")
    
    ax.axhline(brackets["entry"], color="#facc15", linestyle=":", linewidth=1.2, alpha=0.8, label=f"Entry: {brackets['entry']}")
    ax.axhline(brackets["stop_loss"], color="#f87171", linestyle="--", linewidth=1.5, alpha=0.85, label=f"Stop Loss (1.5x ATR): {brackets['stop_loss']}")
    ax.axhline(brackets["target_2r"], color="#4ade80", linestyle="--", linewidth=1.5, alpha=0.85, label=f"Target 2R: {brackets['target_2r']}")
    
    ax.set_title(f"{clean_ticker} — Kronos Foundation Model & DayTrader Analytics", color="#f8fafc", fontsize=14, pad=12, weight='bold')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.grid(True, color='#1e293b', linestyle='--', alpha=0.7)
    
    legend = ax.legend(facecolor='#1e293b', edgecolor='#334155', fontsize=8.5)
    for text in legend.get_texts():
        text.set_color('#e2e8f0')
        
    plt.tight_layout()
    plt.savefig(chart_filepath, dpi=160, facecolor='#0f172a')
    plt.close("all")
    
    logger(f"🚀 Analysis complete for {clean_ticker}!")
    
    result = {
        "job_id": job_id,
        "ticker": clean_ticker,
        "timeframe": timeframe,
        "model": f"Kronos-{model_name}",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_price": round(last_close, 2),
        "chart_url": f"/static/charts/{chart_filename}",
        "band": band_df.to_dict(orient="records"),
        "brackets": brackets,
        "timemachine": timemachine
    }
    return result
