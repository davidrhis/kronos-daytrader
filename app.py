import os
import sys
import json
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(CURRENT_DIR, "static")
CHARTS_DIR = os.path.join(STATIC_DIR, "charts")
HISTORY_FILE = os.path.join(CURRENT_DIR, "history", "history.json")
ASSETS_FILE = os.path.join(CURRENT_DIR, "static", "assets.json")

os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

from engine import run_full_analysis

app = FastAPI(title="Kronos AI + DayTrader Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def load_history():
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_history_item(item):
    history = load_history()
    # Remove existing entry for same job or ticker to keep clean
    history = [h for h in history if h.get("job_id") != item.get("job_id")]
    history.insert(0, item)
    history = history[:60]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Kronos Dashboard initializing...</h1>"

@app.get("/api/assets")
async def get_assets():
    if os.path.exists(ASSETS_FILE):
        with open(ASSETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@app.get("/api/history")
async def get_history():
    return load_history()

@app.get("/api/history/{job_id}")
async def get_history_detail(job_id: str):
    history = load_history()
    for item in history:
        if item.get("job_id") == job_id:
            return item
    raise HTTPException(status_code=404, detail="Forecast not found")

class ForecastRequest(BaseModel):
    ticker: str
    timeframe: str = "daily"
    model: str = "small"
    capital: float = 100000.0
    risk_pct: float = 1.0
    paths: int = 10

@app.post("/api/forecast")
async def api_forecast(req: ForecastRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            run_full_analysis,
            req.ticker.strip().upper(),
            req.timeframe,
            req.model,
            req.capital,
            req.risk_pct,
            req.paths,
            CHARTS_DIR
        )
        save_history_item(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/forecast/stream")
async def api_forecast_stream(
    ticker: str = Query(...),
    timeframe: str = Query("daily"),
    model: str = Query("small"),
    capital: float = Query(100000.0),
    risk_pct: float = Query(1.0),
    paths: int = Query(10)
):
    clean_ticker = ticker.strip().upper()
    
    async def event_generator():
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def push_event(data_obj):
            loop.call_soon_threadsafe(queue.put_nowait, data_obj)

        def log_cb(msg):
            print(f"[{clean_ticker}] {msg}")
            push_event({"type": "log", "message": str(msg)})

        async def worker():
            try:
                log_cb(f"Starting pipeline for {clean_ticker}...")
                result = await loop.run_in_executor(
                    None,
                    run_full_analysis,
                    clean_ticker,
                    timeframe,
                    model,
                    capital,
                    risk_pct,
                    paths,
                    CHARTS_DIR,
                    log_cb
                )
                save_history_item(result)
                push_event({"type": "complete", "data": result})
            except Exception as e:
                push_event({"type": "error", "message": str(e)})
            finally:
                push_event(None)

        task = asyncio.create_task(worker())

        yield f"data: {json.dumps({'type': 'connected', 'ticker': clean_ticker})}\n\n"

        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=2.0)
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
            except asyncio.TimeoutError:
                if task.done():
                    break
                yield ": keep-alive\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    }
    return StreamingResponse(event_generator(), headers=headers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, log_level="info")
