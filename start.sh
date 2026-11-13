#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

echo "=========================================================="
echo "Starting Kronos AI + DayTrader Intelligence Dashboard..."
echo "Local URL: http://$HOST:$PORT"
echo "=========================================================="

# Check if cloudflared is installed to optionally expose publicly
if command -v cloudflared &> /dev/null; then
    echo "cloudflared detected. Starting background tunnel..."
    nohup cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > cloudflared.log 2>&1 &
    sleep 3
    TUNNEL_URL=$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' cloudflared.log | head -n 1 || true)
    if [ -n "$TUNNEL_URL" ]; then
        echo "Public URL: $TUNNEL_URL"
    fi
fi

exec uvicorn app:app --host "$HOST" --port "$PORT"
