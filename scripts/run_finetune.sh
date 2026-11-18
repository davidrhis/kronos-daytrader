#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "$DIR"

PYTHON_CMD="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
fi

CONFIG="${1:-finetune/config_nifty50_small.yaml}"
echo "=========================================================="
echo "Starting Kronos Domain Fine-Tuning Pipeline"
echo "Config: $CONFIG"
echo "=========================================================="

# 1. Check if dataset exists, if not generate it
if [ ! -f "data/nifty50_daily.csv" ]; then
    echo "Dataset missing. Ingesting historical Indian market candles..."
    $PYTHON_CMD scripts/prepare_finetune_data.py data/
fi

# 2. Run sequential training
echo "Executing Sequential Training (Tokenizer + Predictor)..."
$PYTHON_CMD finetune/train_sequential.py --config "$CONFIG"

echo "=========================================================="
echo "Fine-tuning completed successfully!"
echo "Checkpoints saved in finetuned/"
echo "=========================================================="
