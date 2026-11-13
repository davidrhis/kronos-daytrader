#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "      Kronos AI + DayTrader Intelligence Installer"
echo "=========================================================="

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3.10+."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python version: $PYTHON_VERSION"

# Set up virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing Python dependencies from requirements.txt..."
if command -v uv &> /dev/null; then
    uv pip install -r requirements.txt
else
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Pre-cache default models
echo "Pre-caching Kronos foundation models (Kronos-small & Tokenizer)..."
python3 -c "
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from model.kronos import KronosTokenizer, Kronos
print('Downloading tokenizer...')
KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base')
print('Downloading Kronos-small...')
Kronos.from_pretrained('NeoQuasar/Kronos-small')
print('Models successfully pre-cached!')
"

echo "=========================================================="
echo "Installation complete!"
echo "Run 'bash start.sh' to launch the dashboard on port 8080."
echo "=========================================================="
