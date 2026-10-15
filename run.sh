#!/usr/bin/env bash
# ==============================================================================
# EdgeDepth-Lite: One-Click Launch Script
# Automatically manages the Python virtual environment and starts the web UI.
# ==============================================================================

set -e

# Resolve repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python3"
PIP_BIN="$VENV_DIR/bin/pip"

# 1. Check if virtual environment exists; create if missing
if [ ! -d "$VENV_DIR" ] || [ ! -f "$PYTHON_BIN" ]; then
    echo "⚙️  Virtual environment not found. Creating .venv..."
    python3 -m venv "$VENV_DIR"
    echo "📦 Installing required dependencies from requirements.txt..."
    "$PIP_BIN" install --upgrade pip
    "$PIP_BIN" install -r "$SCRIPT_DIR/requirements.txt"
    echo "✅ Environment setup complete."
fi

# 2. Check if sample assets exist; generate if missing
if [ ! -f "$SCRIPT_DIR/assets/sample_desk.jpg" ]; then
    echo "🖼️  Generating sample scene assets..."
    "$PYTHON_BIN" "$SCRIPT_DIR/scripts/create_sample_assets.py"
fi

# 3. Launch the Gradio web application
echo "🚀 Launching EdgeDepth-Lite Synthetic Aperture Prototype..."
echo "📍 Application URL: http://127.0.0.1:7860"
echo "Press Ctrl+C to stop."
echo ""

exec "$PYTHON_BIN" "$SCRIPT_DIR/src/app.py" "$@"
