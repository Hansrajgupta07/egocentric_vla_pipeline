#!/usr/bin/env bash
# ==============================================================================
# setup.sh — Egocentric VLA Pipeline Environment Setup
# ==============================================================================

set -e

echo "🚀 Setting up Egocentric VLA Pipeline environment..."

# 1. Download MediaPipe Hand Landmarker model if missing
TASK_MODEL="hand_landmarker.task"
if [ ! -f "$TASK_MODEL" ]; then
    echo "📥 Downloading MediaPipe hand landmarker model..."
    curl -L -o "$TASK_MODEL" \
        https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    echo "✅ Downloaded $TASK_MODEL"
else
    echo "✅ $TASK_MODEL already present"
fi

# 2. Check Python installation
PYTHON_BIN=$(which python3)
echo "🐍 Using Python: $PYTHON_BIN ($($PYTHON_BIN --version))"

# 3. Install core dependencies
echo "📦 Installing core dependencies..."
$PYTHON_BIN -m pip install --upgrade pip
$PYTHON_BIN -m pip install -r requirements.txt

# 4. Generate ArUco marker assets if missing
if [ ! -f "images/marker_0_cube.png" ] || [ ! -f "images/marker_1_bowl.png" ]; then
    echo "🏷️ Generating ArUco marker templates..."
    $PYTHON_BIN scripts/generate_markers.py
fi

echo "=============================================================================="
echo "🎉 Setup complete! You can now run:"
echo "   python3 scripts/3_run_libero.py --policy mlp --trials 3"
echo "   python3 agents/orchestrator.py --eval-trials 5"
echo "=============================================================================="
