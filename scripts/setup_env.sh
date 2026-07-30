set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Installing Python requirements..."
pip install -q -r requirements.txt

SAM2_DIR="$PROJECT_ROOT/sam2"
if [ ! -d "$SAM2_DIR" ]; then
    echo "Cloning sam2..."
    git clone -q https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
fi
echo "Installing sam2 (editable)..."
pip install -q -e "$SAM2_DIR"
pip install -q sam2

DAV2_DIR="$PROJECT_ROOT/Depth-Anything-V2"
if [ ! -d "$DAV2_DIR" ]; then
    echo "Cloning Depth-Anything-V2..."
    git clone -q https://github.com/DepthAnything/Depth-Anything-V2 "$DAV2_DIR"
fi
echo "Installing Depth-Anything-V2 requirements..."
pip install -q -r "$DAV2_DIR/metric_depth/requirements.txt"
mkdir -p "$DAV2_DIR/metric_depth/checkpoints"

echo ""
echo "Setup complete."
echo "Next steps:"
echo "  1. Put your food image under: $PROJECT_ROOT/data/"
echo "  2. Put your fine-tuned checkpoint under: $PROJECT_ROOT/models/checkpoints/"
echo "  3. Add/confirm a camera profile in: $PROJECT_ROOT/config/config.py"
echo "  4. Run: python demo.py"
