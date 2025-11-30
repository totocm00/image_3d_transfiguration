# ------------------------------
# Load config
# ------------------------------
CONFIG_FILE="config/venv_config.yaml"

VENV_NAME=$(grep 'name:' $CONFIG_FILE | awk '{print $2}')
PY_VERSION=$(grep 'python_version:' $CONFIG_FILE | awk '{print $2}' | tr -d '"')
REQ_FILE=$(grep 'requirements:' $CONFIG_FILE | awk '{print $2}' | tr -d '"')

echo "[INFO] Using venv name: $VENV_NAME"
echo "[INFO] Python version : $PY_VERSION"
echo "[INFO] Requirements   : $REQ_FILE"
echo ""

# ------------------------------
# Python version check
# ------------------------------
if ! command -v python$PY_VERSION &> /dev/null
then
    echo "[ERROR] python$PY_VERSION 이(가) 설치되어 있지 않습니다."
    exit 1
fi

# ------------------------------
# Create VENV
# ------------------------------
echo "[INFO] Creating virtual environment..."
python$PY_VERSION -m venv "$VENV_NAME"

# ------------------------------
# Activate VENV
# ------------------------------
echo "[INFO] Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# ------------------------------
# Install requirements
# ------------------------------
echo "[INFO] Installing requirements..."
pip install --upgrade pip
pip install -r "$REQ_FILE"

echo ""
echo "[SUCCESS] Virtual environment setup complete!"
echo "[INFO] Now active: $VENV_NAME"