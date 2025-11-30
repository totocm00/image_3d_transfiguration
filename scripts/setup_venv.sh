# ------------------------------
# Config
# ------------------------------
CONFIG_FILE="config/venv_config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "[ERROR] config 파일을 찾을 수 없습니다: $CONFIG_FILE"
  exit 1
fi

# ------------------------------
# Profile 결정 (인자 > YAML > 기본 dev)
# ------------------------------
PROFILE="$1"

if [ -z "$PROFILE" ]; then
  PROFILE=$(grep '^venv_profile:' "$CONFIG_FILE" | awk '{print $2}')
fi

if [ -z "$PROFILE" ]; then
  PROFILE="dev"
fi

case "$PROFILE" in
  dev|prod)
    ;;
  *)
    echo "[WARN] 알 수 없는 프로필: $PROFILE → dev로 변경"
    PROFILE="dev"
    ;;
esac

# ------------------------------
# YAML에서 값 읽기 (dev/prod 프로필별)
# ------------------------------
VENV_NAME=$(grep "venv_name_${PROFILE}:" "$CONFIG_FILE" | awk '{print $2}')
PY_CMD=$(grep "venv_python_${PROFILE}:" "$CONFIG_FILE" | awk '{print $2}')
REQ_FILE=$(grep "venv_requirements_${PROFILE}:" "$CONFIG_FILE" | awk '{print $2}')

echo "[INFO] Profile       : $PROFILE"
echo "[INFO] Using venv    : $VENV_NAME"
echo "[INFO] Python command: $PY_CMD"
echo "[INFO] Requirements  : $REQ_FILE"
echo ""

if [ -z "$VENV_NAME" ] || [ -z "$PY_CMD" ] || [ -z "$REQ_FILE" ]; then
  echo "[ERROR] venv_config.yaml 에서 값을 제대로 읽지 못했습니다."
  exit 1
fi

# ------------------------------
# Python 존재 여부 체크
# ------------------------------
if ! command -v "$PY_CMD" >/dev/null 2>&1; then
  echo "[ERROR] ${PY_CMD} 이(가) 설치되어 있지 않습니다."
  exit 1
fi

echo "[INFO] Python OK: $($PY_CMD --version)"
echo ""

# ------------------------------
# 기존 VENV 확인
# ------------------------------
if [ -d "$VENV_NAME" ]; then
  echo "[WARN] 이미 존재하는 가상환경입니다: $VENV_NAME"
  echo "[INFO] 그대로 사용합니다."
else
  echo "[INFO] Creating virtual environment..."
  "$PY_CMD" -m venv "$VENV_NAME"
fi

# ------------------------------
# VENV 활성화
# ------------------------------
echo "[INFO] Activating virtual environment..."
# shellcheck source=/dev/null
source "$VENV_NAME/bin/activate"

# ------------------------------
# 필요한 패키지 설치
# ------------------------------
echo "[INFO] Installing requirements..."
pip install --upgrade pip
pip install -r "$REQ_FILE"

echo ""
echo "[SUCCESS] Virtual environment setup complete!"
echo "[INFO] Now active: $VENV_NAME"