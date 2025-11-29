#!/usr/bin/env bash
set -e

VENV_NAME="tester_env"

echo "🪄 가상환경 생성 중..."
python3 -m venv "${VENV_NAME}"

echo "🪄 가상환경 활성화..."
# 현재 쉘에서 활성화됨
source "${VENV_NAME}/bin/activate"

echo "🪄 pip 업그레이드..."
pip install --upgrade pip

echo "🪄 requirements 설치..."
pip install -r requirements.txt

echo
echo "=============================================="
echo "✨ Setup 완료! 가상환경이 활성화되었습니다."
echo "현재 쉘에서 바로 실행할 수 있습니다:"
echo
echo "  python scripts/run_3d.py --image_name robot.png"
echo
echo "가상환경 비활성화: deactivate"
echo "=============================================="