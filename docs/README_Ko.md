
# 🧙 image_3d_transfiguration  
**2D 이미지를 3D 포인트클라우드로 변환하는 경량 파이프라인**

image_3d_transfiguration은 한 장의 2D 이미지를 입력받아  
**Depth 추정 → 깊이 정규화 → 3D PointCloud → Mesh 생성**까지 한 번에 처리하는  
초경량 2D→3D 변환 모듈입니다.

- 복잡한 kaolin/pytorch3d 설치 없음  
- 단일 이미지로 간단히 3D 형태 추출  
- 출력 파일은 모두 `assets/outputs/` 아래에서 자동 관리  
- OVF(open_vision_factory) 백엔드로 쉽게 이식 가능  

---

# 📦 1. 설치 및 환경 세팅 (중요)

image_3d_transfiguration은 **개발자용(dev) / 배포용(prod) 가상환경을 분리**해서 관리합니다.

- **Linux / macOS**: `scripts/setup_venv.sh` 사용 권장  
- **Windows**: WSL 권장, 또는 PowerShell에서 수동으로 venv 생성

---

## 1-1. 리포 클론

```bash
git clone https://github.com/yourname/image_3d_transfiguration.git
cd image_3d_transfiguration
```

---

## 1-2. Python 버전 확인

```bash
python3 --version
# 예: Python 3.10.x
```

> 권장: **Python 3.10**  
> 최소: 3.8 ~ 3.11 범위

---

## 1-3. 설치 전 준비해야 할 항목 (Prerequisites)

image_3d_transfiguration을 실행하기 위해 필요한 기본 준비물입니다.

### ✔ 필수 준비물

- **Python 3.10**  
  (가상환경 생성에 사용되는 Python 버전은 `config/venv_config.yaml`에서 변경할 수 있습니다.)

- **pip 최신 버전**
  ```bash
  pip install --upgrade pip
  ```

- **requirements_full.txt / requirements_prod.txt**  
  → 선택한 프로필(dev/prod)에 따라 설치되는 의존성 목록입니다.

- **Git 설치**
  - Linux/macOS: 대부분 기본 제공  
  - Windows: https://git-scm.com/download/win

### ✔ GPU 환경 사용 시 (선택)

- **NVIDIA GPU + CUDA Toolkit**  
  - CUDA 11.8 / 12.1 / 12.4 중 하나  
  - 이후 1-6 단계에서 CUDA 버전에 맞는 PyTorch wheel 재설치 권장

### ✔ SAM / YOLO 전처리 사용 시 (선택)

- **SAM / YOLO 가중치 파일**  
  - 1-7 단계의 `download_weights.py`로 자동 다운로드 가능  
  - 저장 위치:
    - `assets/models/sam/`  
    - `assets/models/yolo/`

---

## 1-4. Linux / macOS에서 가상환경 생성 (권장)  
(가상환경 생성에 사용되는 Python 버전은 `config/venv_config.yaml`에서 변경할 수 있습니다.)

- **dev**: 개발용 환경 (FULL 패키지 설치)  
- **prod**: 배포용 환경 (최소 패키지 설치)

### ✅ 개발자용(dev) 환경

```bash
bash scripts/setup_venv.sh dev
```

동작:

- `python3.10 -m venv tester`
- `source tester/bin/activate`
- `pip install -r requirements_full.txt`

설정 완료 후 프롬프트 예시:

```bash
(tester) toto@:~/parent/image_3d_transfiguration$
```

### ✅ 배포용(prod) 환경

```bash
bash scripts/setup_venv.sh prod
```

동작:

- `python3.10 -m venv prod`
- `source prod/bin/activate`
- `pip install -r requirements_prod.txt`

프롬프트 예시:

```bash
(prod) user@host:~/image_3d_transfiguration$
```

### ✅ 기본 프로필로 실행

`venv_profile: prod` 이므로, 아무 인자 없이 실행하면 prod가 사용됩니다.

```bash
bash scripts/setup_venv.sh
```

---

## 1-5. Windows 환경에서 실행하기

### 🔹 방법 1: WSL (권장)

- WSL(Ubuntu) 설치 후  
  → Linux와 동일하게 아래 명령 사용:

```bash
bash scripts/setup_venv.sh dev   # 개발자용
bash scripts/setup_venv.sh prod  # 배포용
```

### 🔹 방법 2: PowerShell에서 수동으로 생성

```powershell
# 리포 위치로 이동
cd C:\path\to\image_3d_transfiguration

# venv 생성 (개발용 예시)
py -3.10 -m venv tester

# venv 활성화
.\tester\Scripts\activate

# 패키지 설치 (FULL)
pip install --upgrade pip
pip install -r requirements_full.txt
```

배포용(prod) 환경은:

```powershell
py -3.10 -m venv prod
.\prod\Scripts\activate

pip install --upgrade pip
pip install -r requirements_prod.txt
```

---

## 1-6. PyTorch / CUDA 설치

requirements 파일에는 기본 torch가 포함되어 있지만,  
**GPU CUDA 버전에 맞는 wheel을 재설치하는 것을 권장**합니다.

### 1) 현재 PyTorch / CUDA 버전 확인

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

예시 출력:

```text
2.5.1+cu124
12.4
```

- `2.5.1+cu124` → PyTorch 2.5.1 + CUDA 12.4 빌드  
- `12.4` → CUDA 12.4 환경

### 2) CUDA 버전에 맞는 PyTorch 설치

본인 CUDA 버전에 맞는 명령어를 선택해 실행하세요:

```bash
# CUDA 12.4
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# NVIDIA GPU가 없는 경우 (CPU-only)
pip install --upgrade torch torchvision torchaudio

# macOS (M1/M2 포함) 또는 CPU 전용
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 1-7. SAM / YOLO 가중치 다운로드

SAM / YOLO 기반 전처리를 사용하려면,  
`download_weights.py`로 필요한 가중치를 자동 다운로드할 수 있습니다.

```bash
python download_weights.py
```

기본적으로 다음 경로에 저장됩니다:

- `assets/models/sam/sam_vit_h_4b8939.pth` (SAM ViT-H, 약 2.4GB)  
- `assets/models/yolo/yolov8n.pt` (YOLOv8n, 약 6MB)

---

# ▶️ 2. 2D → 3D 변환 실행 흐름

일반적인 워크플로우는 아래 순서입니다:

1. 입력 이미지 복사  
2. (선택) YOLO+SAM 기반 전처리  
3. Depth 기반 3D 재구성  
4. PointCloud / Mesh 시각화

---

## 2-1. 입력 이미지 준비

```text
assets/images/
   └─ input.png   # 또는 input.jpg 등
```

원래 예시였던 `robot.png` 대신,  
이제는 **임의의 파일명(input.png 등)을 그대로 사용**하는 방식으로 일반화되었습니다.

---

## 2-2. YOLO + SAM 전처리 실행 (선택)

전처리 실행 스크립트: `run_preprocess.py`  
→ 내부에서 `src/image_3d_transfiguration/preprocess/yolo_sam_pipeline.py`를 호출합니다.

```bash
python run_preprocess.py \
  --input assets/images/input.png \
  --output assets/images/input_yolo_sam.png \
  --output_masked assets/images/input_yolo_sam_masked.png \
  --output_box_vis assets/images/input_yolo_sam_boxes.png \
  --mode single \
  --sam_checkpoint assets/models/sam/sam_vit_h_4b8939.pth
```

전처리 후 생성되는 예시 파일:

```text
assets/images/
  ├─ input.png
  ├─ input_yolo_sam.png
  ├─ input_yolo_sam_masked.png
  └─ input_yolo_sam_boxes.png
```

> `--mode single` : 단일 객체 중심 크롭  
> 추후 multi 모드 등을 확장 가능

---

## 2-3. 3D 재구성 실행

실행 스크립트: `run_3d.py`  
→ 내부에서 `src/image_3d_transfiguration/pipeline.py` + `config_loader.py`를 호출합니다.

전처리 결과(`input_yolo_sam.png`)를 사용하려면:

```bash
python run_3d.py \
  --config config/config.yaml \
  --image_name input_yolo_sam.png
```

원본 이미지만으로 테스트하려면:

```bash
python run_3d.py \
  --config config/config.yaml \
  --image_name input.png
```

성공 시 출력 예시:

```text
[INFO] depth:       assets/outputs/depth/input_yolo_sam_depth.png
[INFO] pointcloud:  assets/outputs/pointcloud/input_yolo_sam_pc.ply
[INFO] mesh:        assets/outputs/mesh/input_yolo_sam_mesh.ply
```

---

# 📁 3. 변환 결과 저장 위치

모든 출력은 자동으로 아래에 정리됩니다:

### ✔ Depth PNG  

```text
assets/outputs/depth/input_yolo_sam_depth.png
```

### ✔ 3D PointCloud (.ply)  

```text
assets/outputs/pointcloud/input_yolo_sam_pc.ply
```

### ✔ 3D Mesh (.ply)  

```text
assets/outputs/mesh/input_yolo_sam_mesh.ply
```

`input_yolo_sam` 부분은 입력 파일명에 따라 자동으로 결정됩니다.

---

# 🧪 4. PointCloud / Mesh 시각화 (Open3D 뷰어)

## 4-1. 수동 파일 지정 뷰어: `view_ply.py`

```bash
python view_ply.py assets/outputs/pointcloud/input_yolo_sam_pc.ply
python view_ply.py assets/outputs/mesh/input_yolo_sam_mesh.ply
```

`view_ply.py`는:

- Mesh인지 PointCloud인지 자동 판별  
- `open3d.visualization.draw_geometries(...)`로 바로 뷰어 띄움

---

## 4-2. 최신 결과 자동 뷰어: `auto_view_ply.py`

가장 최근에 생성된 `.ply` 파일을 자동으로 찾아 띄웁니다.

```bash
python auto_view_ply.py
```

자동 탐색 대상:

- `assets/outputs/pointcloud/*.ply`
- `assets/outputs/mesh/*.ply`

가장 최근 수정된 파일 1개를 선택하여 시각화합니다.

원한다면 수동 지정도 가능합니다:

```bash
python auto_view_ply.py assets/outputs/mesh/input_yolo_sam_mesh.ply
```

---

# 🗂 5. 폴더 구조 (요약)

```text
image_3d_transfiguration/
 ├─ assets/
 │   ├─ images/                 # 입력 이미지
 │   ├─ models/                 # SAM / YOLO 가중치 (download_weights.py로 관리)
 │   │    ├─ sam/
 │   │    └─ yolo/
 │   └─ outputs/                # 변환 결과
 │        ├─ depth/             # depth PNG
 │        ├─ pointcloud/        # point cloud (.ply)
 │        └─ mesh/              # mesh (.ply)
 ├─ config/
 │   ├─ config.yaml             # 3D 파이프라인 설정
 │   └─ venv_config.yaml        # dev/prod 가상환경 설정
 ├─ scripts/
 │   ├─ run_3d.py               # (구) 실행용 CLI 스크립트 (옵션)
 │   └─ setup_venv.sh           # dev/prod venv 자동 생성 스크립트
 ├─ src/
 │   └─ image_3d_transfiguration/
 │        ├─ pipeline.py        # 핵심 2D→3D 파이프라인
 │        ├─ config_loader.py   # YAML 설정 로더
 │        └─ preprocess/        # 전처리 모듈(YOLO, SAM, 수동 크롭 등)
 │             ├─ yolo_sam_pipeline.py
 │             ├─ yolo_crop.py
 │             ├─ manual_crop.py
 │             └─ __init__.py
 ├─ download_weights.py         # SAM / YOLO 가중치 자동 다운로드
 ├─ run_preprocess.py           # YOLO + SAM 전처리 실행 진입점
 ├─ run_3d.py                   # 3D 재구성 실행 진입점
 ├─ view_ply.py                 # .ply 수동 시각화
 ├─ auto_view_ply.py            # 최신 .ply 자동 시각화
 ├─ requirements_full.txt       # 개발용 (FULL) 의존성
 └─ requirements_prod.txt       # 배포용 (PROD) 최소 의존성
```

---

# ⚙️ 6. config.yaml 설정 설명

`config/config.yaml`을 통해 결과 저장 옵션 및 모델 설정을 변경할 수 있습니다.

```yaml
paths:
  input_image_dir: "assets/images"
  output_root: "assets/outputs"
  depth_dir: "depth"
  pointcloud_dir: "pointcloud"
  mesh_dir: "mesh"

output:
  save_depth_png: true
  depth_grayscale: true
  save_pointcloud: true
  save_mesh: true
  point_step: 2
  clip_min: 0.05
  clip_max: 0.95

model:
  id: "LiheYoung/depth-anything-small-hf"
  device: "auto"   # auto / cpu / cuda
```

### ✔ 주요 항목

- **depth_grayscale**  
  깊이를 0~255 그레이스케일로 저장할지 (true/false)

- **point_step**  
  포인트 샘플링 간격 (1 = 매우 촘촘, 2~4 = 적당)

- **clip_min / clip_max**  
  노이즈 제거를 위한 depth 값 제한

- **device**  
  `"auto"`: GPU 있으면 CUDA 자동 사용, 없으면 CPU

---

# 🪄 7. 활용 목적

- 이미지 한 장으로 3D 윤곽을 빠르게 추출  
- 로봇/비전/디지털트윈에서 **시각화용 3D 힌트** 생성  
- OVF(open_vision_factory) 백엔드 플러그인으로 사용 가능  
- 연구/학습용 Depth 기반 3D Reconstruction 템플릿

---

# 🏷️ 8. 출처 및 고지

image_3d_transfiguration은  
**Open Vision Factory(OVF)에서 파생된 실험·연구용 2D→3D 모듈**이며,  
기본 아이디어는 Meta AI의 **SAM-3D Objects** 프로젝트에서 영감을 얻었습니다.

본 리포는 SAM-3D의 개념 중  
“단일 이미지 기반 3D 재구성” 요소만 경량화하여  
Depth Anything 기반으로 재구성한 버전입니다.

원천 프로젝트:  
https://github.com/facebookresearch/sam-3d-objects