# 🪄 image_3d_transfiguration  
**2D 이미지를 3D 포인트클라우드로 변환하는 경량 파이프라인**

image_3d_transfiguration은 한 장의 2D 이미지를 입력받아  
**Depth 추정 → 깊이 정규화 → 3D PointCloud 생성**까지 한 번에 처리하는  
초경량 2D→3D 변환 모듈입니다.

- 복잡한 kaolin/pytorch3d 설치 없음  
- 단일 이미지로 간단히 3D 형태 추출  
- 출력 파일은 모두 `assets/outputs/` 아래에서 자동 관리  
- OVF(open_vision_factory) 백엔드로 쉽게 이식 가능  

---

# 📦 1. 설치 및 환경 세팅 (중요)

image_3d_transfiguration은 **사용자의 Python 버전에 따라 직접 venv 생성**하는 방식을 권장합니다.

### 1) 리포 클론
```bash
git clone https://github.com/yourname/image_3d_transfiguration.git
cd image_3d_transfiguration
```

### 2) Python 버전 확인
```bash
python3 --version
```
Python 3.8 ~ 3.11 권장.

### 3) 가상환경 생성
```bash
python3 -m venv robot3d_env

ex_ $ python3.10 -m venv tester_env
```

### 4) 가상환경 활성화
Linux / macOS:
```bash
source robot3d_env/bin/activate
```

Windows:
```cmd
robot3d_env\Scripts\activate
```



프로젝트 요구사항을 설치하기 전에,  
**본인 PC의 CUDA 버전에 맞는 PyTorch를 먼저 설치해야 합니다.**

### 5) 현재 PyTorch / CUDA 버전 확인
아래 명령을 실행합니다:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

예시 출력:

```
2.5.1+cu124
12.4
```

의미:
- `2.5.1+cu124` → PyTorch 2.5.1 + CUDA 12.4 빌드
- `12.4` → CUDA 12.4 환경

### 6) CUDA 버전에 맞는 PyTorch 설치

본인 CUDA 버전에 맞는 명령어를 선택해 실행하세요:

#### ✔ CUDA 12.4
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### ✔ CUDA 12.1
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### ✔ CUDA 11.8
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### ✔ NVIDIA GPU가 없는 경우 (CPU-only)
```bash
pip install torch torchvision
```

#### ✔ macOS (M1/M2 포함) 설치 오류 발생 시
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```


### 7) 패키지 설치
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

설치되는 패키지 목록:
- accelerate  
- transformers  
- huggingface-hub  
- open3d  
- numpy  
- pillow  
- pyyaml  

⚠️ **torch는 requirements.txt 안에 포함되지 않습니다.**  
CUDA 환경이 사용자마다 다르기 때문에 **각자 자신의 CUDA 버전에 맞게** 별도로 설치해야 합니다.


---

# ▶️ 2. 2D → 3D 변환 실행

먼저 변환할 이미지를 아래 경로에 넣습니다:

```
assets/images/
   └─ robot.png
```

실행:

```bash
python scripts/run_3d.py --image_name robot.png
```

성공 시 출력:

```
=== Image 3D Transfiguration 결과 ===
depth:       assets/outputs/depth/robot_depth.png
point cloud: assets/outputs/pointcloud/robot_pc.ply
```

---

# 📁 3. 변환 결과 저장 위치

모든 출력은 자동으로 아래에 정리됩니다:

### ✔ Depth PNG  
```
assets/outputs/depth/robot_depth.png
```

### ✔ 3D PointCloud (.ply)  
```
assets/outputs/pointcloud/robot_pc.ply
```

---

# 🧪 4. PointCloud 시각화 (Open3D)

```bash
python -c "import open3d as o3d; p=o3d.io.read_point_cloud('assets/outputs/pointcloud/robot_pc.ply'); o3d.visualization.draw_geometries([p])"
```

Open3D 뷰어가 열리고 3D 점 구름을 회전/확대하며 볼 수 있습니다.

---

# 🗂 5. 폴더 구조

```
image_3d_transfiguration/
 ├─ assets/
 │   ├─ images/             # 입력 이미지 저장 위치
 │   └─ outputs/            # 변환 결과 저장 루트
 │        ├─ depth/         # depth PNG 저장
 │        └─ pointcloud/    # point cloud 저장
 ├─ config/
 │   └─ config.yaml         # 출력/모델 설정
 ├─ scripts/
 │   └─ run_3d.py           # 실행용 CLI 스크립트
 └─ src/image_3d_transfiguration/
      ├─ pipeline.py        # 핵심 변환 로직
      └─ config_loader.py   # YAML 설정 로더
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

output:
  save_depth_png: true
  depth_grayscale: true
  save_pointcloud: true
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
  `"auto"`: GPU 있으면 CUDA 자동 사용

---

# 🧙 7. 활용 목적

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