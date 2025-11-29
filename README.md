# 🪄 image_3d_transfiguration  
**2D 이미지를 3D 포인트클라우드로 변환하는 경량 파이프라인**

image_3d_transfiguration은 한 장의 2D 이미지를 입력받아  
**Depth 추정 → 정규화 → 3D PointCloud 생성**까지 한 번에 처리하는  
초경량 2D→3D 변환 모듈입니다.

- 복잡한 kaolin/pytorch3d 설치 없음  
- 단일 이미지만으로 간단히 3D 형태 추출  
- 출력 파일들은 모두 `assets/outputs/` 아래에 자동 정리  
- OVF(open_vision_factory) 프로젝트 백엔드로 쉽게 이식 가능

---

## 📦 1. 설치 및 환경 세팅

리포를 클론한 뒤, 제공된 venv 세팅 스크립트를 실행하면 됩니다.

### 1) 리포 클론
```bash
git clone https://github.com/yourname/image_3d_transfiguration.git
cd image_3d_transfiguration
```

### 2) 가상환경 생성 + 패키지 설치
```bash
bash setup_venv.sh
source robot3d_env/bin/activate
```

### 3) 설치되는 주요 패키지
- torch (이미 시스템에 설치된 버전 사용)
- transformers (Depth Anything 로딩용)
- accelerate
- open3d
- pillow / numpy

---

## 🗂 2. 폴더 구조

```
image_3d_transfiguration/
 ├─ assets/
 │   ├─ images/            # 입력 이미지 저장 위치
 │   └─ outputs/           # 변환 결과 저장 루트
 │        ├─ depth/        # depth PNG 저장
 │        └─ pointcloud/   # point cloud (PLY) 저장
 ├─ config/
 │   └─ config.yaml        # 출력 설정, 모델 설정
 ├─ scripts/
 │   └─ run_3d.py          # 실행 스크립트(CLI)
 └─ src/image_3d_transfiguration/
      ├─ pipeline.py       # 핵심 Depth→3D 변환 로직
      └─ config_loader.py  # YAML config 로더
```

---

## ⚙️ 3. config.yaml에서 설정 가능한 항목

`config/config.yaml` 파일을 열어 변경할 수 있습니다.

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

### ✔ 주요 설정 설명
- **depth_grayscale**: `true`면 깊이를 0~255 그레이스케일 PNG로 저장  
- **point_step**: 포인트 샘플링 간격. 1이면 가장 촘촘  
- **clip_min/max**: 너무 앞/뒤에 있는 이상한 depth 값 제거  
- **device**: `"auto"` 추천 (GPU 있으면 cuda 자동 사용)

---

## 🖼 4. 변환할 이미지 넣기

아래 경로에 이미지를 넣습니다:

```
assets/images/
   └─ robot.png
```

이미지 이름은 무엇이든 상관없습니다.

---

## ▶️ 5. 실행 방법 (2D → 3D 변환)

```bash
python scripts/run_3d.py --image_name robot.png
```

실행되면 콘솔에 다음처럼 출력됩니다:

```
=== Image 3D Transfiguration 결과 ===
depth:       assets/outputs/depth/robot_depth.png
point cloud: assets/outputs/pointcloud/robot_pc.ply
```

---

## 📁 6. 변환 결과 저장 위치

모든 결과는 `assets/outputs/` 아래에 자동 생성됩니다.

### ✔ Depth 이미지 (PNG)
```
assets/outputs/depth/robot_depth.png
```

### ✔ 3D PointCloud (PLY 파일)
```
assets/outputs/pointcloud/robot_pc.ply
```

---

## 🧪 7. PointCloud 열어보기 (Open3D)

```bash
python -c "import open3d as o3d; p=o3d.io.read_point_cloud('assets/outputs/pointcloud/robot_pc.ply'); o3d.visualization.draw_geometries([p])"
```

위 명령을 실행하면 **상호작용 가능한 3D 뷰어**가 뜹니다.

---

## 🧙 8. 목적과 활용

- 사진 한 장으로 3D 구조의 **대략적인 윤곽**을 얻을 때  
- 로봇/공정/디지털트윈에서 **시각적 표시용 3D 힌트** 필요할 때  
- OVF(open_vision_factory) 백엔드 확장 모듈로 사용  
- 학습/연구용 2D→3D 변환 파이프라인으로 활용

---

이 리포는 **압도적으로 가벼운 구성**으로  
“이미지 → 3D 포인트클라우드” 흐름을 빠르게 시도해볼 수 있게 설계돼 있습니다.

---

## 🏷️ 9. 출처 및 고지

이 프로젝트 **image_3d_transfiguration**은  
Open Vision Factory(OVF)에서 파생된 실험·연구용 모듈이며,  
2D→3D 변환 아이디어는 Meta AI의 **SAM-3D Objects** 프로젝트에서 영감을 얻었습니다.

본 리포는 SAM-3D의 실험 개념을 참고하되,  
환경 설치 난이도와 의존성 문제를 줄이기 위해  
Depth Anything 기반으로 재구성한 **경량화 구현 버전**입니다.

원천 프로젝트:  
https://github.com/facebookresearch/sam-3d-objects