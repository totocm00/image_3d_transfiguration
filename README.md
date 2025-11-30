
# 🧙 image_3d_transfiguration  
**A lightweight pipeline that converts 2D images into 3D point clouds and meshes**

`image_3d_transfiguration` takes a single 2D image as input and performs  
**Depth estimation → depth normalization → 3D PointCloud generation → Mesh creation**  
in a single streamlined pipeline.

- No heavy kaolin / pytorch3d dependencies  
- Quickly extract 3D geometry from a single image  
- All outputs are automatically stored under `assets/outputs/`  
- Easy to integrate as a backend module of OVF (open_vision_factory)  

---

# 📦 1. Installation & Environment Setup (Important)

`image_3d_transfiguration` uses separate virtual environments for  
**development (dev)** and **deployment (prod)**.

- **Linux / macOS**: Recommended to use `scripts/setup_venv.sh`  
- **Windows**: WSL is recommended, or create venv manually via PowerShell

---

## 1-1. Clone the repository

```bash
git clone https://github.com/yourname/image_3d_transfiguration.git
cd image_3d_transfiguration
```

---

## 1-2. Check Python version

```bash
python3 --version
# Example: Python 3.10.x
```

> Recommended: **Python 3.10**  
> Supported range: 3.8 ~ 3.11

---

## 1-3. Prerequisites (Before installing)

These are the basic requirements to run `image_3d_transfiguration`.

### ✔ Required

- **Python 3.10**  
  (The Python version used for creating the virtual environment can be changed in `config/venv_config.yaml`.)

- **Latest pip**
  ```bash
  pip install --upgrade pip
  ```

- **requirements_full.txt / requirements_prod.txt**  
  → The actual dependencies will be installed based on the selected profile (`dev` or `prod`).

- **Git**
  - Linux / macOS: Usually pre-installed  
  - Windows: https://git-scm.com/download/win

### ✔ Optional: GPU environment

- **NVIDIA GPU + CUDA Toolkit**  
  - One of CUDA 11.8 / 12.1 / 12.4  
  - Later in section 1-6, you will install a matching PyTorch wheel for your CUDA version

### ✔ Optional: SAM / YOLO preprocessing

- **SAM / YOLO weight files**  
  - Can be downloaded automatically via `download_weights.py` (see 1-7)  
  - Stored under:
    - `assets/models/sam/`  
    - `assets/models/yolo/`

---

## 1-4. Creating virtual environments on Linux / macOS (Recommended)  
(The Python version used for creating the venv can be changed in `config/venv_config.yaml`.)

- **dev**: Development environment (installs the FULL set of packages)  
- **prod**: Production environment (installs a minimal set of packages)

### ✅ Development (dev) environment

```bash
bash scripts/setup_venv.sh dev
```

This will:

- Run `python3.10 -m venv tester`
- Run `source tester/bin/activate`
- Run `pip install -r requirements_full.txt`

Example shell prompt after activation:

```bash
(tester) toto@:~/parent/image_3d_transfiguration$
```

### ✅ Production (prod) environment

```bash
bash scripts/setup_venv.sh prod
```

This will:

- Run `python3.10 -m venv prod`
- Run `source prod/bin/activate`
- Run `pip install -r requirements_prod.txt`

Example prompt:

```bash
(prod) user@host:~/image_3d_transfiguration$
```

### ✅ Using the default profile

Since `venv_profile: prod` is set by default, running the script with no arguments uses the prod profile:

```bash
bash scripts/setup_venv.sh
```

---

## 1-5. Running on Windows

### 🔹 Option 1: WSL (Recommended)

Install WSL (Ubuntu), then use the same commands as on Linux:

```bash
bash scripts/setup_venv.sh dev   # development
bash scripts/setup_venv.sh prod  # production
```

### 🔹 Option 2: Manual venv creation in PowerShell

```powershell
# Move to the repo directory
cd C:\path\to\image_3d_transfiguration

# Create a venv (development example, using Python 3.10)
py -3.10 -m venv tester

# Activate the venv
.\tester\Scripts\activate

# Install dependencies (FULL)
pip install --upgrade pip
pip install -r requirements_full.txt
```

For the production (prod) environment:

```powershell
py -3.10 -m venv prod
.\prod\Scripts\activate

pip install --upgrade pip
pip install -r requirements_prod.txt
```

---

## 1-6. PyTorch / CUDA installation

The requirements files include a default `torch` entry, but  
**if you have a GPU, it is strongly recommended to reinstall PyTorch using the wheel that matches your CUDA version.**

### 1) Check current PyTorch / CUDA versions

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

Example output:

```text
2.5.1+cu124
12.4
```

- `2.5.1+cu124` → PyTorch 2.5.1 built with CUDA 12.4  
- `12.4` → CUDA 12.4 runtime

### 2) Install PyTorch for your CUDA version

Pick the command that matches your environment:

```bash
# CUDA 12.4
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# NVIDIA GPU not available (CPU-only)
pip install --upgrade torch torchvision torchaudio

# macOS (M1/M2 included) or CPU-only build
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 1-7. Downloading SAM / YOLO weights

If you want to use SAM / YOLO-based preprocessing,  
you can automatically download the required weights using `download_weights.py`:

```bash
python download_weights.py
```

By default, the files will be stored as:

- `assets/models/sam/sam_vit_h_4b8939.pth` (SAM ViT-H, ~2.4GB)  
- `assets/models/yolo/yolov8n.pt` (YOLOv8n, ~6MB)

---

# ▶️ 2. 2D → 3D processing pipeline

The typical workflow is:

1. Copy input image  
2. (Optional) Run YOLO + SAM preprocessing  
3. Run depth-based 3D reconstruction  
4. Visualize PointCloud / Mesh

---

## 2-1. Prepare an input image

```text
assets/images/
   └─ input.png   # or input.jpg, etc.
```

Originally, the example used `robot.png`,  
but now the pipeline is generalized to use arbitrary filenames like `input.png`.

---

## 2-2. Run YOLO + SAM preprocessing (optional)

Preprocessing entry script: `run_preprocess.py`  
→ Internally calls `src/image_3d_transfiguration/preprocess/yolo_sam_pipeline.py`.

```bash
python run_preprocess.py \
  --input assets/images/input.png \
  --output assets/images/input_yolo_sam.png \
  --output_masked assets/images/input_yolo_sam_masked.png \
  --output_box_vis assets/images/input_yolo_sam_boxes.png \
  --mode single \
  --sam_checkpoint assets/models/sam/sam_vit_h_4b8939.pth
```

Example output files:

```text
assets/images/
  ├─ input.png
  ├─ input_yolo_sam.png
  ├─ input_yolo_sam_masked.png
  └─ input_yolo_sam_boxes.png
```

> `--mode single` : Focus on a single main object (single-object crop)  
> Multi-object modes can be added later.

---

## 2-3. Run 3D reconstruction

Execution script: `run_3d.py`  
→ Internally uses `src/image_3d_transfiguration/pipeline.py` and `config_loader.py`.

To use the preprocessed image (`input_yolo_sam.png`):

```bash
python run_3d.py \
  --config config/config.yaml \
  --image_name input_yolo_sam.png
```

To test with the original image only:

```bash
python run_3d.py \
  --config config/config.yaml \
  --image_name input.png
```

Example log output:

```text
[INFO] depth:       assets/outputs/depth/input_yolo_sam_depth.png
[INFO] pointcloud:  assets/outputs/pointcloud/input_yolo_sam_pc.ply
[INFO] mesh:        assets/outputs/mesh/input_yolo_sam_mesh.ply
```

---

# 📁 3. Output file locations

All outputs are stored in the following directories:

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

The `input_yolo_sam` part will be replaced automatically based on your input filename.

---

# 🧪 4. PointCloud / Mesh visualization (Open3D viewer)

## 4-1. Manual viewer: `view_ply.py`

```bash
python view_ply.py assets/outputs/pointcloud/input_yolo_sam_pc.ply
python view_ply.py assets/outputs/mesh/input_yolo_sam_mesh.ply
```

`view_ply.py`:

- Automatically checks whether the file is a mesh or point cloud  
- Opens an Open3D window using `open3d.visualization.draw_geometries(...)`

---

## 4-2. Auto viewer for the latest result: `auto_view_ply.py`

This script automatically finds and visualizes the most recently created `.ply` file.

```bash
python auto_view_ply.py
```

Search targets:

- `assets/outputs/pointcloud/*.ply`  
- `assets/outputs/mesh/*.ply`

It selects the most recently modified file and visualizes it.

You can also specify a file explicitly:

```bash
python auto_view_ply.py assets/outputs/mesh/input_yolo_sam_mesh.ply
```

---

# 🗂 5. Folder layout (summary)

```text
image_3d_transfiguration/
 ├─ assets/
 │   ├─ images/                 # Input images
 │   ├─ models/                 # SAM / YOLO weights (managed via download_weights.py)
 │   │    ├─ sam/
 │   │    └─ yolo/
 │   └─ outputs/                # All outputs
 │        ├─ depth/             # Depth PNGs
 │        ├─ pointcloud/        # Point clouds (.ply)
 │        └─ mesh/              # Meshes (.ply)
 ├─ config/
 │   ├─ config.yaml             # 3D pipeline configuration
 │   └─ venv_config.yaml        # dev/prod virtual environment configuration
 ├─ scripts/
 │   ├─ run_3d.py               # (Legacy) CLI runner (optional)
 │   └─ setup_venv.sh           # dev/prod venv automation script
 ├─ src/
 │   └─ image_3d_transfiguration/
 │        ├─ pipeline.py        # Core 2D → 3D pipeline
 │        ├─ config_loader.py   # YAML config loader
 │        └─ preprocess/        # Preprocessing modules (YOLO, SAM, manual crop, etc.)
 │             ├─ yolo_sam_pipeline.py
 │             ├─ yolo_crop.py
 │             ├─ manual_crop.py
 │             └─ __init__.py
 ├─ download_weights.py         # SAM / YOLO weight auto-downloader
 ├─ run_preprocess.py           # Entry point for YOLO + SAM preprocessing
 ├─ run_3d.py                   # Entry point for 3D reconstruction
 ├─ view_ply.py                 # Manual .ply viewer
 ├─ auto_view_ply.py            # Auto viewer for the latest .ply
 ├─ requirements_full.txt       # Development (FULL) dependencies
 └─ requirements_prod.txt       # Production (PROD) minimal dependencies
```

---

# ⚙️ 6. `config.yaml` configuration

You can control the output paths and model settings in `config/config.yaml`:

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

### ✔ Key fields

- **depth_grayscale**  
  Whether to save depth as 0–255 grayscale (true/false)

- **point_step**  
  Sampling step for point clouds (1 = very dense, 2–4 = moderate)

- **clip_min / clip_max**  
  Depth clipping range to suppress noise

- **device**  
  `"auto"`: use CUDA if available, otherwise fallback to CPU

---

# 🪄 7. Intended use-cases

- Quickly derive approximate 3D geometry from a single image  
- Generate **3D visualization hints** for robotics / vision / digital twin pipelines  
- Plug-and-play backend module for OVF (open_vision_factory)  
- Template for research / education on depth-based 3D reconstruction

---

# 🏷️ 8. Credits & Notice

`image_3d_transfiguration` is an  
**experimental / research 2D→3D module derived from Open Vision Factory (OVF)**,  
inspired by Meta AI’s **SAM-3D Objects** project.

This repository focuses on a lightweight implementation of  
“single-image 3D reconstruction” based on Depth Anything.

Original project:  
https://github.com/facebookresearch/sam-3d-objects