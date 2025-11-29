# 🪄 image_3d_transfiguration  
**A lightweight pipeline that transforms a single 2D image into a 3D point cloud**

`image_3d_transfiguration` takes one image as input and performs:  
**Depth Estimation → Depth Normalization → 3D PointCloud Generation**  
in a clean, lightweight pipeline.

- No kaolin or pytorch3d required  
- Extracts a coarse 3D structure from one image  
- All generated files are automatically managed inside `assets/outputs/`  
- Easily integrable as a backend module for OVF (open_vision_factory)

---

# 📦 1. Installation & Environment Setup (Important)

Because Python versions differ per user,  
it is recommended to **manually create a virtual environment**.

---

## 1) Clone the repository

```bash
git clone https://github.com/yourname/image_3d_transfiguration.git
cd image_3d_transfiguration
```

---

## 2) Check your Python version

```bash
python3 --version
```

Python 3.8–3.11 recommended.

---

## 3) Create a virtual environment

```bash
python3 -m venv robot3d_env
```

or with a specific Python version:

```bash
python3.10 -m venv tester_env
```

---

## 4) Activate the virtual environment

Linux / macOS:

```bash
source robot3d_env/bin/activate
```

Windows CMD:

```cmd
robot3d_env\Scripts\activate
```

---

# 🔧 5. Install PyTorch (Check your CUDA version first)

The Depth Anything model requires PyTorch.  
You **must** install PyTorch that matches your system's CUDA version.

---

## ✔ Check your current PyTorch / CUDA version

```bash
python3 -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

Example output:

```
2.5.1+cu124
12.4
```

Meaning:
- `2.5.1+cu124` → PyTorch 2.5.1 compiled with CUDA 12.4  
- `12.4` → Your current CUDA version

---

## ✔ Install the correct PyTorch build for your CUDA version

Choose the correct command:

### ✔ CUDA 12.4
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### ✔ CUDA 12.1
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### ✔ CUDA 11.8
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### ✔ CPU-only (no NVIDIA GPU)
```bash
pip install torch torchvision
```

### ✔ macOS (M1/M2 included) or install failed
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

⚠️ **PyTorch is NOT included in requirements.txt.**  
Every user must install PyTorch according to their own CUDA environment.

---

# 📦 6. Install project dependencies

After PyTorch is successfully installed:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

- accelerate  
- transformers  
- huggingface-hub  
- open3d  
- numpy  
- pillow  
- pyyaml  

---

# ▶️ 2. Run the 2D → 3D Transformation

Place the image you want to convert here:

```
assets/images/
    └─ robot.png
```

Run:

```bash
python scripts/run_3d.py --image_name robot.png
```

Example output:

```
=== Image 3D Transfiguration Result ===
depth:       assets/outputs/depth/robot_depth.png
point cloud: assets/outputs/pointcloud/robot_pc.ply
```

---

# 📁 3. Output File Locations

All results are automatically saved under:

### ✔ Depth PNG  
```
assets/outputs/depth/robot_depth.png
```

### ✔ 3D PointCloud (.ply)  
```
assets/outputs/pointcloud/robot_pc.ply
```

---

# 🧪 4. Visualize the PointCloud (Open3D)

```bash
python -c "import open3d as o3d; p=o3d.io.read_point_cloud('assets/outputs/pointcloud/robot_pc.ply'); o3d.visualization.draw_geometries([p])"
```

This will launch the Open3D interactive viewer  
where you can rotate, zoom, and explore the 3D point cloud.

---

# 🗂 5. Folder Structure

```
image_3d_transfiguration/
 ├─ assets/
 │   ├─ images/             # Input images
 │   └─ outputs/            # Output root
 │        ├─ depth/         # Depth PNG files
 │        └─ pointcloud/    # PointCloud (.ply)
 ├─ config/
 │   └─ config.yaml         # Model & output configuration
 ├─ scripts/
 │   └─ run_3d.py           # CLI execution script
 └─ src/image_3d_transfiguration/
      ├─ pipeline.py        # Core depth → 3D logic
      └─ config_loader.py   # YAML config loader
```

---

# ⚙️ 6. config.yaml Settings

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

### ✔ Explanation of key fields
- **depth_grayscale** → Saves depth as 0–255 grayscale  
- **point_step** → Sampling interval for point cloud density  
- **clip_min / clip_max** → Removes extreme depth noise  
- **device** → `"auto"` uses CUDA automatically if available  

---

# 🧙 7. Use Cases

- Quick 3D shape approximation from a single image  
- Robotics, machine vision, and digital-twin visualization  
- Backend plugin for OVF (open_vision_factory)  
- Educational and research-friendly 3D reconstruction pipeline

---

# 🏷️ 8. Attribution & Notice

`image_3d_transfiguration` is  
an experimental 2D→3D reconstruction module **derived from Open Vision Factory (OVF)**.

The core idea is inspired by  
Meta AI’s **SAM-3D Objects** project.

This repository re-implements the *single-image 3D reconstruction* concept  
in a lightweight manner using Depth Anything,  
avoiding heavy dependencies such as kaolin or pytorch3d.

Original source project:  
https://github.com/facebookresearch/sam-3d-objects