# 🪄 image_3d_transfiguration  
**A lightweight pipeline for converting 2D images into 3D point clouds**

`image_3d_transfiguration` takes a single 2D image and performs:  
**Depth Estimation → Normalization → 3D Point Cloud Generation**  
all in one simple and efficient pipeline.

- No kaolin / pytorch3d required  
- Extracts a coarse 3D structure from a single image  
- All outputs are automatically managed under `assets/outputs/`  
- Easy to integrate into OVF (open_vision_factory) as a backend module  

---

# 📦 1. Installation & Environment Setup

Since Python versions differ between users,  
creating a virtual environment manually is recommended.

### 1) Clone the repository
```bash
git clone https://github.com/yourname/image_3d_transfiguration.git
cd image_3d_transfiguration
```

### 2) Check your Python version
```bash
python3 --version
```
Python 3.8–3.11 recommended.

### 3) Create a virtual environment
```bash
python3 -m venv robot3d_env
```

### 4) Activate the virtual environment  
Linux/macOS:
```bash
source robot3d_env/bin/activate
```

Windows:
```cmd
robot3d_env\Scripts\activate
```

### 5) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# ▶️ 2. Run the 2D → 3D Transformation

Place your input image here:

```
assets/images/
    └─ robot.png
```

Run the transformation:

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

# 📁 3. Output Files

All generated files are stored automatically under `assets/outputs/`.

### ✔ Depth PNG  
```
assets/outputs/depth/robot_depth.png
```

### ✔ 3D Point Cloud (.ply)  
```
assets/outputs/pointcloud/robot_pc.ply
```

---

# 🧪 4. Visualize the Point Cloud (Open3D)

```bash
python -c "import open3d as o3d; p=o3d.io.read_point_cloud('assets/outputs/pointcloud/robot_pc.ply'); o3d.visualization.draw_geometries([p])"
```

This opens an interactive 3D viewer  
where you can rotate and inspect the point cloud.

---

# 🗂 5. Project Structure

```
image_3d_transfiguration/
 ├─ assets/
 │   ├─ images/             # Input images
 │   └─ outputs/            # Generated output files
 │        ├─ depth/         # Depth visualization
 │        └─ pointcloud/    # Generated point clouds (.ply)
 ├─ config/
 │   └─ config.yaml         # Output/model configuration
 ├─ scripts/
 │   └─ run_3d.py           # CLI entry point
 └─ src/image_3d_transfiguration/
      ├─ pipeline.py        # Core depth → 3D logic
      └─ config_loader.py   # YAML config loader
```

---

# ⚙️ 6. Configuration (config.yaml)

You can adjust output settings and model behavior via:

`config/config.yaml`

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

### ✔ Explanation
- **depth_grayscale** → whether to save depth as 0–255 grayscale  
- **point_step** → sampling interval for point cloud density  
- **clip_min / clip_max** → removes extreme/noisy depth values  
- **device** → `"auto"` uses CUDA automatically if available  

---

# 🧙 7. Use Cases

- Rapid prototyping of rough 3D shapes from a single image  
- Robotics / digital twin pipelines that need visual hints  
- OVF(open_vision_factory) backend module  
- Educational & research use for depth-based 3D reconstruction

---

# 🏷️ 8. Attribution & Source Notice

`image_3d_transfiguration` is  
an experimental 2D→3D reconstruction module  
**derived from the Open Vision Factory (OVF)**.

The core idea is inspired by Meta AI’s  
**SAM-3D Objects** project.

This repository re-implements the *single-image 3D reconstruction* concept  
in a lightweight manner using Depth Anything,  
avoiding heavy dependencies such as kaolin or pytorch3d.

Original project:  
https://github.com/facebookresearch/sam-3d-objects