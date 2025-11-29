import os
import yaml
from dataclasses import dataclass


@dataclass
class PathsConfig:
    input_image_dir: str
    output_root: str
    depth_dir: str
    pointcloud_dir: str


@dataclass
class OutputConfig:
    save_depth_png: bool
    depth_grayscale: bool
    save_pointcloud: bool
    point_step: int
    clip_min: float
    clip_max: float


@dataclass
class ModelConfig:
    id: str
    device: str  # "auto" / "cpu" / "cuda"


@dataclass
class AppConfig:
    paths: PathsConfig
    output: OutputConfig
    model: ModelConfig


def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    output = cfg.get("output", {})
    model = cfg.get("model", {})

    paths_cfg = PathsConfig(
        input_image_dir=paths.get("input_image_dir", "assets/images"),
        output_root=paths.get("output_root", "assets/outputs"),
        depth_dir=paths.get("depth_dir", "depth"),
        pointcloud_dir=paths.get("pointcloud_dir", "pointcloud"),
    )

    output_cfg = OutputConfig(
        save_depth_png=output.get("save_depth_png", True),
        depth_grayscale=output.get("depth_grayscale", True),
        save_pointcloud=output.get("save_pointcloud", True),
        point_step=int(output.get("point_step", 2)),
        clip_min=float(output.get("clip_min", 0.05)),
        clip_max=float(output.get("clip_max", 0.95)),
    )

    model_cfg = ModelConfig(
        id=model.get("id", "LiheYoung/depth-anything-small-hf"),
        device=model.get("device", "auto"),
    )

    return AppConfig(paths=paths_cfg, output=output_cfg, model=model_cfg)


def build_output_paths(cfg: AppConfig, image_basename: str):
    """
    image_basename: 예) robot.png → robot
    """
    out_root = cfg.paths.output_root
    depth_root = os.path.join(out_root, cfg.paths.depth_dir)
    pc_root = os.path.join(out_root, cfg.paths.pointcloud_dir)

    os.makedirs(depth_root, exist_ok=True)
    os.makedirs(pc_root, exist_ok=True)

    depth_path = os.path.join(depth_root, f"{image_basename}_depth.png")
    pc_path = os.path.join(pc_root, f"{image_basename}_pc.ply")

    return depth_path, pc_path