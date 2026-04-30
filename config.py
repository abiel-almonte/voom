import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

common = config["common"]
model = config["model"]
train = config["training"]
deploy = config["deployment"]
viz = config["viz"]

_root = Path(__file__).parent
weights_path = (_root / Path(deploy["weights"]).expanduser()).resolve()
engine_dir = (_root / Path(deploy["engine_dir"]).expanduser()).resolve()
prelift_onnx = engine_dir / "prelift.onnx"
prelift_trt = engine_dir / "prelift.trt"
postlift_onnx = engine_dir / "postlift.onnx"
postlift_trt = engine_dir / "postlift.trt"
trtexec = deploy["trtexec"]


class GridDim:
    x = common["grid_dims"]["x_voxels"]
    y = common["grid_dims"]["y_voxels"]
    z = common["grid_dims"]["z_voxels"]
    n = x * y * z

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z


grid_dim = GridDim()

mpv = common["meters_per_voxel"]

# offset, grid center, and max depth
_y_off_voxels = (grid_dim.y - 1) - (common["cam_height_m"] / mpv + 2)
offset_m = ((grid_dim.x / 2) * mpv, _y_off_voxels * mpv, 0.0)
grid_center_m = (0.0, (grid_dim.y / 2) * mpv - offset_m[1], (grid_dim.z / 2) * mpv)
max_depth_m = grid_dim.z * mpv

colors = [color for color in viz["colors"].values()]  # list for order & dupes
panel_size_wh = (viz["panel_width"], viz["panel_height"])

model_config = {
    "backbone_size": model["dinov2_size"],
    "sampled_layers": [2, 5, 8, 11],
    "out_ch": model["out_ch"],
    "embed_dim": model["lss_embed_dim"],
    "refine_dim": model["lss_refine_dim"],
    "depth_bins": model["depth_bins"],
    "grid_dim": grid_dim,
    "offset_m": offset_m,
    "mpv": mpv,
}

trt_config = {
    "backbone_size": model["dinov2_size"],
    "sampled_layers": [2, 5, 8, 11],
    "depth_bins": model["depth_bins"],
    "grid_dim": grid_dim,
    "offset_m": offset_m,
    "mpv": mpv,
}

dataset_config = {
    "root": Path(train["data"]["root_dir"]).expanduser(),
    "seqs": train["data"]["train_seqs"],
    "resize": (train["height"], train["width"]),
    "load_sem": (
        train["loss_weights"]["sem"] > 0 and len(train["semantic"]["enabled"]) > 0
    ),
    "load_depth": (
        train["loss_weights"]["photo"] > 0 or train["loss_weights"]["depth"] > 0
    ),
}

val_dataset_config = {
    "root": Path(train["data"]["root_dir"]).expanduser(),
    "seqs": train["data"]["valid_seqs"],
    "resize": (train["height"], train["width"]),
    "load_sem": True,
    "load_depth": False,
}

cam = deploy["camera"]
camera_config = {
    "device": cam["device"],
    "capture_size": tuple(cam["capture_size"]),
    "fps": cam["fps"],
    "target_hfov_deg": cam["target_hfov_deg"],
    "calib_size": tuple(cam["calib_size"]),
    "fx": cam["fx"],
    "fy": cam["fy"],
    "cx": cam["cx"],
    "cy": cam["cy"],
    "distortion": cam["distortion"],
}
