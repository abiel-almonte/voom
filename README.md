`voom` **- VOlumetric Occupancy from Monocular video. Real-time on a Jetson.**

<p align="center">
  <img src="assets/demo.gif" alt="VOOM demo" width="800"/>
</p>

Predict dense semantic 3D occupancy from a single RGB image.

```bash
git clone https://github.com/abielalmonte/voom && cd voom
uv sync
./scripts/download_weights.sh
```

```python
from voom import VOOMv2, load_state_dict

model = VOOMv2().cuda().half().eval()
model.load_state_dict(load_state_dict("release/voom.safetensors", "cuda"))
grid = model(rgb, K)  # [1, 24, 128, 32, 128]  occ + rgb + 20 sem channels
```

**Accuracy** (SemanticKITTI seq 08):

| Metric | Value |
|---|---|
| Occupancy IoU | 0.316 |
| Semantic mIoU | 0.084 |

**Speed** (Jetson Orin Nano 8GB, TensorRT FP16):

| Metric | Value |
|---|---|
| Latency | 64.39 ms |
| FPS | 15.5 |
| VRAM | 361 MB |

Full per-class IoU + per-stage timings: [blog post / HF model card].

Train your own:
```bash
./scripts/generate_da_depth.sh ~/path/to/skitti/dataset
python train.py  # config.yaml drives everything
```

Deploy on Jetson:
```bash
python deploy/build_trt.py  # build TRT engines
python deploy/run.py        # GL viewer + side-by-side mp4 recording
```

---

Built on [DINOv2](https://github.com/facebookresearch/dinov2) and a [Lift-Splat-Shoot](https://arxiv.org/abs/2008.05711) style implementation. Trained on [SemanticKITTI](http://semantic-kitti.org/) with [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) for photometric & depth supervision.

**voom**, dense 3D from one camera under 25W.
