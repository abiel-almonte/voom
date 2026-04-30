`voom` **- VOlumetric Occupancy from Monocular video. Real-time on a Jetson.**



**15 FPS** on a Jetson Orin Nano, drawing **400 MB** and **16 W**. **31.6 IoU** and **8.4 mIoU** on SemanticKITTI seq 08.

---

Predict dense semantic 3D occupancy from a single RGB image.

```python
from voom import VOOMv2, load_state_dict

model = VOOMv2().cuda().half().eval()
model.load_state_dict(load_state_dict("release/voom.safetensors", "cuda"))
grid = model(rgb, K)  # [1, 24, 128, 32, 128]  occ + rgb + 20 sem channels
```

Download weights:

```bash
./scripts/download_weights.sh
```

Or train your own:

```bash
./scripts/generate_da_depth.sh ~/path/to/skitti/dataset
python train.py  # config.yaml drives everything
```

Deploy on Jetson:

```bash
python deploy/build_trt.py # build TRT engines
python deploy/run.py # GL viewer + side-by-side mp4 recording
```

---

**voom**, occupancy at the edge.

> Built on [DINOv2](https://github.com/facebookresearch/dinov2) backbone and a [Lift-Splat-Shoot](https://arxiv.org/abs/2008.05711) style head. Trained on [SemanticKITTI](http://semantic-kitti.org/) with [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) for photometric and depth supervision.

