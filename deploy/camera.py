import cv2
import numpy as np
from PIL import Image


class Camera:
    def __init__(
        self,
        device,
        capture_size,
        fps,
        target_hfov_deg,
        calib_size,
        fx,
        fy,
        cx,
        cy,
        distortion,
    ):
        self.W, self.H = capture_size
        self.hfov_deg = float(target_hfov_deg)

        calib_w, calib_h = calib_size
        sx, sy = self.W / calib_w, self.H / calib_h

        K = np.array(
            [
                [fx * sx, 0.0, cx * sx],
                [0.0, fy * sy, cy * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        D = np.array(distortion, dtype=np.float64).reshape(-1, 1)

        new_f = (self.W / 2) / np.tan(np.radians(self.hfov_deg / 2))
        self.K_new = np.array(
            [
                [new_f, 0.0, self.W / 2],
                [0.0, new_f, self.H / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K,
            D,
            np.eye(3),
            self.K_new,
            (self.W, self.H),
            cv2.CV_16SC2,
        )

        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        assert self.cap.isOpened(), f"Failed to open /dev/video{device}"
        for _ in range(5):
            self.cap.read()

    def read(self):
        ok, bgr = self.cap.read()
        if not ok:
            return None
        rect = cv2.remap(bgr, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb), self.K_new

    def release(self):
        self.cap.release()


if __name__ == "__main__":
    import time
    import config

    cam = Camera(**config.camera_config)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        cam.read()
    dt = time.perf_counter() - t0
    print(f"{n} frames in {dt:.2f}s = {n/dt:.1f} FPS")
    cam.release()
