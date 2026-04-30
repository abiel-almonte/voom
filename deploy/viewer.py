"""OpenGL viewer for live VOOMv2 voxel inference + side-by-side mp4 recording."""

import os

os.environ["MODERNGL_WINDOW"] = "glfw"

import atexit
import subprocess
import time

import cv2
import numpy as np
import moderngl
import moderngl_window as mglw

import config

CUBE_VERTS = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype="f4",
)

CUBE_INDICES = np.array(
    [
        0,
        1,
        2,
        0,
        2,
        3,
        4,
        5,
        6,
        4,
        6,
        7,
        0,
        3,
        7,
        0,
        7,
        4,
        1,
        2,
        6,
        1,
        6,
        5,
        0,
        1,
        5,
        0,
        5,
        4,
        3,
        2,
        6,
        3,
        6,
        7,
    ],
    dtype="i4",
)

GRID_CENTER_M = np.array(config.grid_center_m)
COLOR_MAP = np.array(config.colors) / 255.0
BRIGHTNESS_SCALE = np.linspace(3.0, 0.4, config.grid_dim.y, dtype=np.float32)


def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, center, up):
    eye = np.asarray(eye, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    f = center - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)

    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


VERTEX_SHADER = """
#version 330
in vec3 in_pos;
in vec3 in_inst_pos;
in vec3 in_inst_color;
uniform mat4 mvp;
out vec3 v_color;

void main() {
    vec3 pos = in_pos * 0.2 + in_inst_pos;
    gl_Position = mvp * vec4(pos, 1.0);
    v_color = in_inst_color;
}
"""

FRAGMENT_SHADER = """
#version 330
in vec3 v_color;
out vec4 fragColor;

void main() {
    fragColor = vec4(v_color, 1.0);
}
"""


class VoomViewer(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "voom"
    window_size = (1280, 800)
    fullscreen = True
    resource_dir = "."

    gen = None  # model inference fn — set externally before run_window_config

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        assert self.gen is not None, "Set VoomGUI.gen before run_window_config"

        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )
        self.vbo = self.ctx.buffer(CUBE_VERTS.tobytes())
        self.ibo = self.ctx.buffer(CUBE_INDICES.tobytes())

        max_bytes = config.grid_dim.n * 3 * 4
        self.inst_center = self.ctx.buffer(reserve=max_bytes)
        self.inst_color = self.ctx.buffer(reserve=max_bytes)

        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo, "3f", "in_pos"),
                (self.inst_center, "3f/i", "in_inst_pos"),
                (self.inst_color, "3f/i", "in_inst_color"),
            ],
            self.ibo,
        )

        pw, ph = config.panel_size_wh
        date = time.strftime("%m-%d-%Y_%H-%M-%S")

        os.makedirs("videos", exist_ok=True)
        self.ffmpeg_proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{pw * 2}x{ph}",
                "-r",
                "15",
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-loglevel",
                "error",
                f"videos/{date}.mp4",
            ],
            stdin=subprocess.PIPE,
        )
        atexit.register(self._close_ffmpeg)

    def _close_ffmpeg(self):
        try:
            self.ffmpeg_proc.stdin.close()
        except Exception:
            pass
        try:
            self.ffmpeg_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.ffmpeg_proc.kill()

    def write_to_video(self, inp, out):
        inp = cv2.resize(
            np.asarray(inp), config.panel_size_wh, interpolation=cv2.INTER_AREA
        )
        out = cv2.resize(out, config.panel_size_wh, interpolation=cv2.INTER_AREA)
        view = cv2.cvtColor(np.hstack([inp, out]), cv2.COLOR_RGB2BGR)
        self.ffmpeg_proc.stdin.write(view.tobytes())

    def on_render(self, time: float, frame_time: float) -> None:
        if frame_time > 0.0:
            self.wnd.title = f"voom - {1/frame_time:.1f} FPS"

        self.ctx.clear(0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        occ, sem, inp_pil = next(self.gen)
        labels = sem[occ].cpu().numpy().clip(0, 19)

        idx = occ.nonzero().cpu().numpy().astype("f4")
        centers = (idx * config.mpv) - config.offset_m
        colors = (
            (COLOR_MAP[labels] * BRIGHTNESS_SCALE[idx[:, 1].astype(np.int32), None])
            .clip(0, 1)
            .astype("f4")
        )

        self.inst_center.write(centers.tobytes())
        self.inst_color.write(colors.tobytes())

        proj = perspective(90.0, self.wnd.aspect_ratio, 0.1, 100.0)
        eye = GRID_CENTER_M + np.array([0, -9, -9], dtype=np.float32)
        view = look_at(eye, GRID_CENTER_M, [0, 0, 1])
        mvp = proj @ view

        self.prog["mvp"].write(mvp.T.astype("f4").tobytes())
        self.vao.render(instances=len(centers))

        w, h = self.wnd.buffer_size
        buf = self.wnd.fbo.read(viewport=(0, 0, w, h), components=3, alignment=1)
        render = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)[::-1]

        self.write_to_video(inp_pil, render)
