from pathlib import Path
import matplotlib.pyplot as plt

# (method, inference memory in GB, mIoU on SemanticKITTI test set)
BASELINES = [
    ("L2COcc-D",    5.01, 18.18),
    ("L2COcc-C",    5.01, 17.03),
    ("CGFormer",    6.55, 16.63),
    ("MonoOcc-L",   12.0, 15.63),
    ("StereoScene", 7.33, 15.36),
    ("Symphonies",  6.36, 15.04),
    ("MonoOcc-S",   8.19, 13.80),
]
VOOM = ("VOOM", 0.38, 8.4)

LABEL_OFFSETS = {
    "L2COcc-D":    ( 6,  6, "left",  "center"),
    "L2COcc-C":    (-6,  0, "right", "center"),
    "CGFormer":    ( 6,  6, "left",  "bottom"),
    "MonoOcc-L":   (-6,  6, "right", "bottom"),
    "StereoScene": ( 6, -6, "left",  "top"),
    "Symphonies":  (-6, -6, "right", "top"),
    "MonoOcc-S":   ( 6,  0, "left",  "center"),
    "VOOM":        ( 8,  6, "left",  "bottom"),
}


def main():
    out_dir = Path(__file__).resolve().parent.parent / "assets"
    out_dir.mkdir(exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    bx = [m for _, m, _ in BASELINES]
    by = [u for _, _, u in BASELINES]
    ax.scatter(bx, by, s=80, color="#888", edgecolor="white", linewidth=1.5, zorder=3)

    vx, vy = VOOM[1], VOOM[2]
    ax.scatter([vx], [vy], s=200, color="#d62728", marker="*",
               edgecolor="white", linewidth=1.5, zorder=4)

    all_pts = BASELINES + [VOOM]

    for name, mem, miou in all_pts:
        dx, dy, ha, va = LABEL_OFFSETS.get(name, (6, 6, "left", "bottom"))
        ax.annotate(
            name, xy=(mem, miou), xytext=(dx, dy),
            textcoords="offset points",
            ha=ha, va=va,
            fontsize=10,
            fontweight="bold" if name == "VOOM" else "normal",
            color="#d62728" if name == "VOOM" else "#222",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference memory (GB)")
    ax.set_ylabel("mIoU on SemanticKITTI")
    ax.set_xlim(0.2, 20)
    ax.set_ylim(6, 20)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    for ext in ("svg", "png"):
        out = out_dir / f"mem.{ext}"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
