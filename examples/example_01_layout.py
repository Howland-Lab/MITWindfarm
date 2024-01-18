from pathlib import Path

import matplotlib.pyplot as plt

from mitwindfarm.Layout import Layout

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    xs = [0, 1, 1, 0]
    ys = [1, 1, 0, 0]

    layout = Layout(xs, ys)
    layout2 = layout.rotate(20)

    plt.plot(layout.x, layout.y, "2k", ms=10, label="unrotated")
    plt.plot(layout2.x, layout2.y, "2r", ms=10, label="rotated")

    for i, (x, y, z) in layout2.iter_downstream():
        plt.text(x, y + 0.02, f"{i+1}")

    plt.legend()

    plt.savefig(FIGDIR / "example_01_layout.png", dpi=300, bbox_inches="tight")
