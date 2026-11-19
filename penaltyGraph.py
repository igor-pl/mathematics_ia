import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("penalties.csv")
print(len(df))
print(df["outcome"].value_counts())

print(df.columns)

fig = plt.figure(figsize=(4, 2), facecolor='lightskyblue',
                 layout='constrained')
fig.suptitle('A nice Matplotlib Figure')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HALF_WIDTH = 3.66   # half the goal width (m)
HEIGHT = 2.44       # crossbar height (m)
POST = 0.12         # thickness of posts and crossbar (m)


def draw_goal(ax):
    # Sky / background
    ax.set_facecolor("#dfeef7")

    # Grass, with light and dark stripes
    stripe = 1.0
    x = -6
    dark = True
    while x < 6:
        ax.add_patch(Rectangle((x, -0.6), stripe, 0.6,
                               color="#4c9a2a" if dark else "#5cae35", zorder=0))
        x += stripe
        dark = not dark

    # Net (thin grey grid inside the goal)
    for nx in [i * 0.25 - HALF_WIDTH for i in range(int(2 * HALF_WIDTH / 0.25) + 1)]:
        ax.plot([nx, nx], [0, HEIGHT], color="grey", linewidth=0.4, alpha=0.5, zorder=1)
    for ny in [i * 0.25 for i in range(int(HEIGHT / 0.25) + 1)]:
        ax.plot([-HALF_WIDTH, HALF_WIDTH], [ny, ny], color="grey", linewidth=0.4, alpha=0.5, zorder=1)

    # Posts and crossbar (white with a black outline).
    # The inside edges of the frame are at x = ±3.66 and y = 2.44.
    frame = dict(facecolor="white", edgecolor="black", linewidth=1, zorder=3)
    ax.add_patch(Rectangle((-HALF_WIDTH - POST, 0), POST, HEIGHT + POST, **frame))   # left post
    ax.add_patch(Rectangle((HALF_WIDTH, 0), POST, HEIGHT + POST, **frame))            # right post
    ax.add_patch(Rectangle((-HALF_WIDTH - POST, HEIGHT), 2 * HALF_WIDTH + 2 * POST, POST, **frame))  # crossbar

    # Goal line
    ax.plot([-6, 6], [0, 0], color="white", linewidth=2, zorder=2)

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-0.6, 4)
    ax.set_aspect("equal")


fig, ax = plt.subplots(figsize=(11, 5))
draw_goal(ax)

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import cm, colors

# Grid: 6 columns, 3 rows
x_edges = np.linspace(-3.66, 3.66, 7)
y_edges = np.linspace(0, 2.44, 4)

# Only shots inside the goal that were a goal or saved
inside = df[(df["x"].abs() <= 3.66) & (df["y"].between(0, 2.44))
            & (df["result"].isin(["Goal", "Saved"]))].copy()

inside["col"] = pd.cut(inside["x"], x_edges, labels=False, include_lowest=True)
inside["row"] = pd.cut(inside["y"], y_edges, labels=False, include_lowest=True)

# Colour scale: 0% = red, 100% = green
cmap = cm.get_cmap("RdYlGn")
norm = colors.Normalize(vmin=0, vmax=1)

for row in range(3):
    for col in range(6):
        square = inside[(inside["row"] == row) & (inside["col"] == col)]
        x0, y0 = x_edges[col], y_edges[row]
        w, h = x_edges[col + 1] - x0, y_edges[row + 1] - y0

        if len(square) == 0:
            # no data: grey square
            ax.add_patch(Rectangle((x0, y0), w, h, facecolor="lightgrey",
                                   alpha=0.6, zorder=2))
            ax.text(x0 + w / 2, y0 + h / 2, "no data", ha="center",
                    va="center", fontsize=8, zorder=5)
            continue

        rate = (square["result"] == "Goal").mean()
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=cmap(norm(rate)),
                               edgecolor="black", linewidth=0.5, alpha=0.75, zorder=2))
        ax.text(x0 + w / 2, y0 + h / 2, f"{rate:.0%}\n(n={len(square)})",
                ha="center", va="center", fontsize=9, fontweight="bold", zorder=5)

# Colour bar on the side
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Scoring rate (on-target penalties)")
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
# Your penalties go here, for example:
# ax.scatter(df["x"], df["height"], c="green", s=25, zorder=4)
colours = {"Goal": "green", "Saved": "blue", "Missed": "red"}

for result, colour in colours.items():
    part = df[df["result"] == result]          # all penalties with this result
    ax.scatter(part["x"], part["y"],       # whole columns at once
               c=colour, s=25, alpha=0.8,
               edgecolors="black", linewidths=0.3,
               label=f"{result} ({len(part)})", zorder=4)

ax.legend(loc="upper right")

ax.set_xlabel("Horizontal position from goal centre (m)")
ax.set_ylabel("Height (m)")
plt.tight_layout()
plt.show()

