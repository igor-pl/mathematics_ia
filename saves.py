"""
Part 3: square grid of save rates on the whole goal, then on the mirrored half goal.
Part 4: 6 x 6 grid on the half goal, plus save rate per column (x) and per row (y).
Save rate = saved / (goals + saved), on-target shots only.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Settings ----------
CSV_FILE = 'penalties.csv'
W = 3.66              # half the goal width (m)
H = 2.44              # crossbar height (m)
SQ_ROWS = 4           # Part 3: rows of squares (4 rows -> 12 x 4 squares of 0.61 m)
NU, NY = 6, 6         # Part 4: half-goal grid

GROUPS = {
    "Goal": "Goal",
    "Saved": "Saved", "Saved Off Target": "Saved", "Saved to Post": "Saved",
    "Off T": "Missed", "Post": "Missed", "Wayward": "Missed", "Blocked": "Missed",
}
COLOURS = {"Goal": "green", "Saved": "blue", "Missed": "red"}
CMAP = plt.get_cmap('Reds')   # light = few saves, dark = many saves


# ---------- Load ----------
df = pd.read_csv(CSV_FILE)
df['result'] = df['outcome'].map(GROUPS)
df['u'] = df['x'].abs()                    # mirrored horizontal position

on = df[(df['x'].abs() <= W) & df['y'].between(0, H)
        & df['result'].isin(['Goal', 'Saved'])].copy()
on['saved'] = (on['result'] == 'Saved').astype(int)
p_bar = on['saved'].mean()
print(f"On-target shots: {len(on)}, overall save rate p = {p_bar:.3f}")

left, right = on[on['x'] < 0], on[on['x'] >= 0]
print(f"Left side save rate:  {left['saved'].mean():.1%} (n = {len(left)})")
print(f"Right side save rate: {right['saved'].mean():.1%} (n = {len(right)})")


def grid_rates(data, xcol, x_edges, y_edges):
    """Save rate and count in each cell. Rows go bottom to top."""
    col = pd.cut(data[xcol], x_edges, labels=False, include_lowest=True)
    row = pd.cut(data['y'], y_edges, labels=False, include_lowest=True)
    rate = np.full((len(y_edges) - 1, len(x_edges) - 1), np.nan)
    n = np.zeros_like(rate, dtype=int)
    for (c, r), g in data.groupby([col, row]):
        n[int(r), int(c)] = len(g)
        rate[int(r), int(c)] = g['saved'].mean()
    return rate, n


def draw_grid(ax, rate, n, x_edges, y_edges, points, xcol, title):
    """Coloured cells, all shots as dots, save rate and n written in each cell."""
    for r in range(len(y_edges) - 1):
        for c in range(len(x_edges) - 1):
            x0, y0 = x_edges[c], y_edges[r]
            w, h = x_edges[c + 1] - x0, y_edges[r + 1] - y0
            face = 'white' if n[r, c] == 0 else CMAP(rate[r, c])
            ax.add_patch(plt.Rectangle((x0, y0), w, h, facecolor=face, edgecolor='grey',
                                       linewidth=0.5, zorder=1))
            label = "-" if n[r, c] == 0 else f"{rate[r, c]:.0%}\nn={n[r, c]}"
            ax.text(x0 + w / 2, y0 + h / 2, label, ha='center', va='center', fontsize=7,
                    zorder=4, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
    for res, col in COLOURS.items():
        part = points[points['result'] == res]
        ax.plot(part[xcol], part['y'], '.', color=col, markersize=3, zorder=3, label=res)
    ax.axhline(0, color='green', linewidth=2, zorder=2)
    ax.set_aspect('equal')
    ax.set_ylim(-0.2, H + 0.3)
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(loc='upper center', fontsize=7, bbox_to_anchor=(0.5, -0.2), ncol=3)


def add_colourbar(fig, ax):
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, label="Save rate")


def print_grid(rate, n, x_mid, y_mid, xname):
    header = ("y / " + xname).ljust(10)
    print(header + "".join(f"{m:>12.2f}" for m in x_mid))
    for r in reversed(range(len(y_mid))):
        line = f"{y_mid[r]:<10.2f}"
        for c in range(len(x_mid)):
            line += f"{'-':>12}" if n[r, c] == 0 else f"{rate[r, c]:>6.0%} ({n[r, c]:>3})"
        print(line)


# ===================== PART 3: square grids =====================
size = H / SQ_ROWS                                   # side of each square (m)
y_sq = np.linspace(0, H, SQ_ROWS + 1)

# Whole goal
x_sq = np.linspace(-W, W, int(round(2 * W / size)) + 1)
rate, n = grid_rates(on, 'x', x_sq, y_sq)

for row in range(rate.shape[0]):
    for col in range(rate.shape[1]):
        x_range = f"[{x_sq[col]:.2f}, {x_sq[col+1]:.2f}]"
        y_range = f"[{y_sq[row]:.2f}, {y_sq[row+1]:.2f}]"
        val = f"{rate[row, col]:.1%}" if not np.isnan(rate[row, col]) else "N/A"
        print(f"Box (row {row}, col {col}) | x: {x_range}, y: {y_range} -> Rate: {val} (n={n[row, col]})")

fig, ax = plt.subplots(figsize=(9.5, 3.8))
draw_grid(ax, rate, n, x_sq, y_sq, df, 'x', f"Save rate per square ({size:.2f} m), whole goal")
ax.plot([-W, -W, W, W], [0, H, H, 0], color='black', linewidth=3, zorder=2)
ax.set_xlim(-W - 0.3, W + 0.3)
ax.set_xlabel("x (m)")
add_colourbar(fig, ax)
plt.tight_layout()
plt.savefig("3a_squares_whole_goal.png", dpi=200)


# Half goal, left side mirrored onto the right
u_sq = np.linspace(0, W, int(round(W / size)) + 1)
rate, n = grid_rates(on, 'u', u_sq, y_sq)
fig, ax = plt.subplots(figsize=(6.5, 5))
draw_grid(ax, rate, n, u_sq, y_sq, df, 'u', f"Save rate per square ({size:.2f} m), mirrored half goal")
ax.plot([0, W, W], [H, H, 0], color='black', linewidth=3, zorder=2)
ax.set_xlim(-0.1, W + 0.3)
ax.set_xlabel("u = |x| (m)")
add_colourbar(fig, ax)
plt.tight_layout()
plt.savefig("3b_squares_half_goal.png", dpi=200)


# ===================== PART 4: 6 x 6 half goal + axes =====================
u_edges = np.linspace(0, W, NU + 1)
y_edges = np.linspace(0, H, NY + 1)
u_mid = (u_edges[:-1] + u_edges[1:]) / 2
y_mid = (y_edges[:-1] + y_edges[1:]) / 2

rate, n = grid_rates(on, 'u', u_edges, y_edges)
print(f"\n6 x 6 half goal (save rate and n):")
print_grid(rate, n, u_mid, y_mid, "u")

fig, ax = plt.subplots(figsize=(6.5, 5))
draw_grid(ax, rate, n, u_edges, y_edges, df, 'u', "Save rate per zone, 6 x 6 half goal")
ax.plot([0, W, W], [H, H, 0], color='black', linewidth=3, zorder=2)
ax.set_xlim(-0.1, W + 0.3)
ax.set_xlabel("u = |x| (m)")
add_colourbar(fig, ax)
plt.tight_layout()
plt.savefig("4a_half_goal_6x6.png", dpi=200)

# Save rate per column (all heights) and per row (all widths)
on['col'] = pd.cut(on['u'], u_edges, labels=False, include_lowest=True)
on['row'] = pd.cut(on['y'], y_edges, labels=False, include_lowest=True)
cols = on.groupby('col')['saved'].agg(['mean', 'size']).reindex(range(NU))
rows = on.groupby('row')['saved'].agg(['mean', 'size']).reindex(range(NY))

print("\nHorizontal (per column):")
for i in range(NU):
    print(f"  u = {u_mid[i]:.2f} m   s = {cols['mean'][i]:.3f}   n = {int(cols['size'][i])}")
print("Vertical (per row):")
for i in range(NY):
    print(f"  y = {y_mid[i]:.2f} m   s = {rows['mean'][i]:.3f}   n = {int(rows['size'][i])}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for a, mid, data, lab, xmax in [(ax1, u_mid, cols, "u = |x| (m), middle of column", W),
                                (ax2, y_mid, rows, "y (m), middle of row", H)]:
    a.plot(mid, data['mean'], 'o', color='black')
    for i in range(len(mid)):
        a.annotate(f"n={int(data['size'][i])}", (mid[i], data['mean'][i]),
                   textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8)
    a.axhline(p_bar, color='grey', linestyle='--', linewidth=1, label=f"average {p_bar:.3f}")
    a.set_xlim(0, xmax)
    a.set_ylim(0, max(0.5, np.nanmax(data['mean']) + 0.1))
    a.set_xlabel(lab)
    a.set_ylabel("Save rate")
    a.legend()
ax1.set_title("Horizontal save rate")
ax2.set_title("Vertical save rate")
plt.tight_layout()
plt.savefig("4b_axis_save_rates.png", dpi=200)

plt.show()