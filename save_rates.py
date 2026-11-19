import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Settings ----------
CSV_FILE = 'penalties.csv'
HALF_WIDTH = 3.66     # half the goal width (m)
HEIGHT = 2.44         # crossbar height (m)
NU, NY = 6, 6         # grid on the half goal: 6 columns x 6 rows

GROUPS = {
    "Goal": "Goal",
    "Saved": "Saved", "Saved Off Target": "Saved", "Saved to Post": "Saved",
    "Off T": "Missed", "Post": "Missed", "Wayward": "Missed", "Blocked": "Missed",
}

# ---------- Load and mirror ----------
df = pd.read_csv(CSV_FILE)
df['result'] = df['outcome'].map(GROUPS)
df['u'] = df['x'].abs()                     # mirror: distance from the centre of the goal

# Symmetry check: save rate on each side before mirroring
on_all = df[(df['x'].abs() <= HALF_WIDTH) & df['y'].between(0, HEIGHT)
            & df['result'].isin(['Goal', 'Saved'])]
left, right = on_all[on_all['x'] < 0], on_all[on_all['x'] >= 0]
print(f"Left side:  save rate {(left['result'] == 'Saved').mean():.1%} (n={len(left)})")
print(f"Right side: save rate {(right['result'] == 'Saved').mean():.1%} (n={len(right)})")

# On-target shots only (goal or saved) are used for save rates
on = on_all.copy()
on['u'] = on['x'].abs()
on['saved'] = (on['result'] == 'Saved').astype(int)
p_bar = on['saved'].mean()
print(f"Overall save rate (on target): {p_bar:.1%} (n={len(on)})")

# ---------- 6 x 6 grid on the half goal ----------
u_edges = np.linspace(0, HALF_WIDTH, NU + 1)
y_edges = np.linspace(0, HEIGHT, NY + 1)
on['col'] = pd.cut(on['u'], u_edges, labels=False, include_lowest=True)
on['row'] = pd.cut(on['y'], y_edges, labels=False, include_lowest=True)
u_mid = (u_edges[:-1] + u_edges[1:]) / 2
y_mid = (y_edges[:-1] + y_edges[1:]) / 2

grid_rate = np.full((NY, NU), np.nan)
grid_n = np.zeros((NY, NU), dtype=int)
for (col, row), g in on.groupby(['col', 'row']):
    grid_n[int(row), int(col)] = len(g)
    grid_rate[int(row), int(col)] = g['saved'].mean()

print("\nZone save rates (rows from top to bottom, columns from centre to post):")
header = "y \\ u".ljust(13) + "".join(f"{m:>12.2f}" for m in u_mid)
print(header)
for row in reversed(range(NY)):
    cells = ""
    for col in range(NU):
        if grid_n[row, col] == 0:
            cells += f"{'-':>12}"
        else:
            cells += f"{grid_rate[row, col]:>6.0%} ({grid_n[row, col]:>3})"
    print(f"{y_mid[row]:>5.2f} m".ljust(13) + cells)

# ---------- Save rates along each axis ----------
cols = on.groupby('col')['saved'].agg(['mean', 'size']).reindex(range(NU))
rows = on.groupby('row')['saved'].agg(['mean', 'size']).reindex(range(NY))

print("\nHorizontal (each column, all heights together):")
for i in range(NU):
    print(f"  u = {u_mid[i]:.2f} m   save rate = {cols['mean'][i]:.1%}   n = {int(cols['size'][i])}")
print("\nVertical (each row, all widths together):")
for i in range(NY):
    print(f"  y = {y_mid[i]:.2f} m   save rate = {rows['mean'][i]:.1%}   n = {int(rows['size'][i])}")

# ---------- Plot 1: mirrored shots on the half goal with zone save rates ----------
fig, ax = plt.subplots(figsize=(8, 6))
ax.axhline(0, color='green', linewidth=3, zorder=1)
ax.plot([0, HALF_WIDTH, HALF_WIDTH], [HEIGHT, HEIGHT, 0], color='black', linewidth=4, zorder=2)
ax.axvline(0, color='grey', linewidth=1, linestyle=':', zorder=1)        # centre of goal

for v in u_edges[1:-1]:
    ax.plot([v, v], [0, HEIGHT], 'k--', linewidth=0.6, alpha=0.5, zorder=1)
for v in y_edges[1:-1]:
    ax.plot([0, HALF_WIDTH], [v, v], 'k--', linewidth=0.6, alpha=0.5, zorder=1)

colours = {"Goal": "green", "Saved": "blue", "Missed": "red"}
for res, col in colours.items():
    part = df[df['result'] == res]
    ax.scatter(part['u'], part['y'], c=col, s=12, alpha=0.5, label=res, zorder=3)

for row in range(NY):
    for col in range(NU):
        if grid_n[row, col] > 0:
            ax.text(u_mid[col], y_mid[row], f"{grid_rate[row, col]:.0%}\nn={grid_n[row, col]}",
                    ha='center', va='center', fontsize=8, zorder=4,
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

ax.set_xlim(-0.2, 5)
ax.set_ylim(-0.2, 4)
ax.set_aspect('equal')
ax.set_xlabel("u = |x|: distance from centre of goal (m)")
ax.set_ylabel("Height y (m)")
ax.set_title("Mirrored penalties on the half goal: save rate per zone")
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig("half_goal_save_rates.png", dpi=200)

# ---------- Plot 2: save rate along each axis ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.scatter(u_mid, cols['mean'], s=cols['size'] * 0.5 + 20, color='blue', zorder=3)
for i in range(NU):
    ax1.annotate(f"n={int(cols['size'][i])}", (u_mid[i], cols['mean'][i]),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
ax1.axhline(p_bar, color='grey', linestyle='--', linewidth=1, label=f"overall {p_bar:.0%}")
ax1.set_xlabel("u = |x|: middle of column (m)")
ax1.set_ylabel("Save rate")
ax1.set_title("Horizontal: save rate per column")
ax1.set_xlim(0, HALF_WIDTH)
ax1.set_ylim(0, max(0.6, np.nanmax(cols['mean']) + 0.1))
ax1.legend()

ax2.scatter(y_mid, rows['mean'], s=rows['size'] * 0.5 + 20, color='blue', zorder=3)
for i in range(NY):
    ax2.annotate(f"n={int(rows['size'][i])}", (y_mid[i], rows['mean'][i]),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
ax2.axhline(p_bar, color='grey', linestyle='--', linewidth=1, label=f"overall {p_bar:.0%}")
ax2.set_xlabel("Height y: middle of row (m)")
ax2.set_ylabel("Save rate")
ax2.set_title("Vertical: save rate per row")
ax2.set_xlim(0, HEIGHT)
ax2.set_ylim(0, max(0.6, np.nanmax(rows['mean']) + 0.1))
ax2.legend()

plt.tight_layout()
plt.savefig("save_rate_axes.png", dpi=200)
plt.show()