"""
Part 1: all penalties on the goal, with totals.
Part 2: zones, errors from each zone centre, variance and c_x, c_y for the landing PDF.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Settings ----------
CSV_FILE = 'penalties.csv'
W = 3.66              # half the goal width (m)
H = 2.44              # crossbar height (m)
SIDE_EDGE = 1.22      # zone border between side and middle (m)
LEVEL_EDGE = 1.22     # zone border between low and high (m)

GROUPS = {
    "Goal": "Goal",
    "Saved": "Saved", "Saved Off Target": "Saved", "Saved to Post": "Saved",
    "Off T": "Missed", "Post": "Missed", "Wayward": "Missed", "Blocked": "Missed",
}
COLOURS = {"Goal": "green", "Saved": "blue", "Missed": "red"}


def draw_goal(ax):
    ax.axhline(0, color='green', linewidth=2)
    ax.plot([-W, -W, W, W], [0, H, H, 0], color='black', linewidth=3)
    ax.set_aspect('equal')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


# ---------- Load ----------
df = pd.read_csv(CSV_FILE)
df['result'] = df['outcome'].map(GROUPS)


# ===================== PART 1: all penalties =====================
counts = df['result'].value_counts()
total = len(df)

fig, ax = plt.subplots(figsize=(10, 4.5))
draw_goal(ax)
for res, col in COLOURS.items():
    part = df[df['result'] == res]
    ax.plot(part['x'], part['y'], '.', color=col, label=res)

ax.text(0.01, 0.97,
        f"Total: {total}\n"
        f"Made: {counts.get('Goal', 0)} ({counts.get('Goal', 0) / total:.1%})\n"
        f"Saved: {counts.get('Saved', 0)} ({counts.get('Saved', 0) / total:.1%})\n"
        f"Missed: {counts.get('Missed', 0)} ({counts.get('Missed', 0) / total:.1%})",
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(facecolor='white', edgecolor='grey'))
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-0.2, 4)
ax.legend(loc='upper right')
ax.set_title("Penalties at the goal line")
plt.tight_layout()
plt.savefig("1_all_penalties.png", dpi=200)


# ===================== PART 2: zones and c =====================
def zone(row):
    side = "left" if row['x'] < -SIDE_EDGE else ("right" if row['x'] > SIDE_EDGE else "middle")
    level = "low" if row['y'] < LEVEL_EDGE else "high"
    return f"{side}-{level}"

df['zone'] = df.apply(zone, axis=1)

# Errors from each zone's centre (the assumed aim point)
df['ex'] = df['x'] - df.groupby('zone')['x'].transform('mean')
df['ey'] = df['y'] - df.groupby('zone')['y'].transform('mean')
r_all = df['ex'].corr(df['ey'])
print(f"Correlation between horizontal and vertical errors (all shots): r = {r_all:.3f}")

for z, g in df.groupby('zone'):
    print(f"  {z:<12} r = {g['ex'].corr(g['ey']):.3f}  (n = {len(g)})")

import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
plt.plot(df['ex'], df['ey'], '.', alpha=0.4)
plt.axhline(0, color='grey', linewidth=0.8)
plt.axvline(0, color='grey', linewidth=0.8)
plt.xlabel("Horizontal error $e_x$ (m)")
plt.ylabel("Vertical error $e_y$ (m)")
plt.title(f"Horizontal vs vertical error, r = {r_all:.2f}")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig("error_correlation.png", dpi=200)
plt.show()
# Variance = mean of squared errors; c = sqrt(5 * variance)
print(f"{'Zone':<12} {'n':>5} {'x_mean':>7} {'y_mean':>7} {'var_x':>7} {'var_y':>7} "
      f"{'c_x':>6} {'c_y':>6} {'max|ex|':>8} {'max|ey|':>8}")
for z, g in df.groupby('zone'):
    vx, vy = (g['ex']**2).mean(), (g['ey']**2).mean()
    print(f"{z:<12} {len(g):>5} {g['x'].mean():>7.2f} {g['y'].mean():>7.2f} {vx:>7.3f} {vy:>7.3f} "
          f"{np.sqrt(5 * vx):>6.2f} {np.sqrt(5 * vy):>6.2f} "
          f"{g['ex'].abs().max():>8.2f} {g['ey'].abs().max():>8.2f}")

var_x, var_y = (df['ex']**2).mean(), (df['ey']**2).mean()
c_x, c_y = np.sqrt(5 * var_x), np.sqrt(5 * var_y)
print(f"\nAll shots: n = {total}, var_x = {var_x:.3f}, var_y = {var_y:.3f}, "
      f"c_x = {c_x:.2f} m, c_y = {c_y:.2f} m")
print(f"k_x = 3/(4c_x^3) = {3 / (4 * c_x**3):.3f}, k_y = 3/(4c_y^3) = {3 / (4 * c_y**3):.3f}")

# Graph: zones, shots and zone centres
fig, ax = plt.subplots(figsize=(10, 4.5))
draw_goal(ax)
for v in [-SIDE_EDGE, SIDE_EDGE]:
    ax.axvline(v, color='grey', linestyle='--', linewidth=0.8)
ax.axhline(LEVEL_EDGE, color='grey', linestyle='--', linewidth=0.8)

for res, col in COLOURS.items():
    part = df[df['result'] == res]
    ax.plot(part['x'], part['y'], '.', color=col, label=res)

centres = df.groupby('zone')[['x', 'y']].mean()
ax.plot(centres['x'], centres['y'], 'kx', markersize=12, markeredgewidth=2, label="Zone centre")
for z, row in centres.iterrows():
    ax.annotate(z, (row['x'], row['y']), textcoords="offset points", xytext=(0, 8),
                ha='center', fontsize=8)

ax.text(0.01, 0.97, f"c_x = {c_x:.2f} m\nc_y = {c_y:.2f} m", transform=ax.transAxes,
        va='top', fontsize=9, bbox=dict(facecolor='white', edgecolor='grey'))
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-0.2, 4)
ax.legend(loc='upper right', fontsize=8)
ax.set_title("Zones used to estimate the aiming error")
plt.tight_layout()
plt.savefig("2_zones.png", dpi=200)

plt.show()