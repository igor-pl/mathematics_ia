"""
Step function + landing PDF: probability of scoring when aiming at the centre of each square.

P(a,b) = sum_i sum_j [Gx(x_{j+1}-a) - Gx(x_j-a)] [Gy(y_{i+1}-b) - Gy(y_i-b)] (1 - s_ij)
G(e) = 3/(4c^3)(c^2 e - e^3/3), cut at +-1/2 beyond +-c.
Bottom row: lower limit -infinity (the part below the ground counts as a ground shot).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------- Settings ----------
CSV_FILE = 'penalties.csv'
W, H = 3.66, 2.44
CX, CY = 1.36, 1.05          # accuracy of the landing hill (m)
SQ_ROWS = 4                  # 4 rows -> squares of 0.61 m, 12 x 4 on the whole goal

GROUPS = {
    "Goal": "Goal",
    "Saved": "Saved", "Saved Off Target": "Saved", "Saved to Post": "Saved",
    "Off T": "Missed", "Post": "Missed", "Wayward": "Missed", "Blocked": "Missed",
}

# ---------- Step save function s_ij (mirrored half goal, then copied to both sides) ----------
df = pd.read_csv(CSV_FILE)
df['result'] = df['outcome'].map(GROUPS)
on = df[(df['x'].abs() <= W) & df['y'].between(0, H) & df['result'].isin(['Goal', 'Saved'])].copy()
on['saved'] = (on['result'] == 'Saved').astype(int)
on['u'] = on['x'].abs()
p_bar = on['saved'].mean()

size = H / SQ_ROWS
u_edges = np.linspace(0, W, int(round(W / size)) + 1)
y_edges = np.linspace(0, H, SQ_ROWS + 1)
on['col'] = pd.cut(on['u'], u_edges, labels=False, include_lowest=True)
on['row'] = pd.cut(on['y'], y_edges, labels=False, include_lowest=True)

NU, NY = len(u_edges) - 1, len(y_edges) - 1
s_half = np.full((NY, NU), p_bar)            # empty squares: average save rate
for (c, r), grp in on.groupby(['col', 'row']):
    s_half[int(r), int(c)] = grp['saved'].mean()

x_edges = np.concatenate([-u_edges[::-1], u_edges[1:]])     # whole goal
s = np.hstack([s_half[:, ::-1], s_half])                    # s[row, col], row 0 = bottom
x_mid = (x_edges[:-1] + x_edges[1:]) / 2
y_mid = (y_edges[:-1] + y_edges[1:]) / 2


# ---------- The formula ----------
def G(e, c):
    e = np.clip(e, -c, c)
    return 3 / (4 * c**3) * (c**2 * e - e**3 / 3)

def P(a, b):
    col_part = G(x_edges[1:] - a, CX) - G(x_edges[:-1] - a, CX)       # hill in each column
    y_low = y_edges[:-1].copy()
    y_low[0] = -np.inf                                                # ground shots
    row_part = G(y_edges[1:] - b, CY) - G(y_low - b, CY)              # hill in each row
    V = np.outer(row_part, col_part)                                  # V_ij
    return np.sum(V * (1 - s)), V


# ---------- Evaluate at the centre of every square of the 6 x 4 half-goal grid ----------
# (the sum above still runs over the whole goal, because the hill can reach the other half)
u_mid = (u_edges[:-1] + u_edges[1:]) / 2
Pc = np.zeros((NY, NU))
for i, b in enumerate(y_mid):
    for j, a in enumerate(u_mid):
        Pc[i, j], _ = P(a, b)

print(f"Square size {size:.2f} m, overall save rate {p_bar:.3f}\n")
print(f"P(goal) when aiming at the centre of each square, {NU} x {NY} half goal (top row first):")
print("         " + "".join(f"{a:>7.2f}" for a in u_mid))
for i in reversed(range(NY)):
    print(f"y={y_mid[i]:.2f}  " + "".join(f"{Pc[i, j]:>7.3f}" for j in range(NU)))

bi, bj = np.unravel_index(np.argmax(Pc), Pc.shape)
best_a, best_b = u_mid[bj], y_mid[bi]
print(f"\nBest square: centre ({best_a:.2f}, {best_b:.2f}), P = {Pc[bi, bj]:.3f}")


# ---------- Figure 1: P at the centre of every half-goal square ----------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
cmap = plt.get_cmap('RdYlGn')
lo, hi = Pc.min(), Pc.max()
for i in range(NY):
    for j in range(NU):
        ax.add_patch(Rectangle((u_edges[j], y_edges[i]), size, size,
                               facecolor=cmap((Pc[i, j] - lo) / (hi - lo)),
                               edgecolor='grey', linewidth=0.5))
        ax.text(u_mid[j], y_mid[i], f"{Pc[i, j]:.3f}", ha='center', va='center', fontsize=9)
ax.add_patch(Rectangle((best_a - size / 2, best_b - size / 2), size, size,
                       fill=False, edgecolor='black', linewidth=3))
ax.plot([0, W, W], [H, H, 0], color='black', linewidth=3)
ax.axvline(0, color='grey', linestyle=':', linewidth=1)
ax.axhline(0, color='green', linewidth=2)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, label="P(goal)")
ax.set_xlim(-0.1, W + 0.3)
ax.set_ylim(-0.2, H + 0.3)
ax.set_aspect('equal')
ax.set_xlabel("Aim point a = |x| (m), 0 = centre of goal")
ax.set_ylabel("Aim point b (m)")
ax.set_title("P(goal) when aiming at the centre of each square")
plt.tight_layout()
plt.savefig("8_step_square_centres.png", dpi=200)


# ---------- Figure 2: how the hill covers several squares (best aim point) ----------
_, V = P(best_a, best_b)
fig, ax = plt.subplots(figsize=(10, 4))
for i in range(NY):
    for j in range(len(x_mid)):
        x0, y0 = x_edges[j], y_edges[i]
        ax.add_patch(Rectangle((x0, y0), size, size, facecolor=plt.cm.Blues(V[i, j] / V.max()),
                               edgecolor='grey', linewidth=0.5))
        if V[i, j] > 0.001:
            ax.text(x_mid[j], y_mid[i], f"V={V[i, j]:.3f}\ns={s[i, j]:.2f}",
                    ha='center', va='center', fontsize=7)
ax.add_patch(Rectangle((best_a - CX, best_b - CY), 2 * CX, 2 * CY, fill=False,
                       edgecolor='red', linewidth=2, linestyle='--', label="extent of the hill"))
ax.plot(best_a, best_b, 'rx', markersize=12, markeredgewidth=2, label="aim point")
ax.plot([-W, -W, W, W], [0, H, H, 0], color='black', linewidth=3)
ax.axhline(0, color='green', linewidth=2)
ax.set_xlim(-W - 0.3, W + 0.3)
ax.set_ylim(min(-0.2, best_b - CY - 0.1), H + 0.3)
ax.set_aspect('equal')
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"Volume of the hill above each square when aiming at ({best_a:.2f}, {best_b:.2f})")
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig("9_hill_over_squares.png", dpi=200)

plt.show()