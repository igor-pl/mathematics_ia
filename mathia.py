

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore")  # statsbombpy warns that we use the free data

# ---------------- settings ----------------
CSV_FILE = "penalties.csv"
INCLUDE_SHOOTOUTS = True   # period 5 = penalty shootout
# None = every free competition (slow, but most data).
# Or a list of names to test quickly, e.g. ["FIFA World Cup"]
ONLY_COMPETITIONS = None
YARD = 0.9144              # StatsBomb uses yards
HALF_WIDTH = 3.66          # goal is 7.32 m wide
HEIGHT = 2.44              # crossbar height in metres

# Outcomes grouped into the three results the model uses
GROUP = {
    "Goal": "Goal",
    "Saved": "Saved", "Saved Off Target": "Saved", "Saved to Post": "Saved",
    "Off T": "Missed", "Wayward": "Missed", "Post": "Missed", "Blocked": "Missed",
}
COLOURS = {"Goal": "tab:green", "Saved": "tab:blue", "Missed": "tab:red"}


def download_penalties():
    """Go through every free match and keep the penalties."""
    rows = []
    comps = sb.competitions()
    if ONLY_COMPETITIONS:
        comps = comps[comps.competition_name.isin(ONLY_COMPETITIONS)]
    for i, c in enumerate(comps.itertuples(), start=1):
        print(f"[{i}/{len(comps)}] {c.competition_name} {c.season_name}")
        matches = sb.matches(competition_id=c.competition_id, season_id=c.season_id)
        for match_id in matches.match_id:
            ev = sb.events(match_id=match_id)
            if "shot_type" not in ev.columns:
                continue
            pens = ev[(ev["type"] == "Shot") & (ev["shot_type"] == "Penalty")]
            if not INCLUDE_SHOOTOUTS:
                pens = pens[pens["period"] != 5]
            for p in pens.itertuples():
                end = p.shot_end_location
                if not isinstance(end, (list, np.ndarray)) or len(end) < 3:
                    continue  # no height recorded, cannot place it on the goal
                rows.append({
                    "competition": c.competition_name,
                    "season": c.season_name,
                    "player": p.player,
                    "shootout": p.period == 5,
                    "outcome": p.shot_outcome,
                    # StatsBomb: goal centre is y = 40 yards.
                    # Positive x = shooter's right (check with one video).
                    "x": (end[1] - 40) * YARD,
                    "y": end[2] * YARD,
                })
    df = pd.DataFrame(rows)
    df["result"] = df["outcome"].map(GROUP).fillna("Missed")
    df.to_csv(CSV_FILE, index=False)
    return df


def save_rate_grid(df):
    """3 x 3 grid: saved / (goals + saved) for shots inside the goal."""
    inside = df[(df.x.abs() <= HALF_WIDTH) & (df.y.between(0, HEIGHT))
                & df.result.isin(["Goal", "Saved"])].copy()
    xb = np.linspace(-HALF_WIDTH, HALF_WIDTH, 4)
    yb = np.linspace(0, HEIGHT, 4)
    inside["col"] = pd.cut(inside.x, xb, labels=False, include_lowest=True)
    inside["row"] = pd.cut(inside.y, yb, labels=False, include_lowest=True)
    grid = np.full((3, 3), np.nan)
    count = np.zeros((3, 3), dtype=int)
    for (r, col), g in inside.groupby(["row", "col"]):
        count[int(r), int(col)] = len(g)
        grid[int(r), int(col)] = (g.result == "Saved").mean()
    return grid, count, xb, yb


def draw(df):
    fig, ax = plt.subplots(figsize=(11, 5))

    # Goal frame and ground
    ax.plot([-HALF_WIDTH, -HALF_WIDTH, HALF_WIDTH, HALF_WIDTH],
            [0, HEIGHT, HEIGHT, 0], color="black", lw=4)
    ax.axhline(0, color="darkgreen", lw=2)

    # 3 x 3 grid with save rates
    grid, count, xb, yb = save_rate_grid(df)
    for v in xb[1:-1]:
        ax.plot([v, v], [0, HEIGHT], "k--", lw=0.8, alpha=0.5)
    for v in yb[1:-1]:
        ax.plot([-HALF_WIDTH, HALF_WIDTH], [v, v], "k--", lw=0.8, alpha=0.5)
    for r in range(3):
        for col in range(3):
            if count[r, col] > 0:
                ax.text((xb[col] + xb[col + 1]) / 2, (yb[r] + yb[r + 1]) / 2,
                        f"{grid[r, col]:.0%} saved\n(n={count[r, col]})",
                        ha="center", va="center", fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # Penalties
    for result, colour in COLOURS.items():
        part = df[df.result == result]
        ax.scatter(part.x, part.y, s=25, c=colour, alpha=0.7,
                   edgecolors="white", linewidths=0.5,
                   label=f"{result} ({len(part)})")

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-0.3, 4)
    ax.set_aspect("equal")
    ax.set_xlabel("x: horizontal position from goal centre (m)")
    ax.set_ylabel("y: height (m)")
    ax.set_title(f"Where {len(df)} penalties crossed the goal line (StatsBomb Open Data)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("penalties_on_goal.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    if os.path.exists(CSV_FILE):
        data = pd.read_csv(CSV_FILE)
    else:
        data = download_penalties()

    print(f"\nPenalties with a known end point: {len(data)}")
    print(data.result.value_counts().to_string())

    # Step 4 of the plan: are left/right and height related?
    r = data.x.corr(data.y)
    print(f"\nCorrelation between x and height: r = {r:.3f}")

    draw(data)