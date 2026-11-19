import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore")

comps = sb.competitions()
# print(comps[["competition_name", "season_name"]].to_string())

wanted = ["Premier League", "La Liga", "FIFA World Cup", "Champions League", "Bundesliga",
          "UEFA Euro", "UEFA Europa League", "Copa del Rey"]

comps = comps[comps["competition_name"].isin(wanted)]
# print(comps[["competition_name", "season_name"]])

all_penalties = []

for c in comps.itertuples():
    # print(c.competition_name, c.season_name)
    matches = sb.matches(competition_id=c.competition_id, season_id=c.season_id)

    for match_id in matches["match_id"]:
        events = sb.events(match_id=match_id)

        if "shot_type" not in events.columns:
            continue  # no shots in this match

        shots = events[events["type"] == "Shot"]
        pens = shots[shots["shot_type"] == "Penalty"]

        for p in pens.itertuples():
            end = p.shot_end_location  # [x, y, z]
            if not isinstance(end, list) or len(end) < 3:
                continue  # no height recorded, skip it
            all_penalties.append({
                "competition": c.competition_name,
                "season": c.season_name,
                "x": (end[1] - 40) * 0.9144,  # left/right in metres
                "height": end[2] * 0.9144,    # height in metres
                "outcome": p.shot_outcome,
                "shootout": p.period == 5,
            })

df = pd.DataFrame(all_penalties)
df.to_csv("penalties.csv", index=False)

# print(len(df))
# print(df.head())

