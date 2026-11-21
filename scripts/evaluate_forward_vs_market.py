"""Forward (2026-27) evaluation of the August production model on every priced FanDuel line.

Read-only on project data. Usage: python scripts/evaluate_forward_vs_market.py OUT.csv (DEC-104).
For each slate date it uses the first morning prediction file (the run the card
was built from), joins the FanDuel main-line price from the latest snapshot at or
before that run, and the actual stat from nba_data.csv.
"""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1])
PROPS = {"pts": "player_points", "trb": "player_rebounds", "ast": "player_assists",
         "pra": "player_points_rebounds_assists", "pr": "player_points_rebounds",
         "pa": "player_points_assists", "stl": "player_steals", "blk": "player_blocks"}


def norm(name):
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-z ]", "", name.lower())
    return " ".join(re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name).split())


def american_to_prob(a):
    a = pd.to_numeric(a, errors="coerce")
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def morning_file(day_dir):
    files = []
    for f in day_dir.glob("predictions_*_*.csv"):
        m = re.search(r"_(\d{6})(?:_\d+)?\.csv$", f.name)
        if m:
            files.append((m.group(1), f))
    if not files:
        return None, None
    files.sort()
    morning = [x for x in files if "060000" <= x[0] <= "120000"]
    stamp, path = (morning or files)[0]
    return stamp, path


rows = []
for day_dir in sorted((ROOT / "output" / "archive").iterdir()):
    date = day_dir.name
    if not ("2026-10-21" <= date <= "2026-11-19"):
        continue
    stamp, path = morning_file(day_dir)
    if path is None:
        continue
    p = pd.read_csv(path, low_memory=False)
    run_ts = pd.Timestamp(f"{date} {stamp[:2]}:{stamp[2:4]}:{stamp[4:]}").tz_localize("America/Chicago")
    for k in PROPS:
        need = [f"{k}_line", f"{k}_p_over_raw", f"{k}_p_over_cal", f"{k}_pred"]
        if not set(need) <= set(p.columns):
            continue
        sub = pd.DataFrame({
            "game_date": date, "run_ts": run_ts, "player": p["player"], "team": p["team"],
            "prop": k, "line": p[f"{k}_line"], "pred": p[f"{k}_pred"],
            "p_over_raw": p[f"{k}_p_over_raw"], "p_over_cal": p[f"{k}_p_over_cal"],
            "sigma": p.get(f"{k}_sigma"), "meta_prob": p.get(f"{k}_meta_prob"),
            "l10": p.get(f"{k}_l10"), "anchor": p.get(f"{k}_anchor"),
            "mp_l10": p.get("mp_l10"), "injury_adj": p.get("injury_adj"),
            "game_total": p.get("game_total"), "team_spread": p.get("team_spread"),
        })
        rows.append(sub.dropna(subset=["line", "p_over_cal"]))
pred = pd.concat(rows, ignore_index=True)
pred["player_norm"] = pred.player.map(norm)
print("prediction rows", len(pred), "dates", pred.game_date.nunique())

# FanDuel prices: latest main-line snapshot at or before the morning run.
chunks = []
for ch in pd.read_csv(ROOT / "data" / "historical_lines.csv", chunksize=500000, low_memory=False):
    ch = ch[(ch.source == "odds_api_fanduel") & (ch.game_date >= "2026-10-21")]
    if len(ch):
        chunks.append(ch[["game_date", "player_norm", "prop", "line", "snapshot_ts", "over_price", "under_price"]])
fd = pd.concat(chunks, ignore_index=True)
fd["player_norm"] = fd.player_norm.map(norm)
fd["snap"] = pd.to_datetime(fd.snapshot_ts, utc=True, format="mixed")
inv = {v: k for k, v in PROPS.items()}
fd["prop"] = fd.prop.map(inv)
fd = fd.dropna(subset=["prop", "over_price", "under_price"])
m = pred.merge(fd, on=["game_date", "player_norm", "prop", "line"], how="inner")
m = m[m.snap <= m.run_ts.dt.tz_convert("UTC")]
m = m.sort_values("snap").groupby(["game_date", "player_norm", "prop", "line"], as_index=False).tail(1)
po, pu = american_to_prob(m.over_price), american_to_prob(m.under_price)
m["nv_over"] = po / (po + pu)
m["dec_under"] = np.where(m.under_price < 0, 1 + 100 / -m.under_price, 1 + m.under_price / 100)
m["dec_over"] = np.where(m.over_price < 0, 1 + 100 / -m.over_price, 1 + m.over_price / 100)
print("priced rows", len(m))

# Actuals.
box = pd.read_csv(ROOT / "data" / "nba_data.csv", low_memory=False,
                  usecols=["player", "game_date", "mp", "pts", "trb", "ast", "stl", "blk"])
box["game_date"] = pd.to_datetime(box.game_date).dt.strftime("%Y-%m-%d")
box = box[box.game_date >= "2026-10-21"].copy()
box["pra"], box["pr"], box["pa"] = box.pts + box.trb + box.ast, box.pts + box.trb, box.pts + box.ast
box["player_norm"] = box.player.map(norm)
long = box.melt(id_vars=["game_date", "player_norm", "mp"], value_vars=list(PROPS), var_name="prop", value_name="actual")
long = long.drop_duplicates(["game_date", "player_norm", "prop"])
m = m.merge(long, on=["game_date", "player_norm", "prop"], how="inner")
m = m[pd.to_numeric(m.mp, errors="coerce").fillna(0) > 0]
m = m[m.actual != m.line]  # drop pushes
m["over_hit"] = (m.actual > m.line).astype(int)
print("graded non-push rows", len(m), "dates", m.game_date.nunique())
m.drop(columns=["snapshot_ts"]).to_csv(OUT, index=False)
