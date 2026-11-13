"""Retain immutable observations when graded history disagrees with refreshed boxes."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nba_props as nba


def reconcile():
    box_path = ROOT / "data/nba_data.csv"
    box = pd.read_csv(box_path, low_memory=False)
    box["game_date"] = pd.to_datetime(box.game_date, format="mixed").dt.strftime("%Y-%m-%d")
    box["player_norm"] = box.player.map(nba._normalize_name)
    for prop, components in {"pra": ["pts", "trb", "ast"], "pr": ["pts", "trb"], "pa": ["pts", "ast"]}.items():
        box[prop] = box[components].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(components))
    if box.duplicated(["game_date", "player_norm"]).any():
        raise ValueError("Ambiguous player/date box scores prevent reconciliation")
    lookup = box.set_index(["game_date", "player_norm"])
    box_hash = hashlib.sha256(box_path.read_bytes()).hexdigest()
    directory = ROOT / "output/grade_reconciliation"
    directory.mkdir(parents=True, exist_ok=True)
    conflicts = []
    checked = 0
    for name in ["picks_history.csv", "candidates_history.csv"]:
        source = ROOT / "output" / name
        history = pd.read_csv(source, keep_default_na=False)
        dates = pd.to_datetime(history.game_date, errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
        for index, row in history[history.result.isin(["WIN", "LOSS", "PUSH"])].iterrows():
            key = (dates.loc[index], nba._normalize_name(row.player))
            prop = str(row.prop).lower()
            if key not in lookup.index or prop not in lookup.columns:
                continue
            actual = pd.to_numeric(lookup.loc[key, prop], errors="coerce")
            old_actual = pd.to_numeric(row.actual, errors="coerce")
            line = pd.to_numeric(row.line, errors="coerce")
            if pd.isna(actual) or pd.isna(line) or str(row.direction).upper() not in {"OVER", "UNDER"}:
                continue
            checked += 1
            outcome = "PUSH" if actual == line else "WIN" if (actual > line) == (str(row.direction).upper() == "OVER") else "LOSS"
            if actual == old_actual and outcome == row.result:
                continue
            event = {"source_file": name, "game_date": key[0], "player": row.player,
                     "prop": row.prop, "direction": row.direction, "line": float(line),
                     "stored_actual": float(old_actual) if pd.notna(old_actual) else None,
                     "observed_actual": float(actual), "stored_result": row.result,
                     "observed_result": outcome}
            event_id = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
            destination = directory / f"{event_id}.json"
            if not destination.exists():
                event.update(observed_at=pd.Timestamp.now(tz="UTC").isoformat(), box_sha256=box_hash)
                nba._atomic_text(json.dumps(event, indent=2), destination)
            else:
                event = json.loads(destination.read_text(encoding="utf-8"))
            conflicts.append(event)
    summary = {"checked_history_rows": checked, "current_conflicts": len(conflicts),
               "outcome_conflicts": sum(e["stored_result"] != e["observed_result"] for e in conflicts),
               "conflicts": conflicts,
               "scope": "Local refreshed box observations. Raw grades preserved. Missing players and DNP decisions require event-final evidence."}
    nba._atomic_text(json.dumps(summary, indent=2), ROOT / "output/grade_reconciliation_summary.json")
    return summary


if __name__ == "__main__":
    print(json.dumps(reconcile(), indent=2))
