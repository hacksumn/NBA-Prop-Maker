"""Read-only repository audit; writes evidence only when --output is supplied.

Run from any directory. Counts every workspace file outside .git, parses active
Python sources, and profiles canonical data without fetching or retraining.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _pick_summary(picks: pd.DataFrame) -> list[dict]:
    """Flat one-unit returns only for graded, exactly priced FanDuel picks."""
    selected = picks[picks["line_source"].eq("fanduel") & picks["result"].isin(["WIN", "LOSS"])
                     & picks["game_date"].ge("2026-10-26")].copy()
    price = pd.to_numeric(selected["side_price"], errors="coerce")
    selected["priced"] = price.abs().ge(100) & np.isfinite(price)
    decimal = np.where(price > 0, 1 + price / 100, 1 + 100 / -price)
    selected["profit"] = np.where(selected.result.eq("WIN"), decimal - 1, -1)
    rows = []
    for prop, group in selected.groupby("prop"):
        priced = group[group.priced]
        rows.append({"prop": prop, "graded": len(group), "wins": int(group.result.eq("WIN").sum()),
                     "priced_graded": len(priced), "priced_wins": int(priced.result.eq("WIN").sum()),
                     "profit_units": float(priced.profit.sum()),
                     "roi": float(priced.profit.mean()) if len(priced) else None})
    return rows


def audit() -> dict:
    """Return JSON-safe diagnostics with raw-source hashes and explicit scope."""
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.relative_to(ROOT).parts)
    roots = {}
    syntax_errors, duplicate_definitions = [], []
    parsed = 0
    for path in files:
        relative = path.relative_to(ROOT)
        bucket = relative.parts[0] if len(relative.parts) > 1 else "root"
        entry = roots.setdefault(bucket, {"files": 0, "bytes": 0})
        entry["files"] += 1
        entry["bytes"] += path.stat().st_size
        if path.suffix != ".py" or bucket not in {"root", "scripts", "tests"}:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            parsed += 1
            names = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.setdefault(node.name, []).append(node.lineno)
            for name, locations in names.items():
                if len(locations) > 1:
                    duplicate_definitions.append({"path": str(relative), "name": name, "lines": locations})
        except (SyntaxError, UnicodeError) as error:
            syntax_errors.append({"path": str(relative), "error": str(error)})

    paths = [ROOT / p for p in ["data/nba_data.csv", "data/historical_lines.csv", "output/picks_history.csv",
                               "output/candidates_history.csv", "data/candidate_future_holdout_predictions.csv"]]
    sources = [_source(p) for p in paths if p.exists()]
    box = pd.read_csv(paths[0], low_memory=False)
    lines = pd.read_csv(paths[1], low_memory=False)
    picks = pd.read_csv(paths[2], low_memory=False)
    snap = pd.to_datetime(lines.snapshot_ts, errors="coerce", utc=True, format="mixed")
    old_snap = pd.to_datetime(lines.snapshot_ts, errors="coerce", utc=True)
    tip = pd.to_datetime(lines.commence_time, errors="coerce", utc=True, format="mixed")
    checks = {
        "box_scores": {"rows": len(box), "players": int(box.player.nunique()),
                       "min_date": str(box.game_date.min()), "max_date": str(box.game_date.max()),
                       "duplicate_player_game_rows": int(box.duplicated(["PLAYER_ID", "GAME_ID"]).sum())},
        "historical_lines": {"rows": len(lines), "unparseable_snapshot_rows": int(snap.isna().sum()),
                             "valid_rows_rejected_by_default_parser": int((old_snap.isna() & snap.notna()).sum()),
                             "unknown_tip_rows": int(tip.isna().sum()),
                             "at_or_after_tip_rows": int((snap >= tip).sum()),
                             "duplicate_physical_keys": int(lines.duplicated(
                                 ["game_date", "player_norm", "prop", "line", "source", "snapshot_ts"]).sum()),
                             "source_counts": lines.source.value_counts().to_dict()},
        "picks": {"rows": len(picks), "result_counts": picks.result.fillna("PENDING").value_counts().to_dict(),
                  "fanduel_performance_start": "2026-10-26",
                  "fanduel_by_prop": _pick_summary(picks)},
    }
    if paths[4].exists():
        # Imported without executing scoring, grading or model loading.
        from scripts.capture_candidate_future_holdout import _evaluation_rows
        stored = pd.read_csv(paths[4], low_memory=False)
        first, counts = _evaluation_rows(stored)
        counts["dates"] = int(first.game_date.nunique())
        checks["future_holdout"] = counts
    return {"generated_at": pd.Timestamp.now(tz="America/Chicago").isoformat(),
            "scope": "Full file inventory; AST check of root/scripts/tests Python; canonical CSV profiles. "
                     "Not a semantic review of every cache, model, notebook or dashboard.",
            "inventory": roots, "python_files_parsed": parsed, "syntax_errors": syntax_errors,
            "duplicate_top_level_definitions": duplicate_definitions, "sources": sources, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New evidence JSON; an existing path is never overwritten")
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(ROOT))
    result = audit()
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
