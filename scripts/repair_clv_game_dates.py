"""Repair paid CLV rows labelled with UTC date instead of NBA local game date.

The manifest date is authoritative. By default this prints a preview. ``--write``
atomically rewrites parsed checkpoints and replaces only the paid checkpoint
slice of the master line store.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from scripts.execute_clv_snapshot_manifest import (
        PHYSICAL_KEY, ROOT, atomic_write_csv, normalize_name, normalize_lines,
    )
except ModuleNotFoundError:
    from execute_clv_snapshot_manifest import (
        PHYSICAL_KEY, ROOT, atomic_write_csv, normalize_name, normalize_lines,
    )


def repair(manifest_path: Path, checkpoint_path: Path, master_path: Path, write: bool) -> dict:
    manifest = pd.read_csv(manifest_path, dtype={"nba_game_id": str})
    expected = manifest.set_index("manifest_key")["game_date"].astype(str).to_dict()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    completed = checkpoint.get("completed", {})

    frames = []
    mismatched_files = 0
    mismatched_rows = 0
    player_norm_mismatched_files = 0
    player_norm_mismatched_rows = 0
    paid_pairs = set()
    for key, record in completed.items():
        parsed_path = ROOT / str(record["parsed_path"])
        frame = normalize_lines(pd.read_csv(parsed_path, low_memory=False))
        target_date = expected[key]
        mismatch = ~frame["game_date"].astype(str).eq(target_date)
        if mismatch.any():
            mismatched_files += 1
            mismatched_rows += int(mismatch.sum())
            frame.loc[:, "game_date"] = target_date
            if write:
                atomic_write_csv(parsed_path, frame)
        corrected_norm = frame["player"].map(normalize_name)
        norm_mismatch = ~frame["player_norm"].astype(str).eq(corrected_norm)
        if norm_mismatch.any():
            player_norm_mismatched_files += 1
            player_norm_mismatched_rows += int(norm_mismatch.sum())
            frame.loc[:, "player_norm"] = corrected_norm
            if write:
                atomic_write_csv(parsed_path, frame)
        frames.append(frame)
        paid_pairs.add((str(record.get("odds_api_event_id", "")), str(manifest.loc[manifest.manifest_key.eq(key), "requested_snapshot_ts"].iloc[0])))

    paid = normalize_lines(pd.concat(frames, ignore_index=True)) if frames else normalize_lines(pd.DataFrame())
    master = normalize_lines(pd.read_csv(master_path, low_memory=False))
    master_pairs = list(zip(master["event_id"].fillna("").astype(str), master["requested_snapshot_ts"].fillna("").astype(str)))
    paid_mask = pd.Series([pair in paid_pairs for pair in master_pairs], index=master.index)
    replaced_rows = int(paid_mask.sum())
    rebuilt = pd.concat([master.loc[~paid_mask], paid], ignore_index=True)
    rebuilt = normalize_lines(rebuilt).drop_duplicates(PHYSICAL_KEY, keep="last").reset_index(drop=True)
    if write:
        atomic_write_csv(master_path, rebuilt)

    return {
        "write": write,
        "completed_checkpoints": len(completed),
        "mismatched_parsed_files": mismatched_files,
        "mismatched_parsed_rows": mismatched_rows,
        "player_norm_mismatched_files": player_norm_mismatched_files,
        "player_norm_mismatched_rows": player_norm_mismatched_rows,
        "paid_rows_in_checkpoints": int(len(paid)),
        "paid_master_rows_replaced": replaced_rows,
        "master_rows_before": int(len(master)),
        "master_rows_after": int(len(rebuilt)),
        "physical_duplicates_after": int(rebuilt.duplicated(PHYSICAL_KEY).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "clv_snapshot_manifest_2026_live.csv")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data" / "clv_snapshot_execution_checkpoint.json")
    parser.add_argument("--master", type=Path, default=ROOT / "data" / "historical_lines.csv")
    args = parser.parse_args()
    print(json.dumps(repair(args.manifest, args.checkpoint, args.master, args.write), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
