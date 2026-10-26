"""Audit the paid CLV panel from raw checkpoint through analytical joins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from scripts.execute_clv_snapshot_manifest import (
        PHYSICAL_KEY, ROOT, atomic_write_csv, atomic_write_json, atomic_write_text, normalize_name,
    )
except ModuleNotFoundError:
    from execute_clv_snapshot_manifest import (
        PHYSICAL_KEY, ROOT, atomic_write_csv, atomic_write_json, atomic_write_text, normalize_name,
    )


def load_paid_panel(
    manifest_path: Path | None = None,
    ledger_path: Path | None = None,
    checkpoint_path: Path | None = None,
    master_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load one checkpointed panel; custom paths support isolated expansions."""
    manifest_path = manifest_path or ROOT / "data" / "clv_snapshot_manifest_2026_live.csv"
    ledger_path = ledger_path or ROOT / "data" / "clv_snapshot_execution_ledger.csv"
    checkpoint_path = checkpoint_path or ROOT / "data" / "clv_snapshot_execution_checkpoint.json"
    master_path = master_path or ROOT / "data" / "historical_lines.csv"
    manifest = pd.read_csv(manifest_path, dtype={"nba_game_id": str})
    ledger = pd.read_csv(ledger_path, dtype={"nba_game_id": str})
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    master = pd.read_csv(master_path, low_memory=False)
    mappings = []
    for key, record in checkpoint["completed"].items():
        row = manifest.loc[manifest["manifest_key"].eq(key)].iloc[0]
        mappings.append({
            "event_id": str(record["odds_api_event_id"]),
            "requested_snapshot_ts": str(row["requested_snapshot_ts"]),
            "manifest_key": key,
            "nba_game_id": str(row["nba_game_id"]),
            "snapshot_kind": row["snapshot_kind"],
            "expected_game_date": row["game_date"],
            "tip_ts_utc": row["tip_ts_utc"],
        })
    mapping = pd.DataFrame(mappings).drop_duplicates(["event_id", "requested_snapshot_ts"])
    master["event_id"] = master["event_id"].fillna("").astype(str)
    master["requested_snapshot_ts"] = master["requested_snapshot_ts"].fillna("").astype(str)
    paid = master.merge(mapping, on=["event_id", "requested_snapshot_ts"], how="inner", validate="many_to_one")
    return paid, manifest, ledger, checkpoint


def audit(
    manifest_path: Path | None = None,
    ledger_path: Path | None = None,
    checkpoint_path: Path | None = None,
    master_path: Path | None = None,
) -> tuple[dict, pd.DataFrame, str]:
    master_path = master_path or ROOT / "data" / "historical_lines.csv"
    paid, manifest, ledger, checkpoint = load_paid_panel(
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        checkpoint_path=checkpoint_path,
        master_path=master_path,
    )
    ready_ledger = ledger[ledger["manifest_status"].eq("ready_for_event_lookup")].copy()
    paired = paid[["over_price", "under_price"]].notna().all(axis=1)
    timing = ready_ledger.merge(manifest[["manifest_key", "tip_ts_utc"]], on="manifest_key")
    timing["request"] = pd.to_datetime(timing["requested_snapshot_ts"], utc=True)
    timing["actual"] = pd.to_datetime(timing["actual_snapshot_ts"], utc=True)
    timing["tip"] = pd.to_datetime(timing["tip_ts_utc"], utc=True)
    timing["lag_seconds"] = (timing["request"] - timing["actual"]).dt.total_seconds()

    box = pd.read_csv(ROOT / "data" / "nba_data.csv", usecols=["player", "game_date"], low_memory=False)
    box["game_date"] = pd.to_datetime(box["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    box["player_norm"] = box["player"].map(normalize_name)
    box = box[["game_date", "player_norm"]].drop_duplicates().assign(box_join=1)
    paid_keys = paid[["game_date", "player_norm", "prop"]].drop_duplicates()
    paid_key_join = paid_keys.merge(box, on=["game_date", "player_norm"], how="left")

    coverage = paid.groupby("prop").agg(
        rows=("line", "size"),
        event_snapshots=("manifest_key", "nunique"),
        games=("nba_game_id", "nunique"),
        bookmakers=("bookmaker", "nunique"),
    ).reset_index().sort_values("event_snapshots", ascending=False)

    by_kind = paid.groupby("snapshot_kind").agg(
        rows=("line", "size"),
        event_snapshots=("manifest_key", "nunique"),
        games=("nba_game_id", "nunique"),
        bookmakers=("bookmaker", "nunique"),
        markets=("prop", "nunique"),
    ).to_dict("index")
    for values in by_kind.values():
        for key, value in list(values.items()):
            values[key] = int(value)

    empty = ready_ledger[ready_ledger["execution_status"].eq("completed_empty")]
    summary = {
        "manifest": {
            "total_rows": int(len(manifest)),
            "ready_rows": int(manifest["execution_status"].eq("ready_for_event_lookup").sum()),
            "blocked_rows": int(manifest["execution_status"].eq("blocked").sum()),
            "completed_rows": int(ready_ledger["execution_status"].str.startswith("completed", na=False).sum()),
            "completed_with_data": int(ready_ledger["execution_status"].eq("completed_with_data").sum()),
            "completed_empty": int(len(empty)),
        },
        "credits": {
            "used": int(checkpoint["credits_used"]),
            "remaining": checkpoint.get("last_api_balance", {}).get("requests_remaining"),
        },
        "rows": {
            "master": int(len(pd.read_csv(master_path, usecols=["game_date"]))),
            "paid": int(len(paid)),
            "paired_prices": int(paired.sum()),
            "paired_price_rate": float(paired.mean()),
            "physical_duplicates": int(paid.duplicated(PHYSICAL_KEY).sum()),
            "date_mismatches": int((paid["game_date"] != paid["expected_game_date"]).sum()),
        },
        "timing": {
            "actual_after_request": int((timing["actual"] > timing["request"]).sum()),
            "actual_at_or_after_tip": int((timing["actual"] >= timing["tip"]).sum()),
            "median_lag_seconds": float(timing["lag_seconds"].median()),
            "max_lag_seconds": float(timing["lag_seconds"].max()),
        },
        "joins": {
            "unique_player_prop_date_keys": int(len(paid_key_join)),
            "box_score_joined_keys": int(paid_key_join["box_join"].notna().sum()),
            "box_score_join_rate": float(paid_key_join["box_join"].notna().mean()),
        },
        "by_snapshot_kind": by_kind,
        "empty_manifest_keys": empty["manifest_key"].tolist(),
        "markets_returned": coverage["prop"].tolist(),
        "market_never_returned": ["player_turnovers"],
    }

    report = f"""# CLV Panel Acquisition Audit

## Outcome

- Completed **{summary['manifest']['completed_rows']}/{summary['manifest']['ready_rows']}** evidence-safe snapshots; **{summary['manifest']['completed_with_data']}** contain odds and **{summary['manifest']['completed_empty']}** are genuine empty-market observations.
- Used **{summary['credits']['used']:,}** Odds API credits; **{summary['credits']['remaining']:,}** remain.
- Preserved **{summary['rows']['paid']:,}** paid line/price rows in the master store with **{summary['rows']['physical_duplicates']}** physical-key duplicates.
- Paired over/under prices: **{summary['rows']['paired_prices']:,} / {summary['rows']['paid']:,} ({summary['rows']['paired_price_rate']:.2%})**.
- Box-score join coverage at unique player/prop/date grain: **{summary['joins']['box_score_join_rate']:.2%}**.

## Integrity checks

- Paid rows with wrong local game date: **{summary['rows']['date_mismatches']}**.
- Actual API snapshots after requested time: **{summary['timing']['actual_after_request']}**.
- Actual API snapshots at/after tip: **{summary['timing']['actual_at_or_after_tip']}**.
- API snapshot lag: median **{summary['timing']['median_lag_seconds']:.0f}s**, maximum **{summary['timing']['max_lag_seconds']:.0f}s**.

## Coverage implications

- Historical props were returned for eight markets. `player_turnovers` was never returned and should not be retried blindly.
- Empty event snapshots are retained as market-withdrawal/unavailability labels, not treated as ingest failures.
- The panel supports decision/T-60/T-5 CLV where the archived decision timestamp exists and T-60/T-5 late movement for the complete 187-game window.
- Remaining credits should be spent only after feature materialization identifies a specific missing comparison.
"""
    return summary, coverage, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "clv_snapshot_manifest_2026_live.csv")
    parser.add_argument("--ledger", type=Path, default=ROOT / "data" / "clv_snapshot_execution_ledger.csv")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data" / "clv_snapshot_execution_checkpoint.json")
    parser.add_argument("--master", type=Path, default=ROOT / "data" / "historical_lines.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "output" / "clv_panel_audit.json")
    parser.add_argument("--coverage", type=Path, default=ROOT / "output" / "clv_panel_market_coverage.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "output" / "CLV_PANEL_AUDIT.md")
    args = parser.parse_args()
    summary, coverage, report = audit(
        manifest_path=args.manifest,
        ledger_path=args.ledger,
        checkpoint_path=args.checkpoint,
        master_path=args.master,
    )
    atomic_write_json(args.summary, summary)
    atomic_write_csv(args.coverage, coverage)
    atomic_write_text(args.report, report)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
