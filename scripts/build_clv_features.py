"""Materialize pre-tip consensus, price, movement, dispersion, and PP-gap features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.audit_clv_snapshot_panel import load_paid_panel
    from scripts.execute_clv_snapshot_manifest import ROOT, atomic_write_csv, atomic_write_json, normalize_name
except ModuleNotFoundError:
    from audit_clv_snapshot_panel import load_paid_panel
    from execute_clv_snapshot_manifest import ROOT, atomic_write_csv, atomic_write_json, normalize_name


PROP_FORMULAS = {
    "player_points": lambda d: d["pts"],
    "player_rebounds": lambda d: d["trb"],
    "player_assists": lambda d: d["ast"],
    "player_steals": lambda d: d["stl"],
    "player_blocks": lambda d: d["blk"],
    "player_turnovers": lambda d: d["tov"],
    "player_points_assists": lambda d: d["pts"] + d["ast"],
    "player_points_rebounds": lambda d: d["pts"] + d["trb"],
    "player_points_rebounds_assists": lambda d: d["pts"] + d["trb"] + d["ast"],
}


def american_implied(series: pd.Series) -> pd.Series:
    odds = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0)),
        index=series.index,
        dtype=float,
    )


def parse_local_snapshot(value: object) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(stamp):
        return pd.NaT
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="shift_forward")
    return stamp.tz_convert("UTC")


def choose_main_book_lines(paid: pd.DataFrame) -> pd.DataFrame:
    frame = paid.copy()
    over = american_implied(frame["over_price"])
    under = american_implied(frame["under_price"])
    frame["p_over_no_vig"] = over / (over + under)
    frame["main_line_score"] = -(frame["p_over_no_vig"] - 0.5).abs()
    frame["main_line_score"] = frame["main_line_score"].fillna(-1.0)
    keys = ["manifest_key", "player_norm", "prop", "bookmaker"]
    frame = frame.sort_values([*keys, "main_line_score", "line"], kind="stable")
    return frame.drop_duplicates(keys, keep="last").reset_index(drop=True)


def attach_latest_prizepicks(consensus: pd.DataFrame, master_path: Path) -> pd.DataFrame:
    master = pd.read_csv(master_path, low_memory=False)
    source = master["source"].fillna("").astype(str).str.lower()
    pp = master[source.str.contains("prizepicks")].copy()
    pp["game_date"] = pd.to_datetime(pp["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    pp["player_norm"] = pp["player"].map(normalize_name)
    pp["pp_snapshot_ts"] = pp["snapshot_ts"].map(parse_local_snapshot)
    pp["pp_line"] = pd.to_numeric(pp["line"], errors="coerce")
    pp = pp.dropna(subset=["game_date", "player_norm", "prop", "pp_snapshot_ts", "pp_line"])
    pp = pp.sort_values("pp_snapshot_ts").drop_duplicates(
        ["game_date", "player_norm", "prop", "pp_snapshot_ts"], keep="last"
    )

    left = consensus.copy()
    left["request_ts"] = pd.to_datetime(left["requested_snapshot_ts"], utc=True)
    left = left.sort_values("request_ts")
    right = pp[["game_date", "player_norm", "prop", "pp_snapshot_ts", "pp_line"]].sort_values("pp_snapshot_ts")
    matched = pd.merge_asof(
        left,
        right,
        left_on="request_ts",
        right_on="pp_snapshot_ts",
        by=["game_date", "player_norm", "prop"],
        direction="backward",
        allow_exact_matches=True,
    )
    matched["pp_snapshot_age_minutes"] = (
        matched["request_ts"] - matched["pp_snapshot_ts"]
    ).dt.total_seconds() / 60.0
    matched["pp_minus_consensus"] = matched["pp_line"] - matched["consensus_line"]
    return matched


def _column_or_nan(frame: pd.DataFrame, name: str) -> pd.Series:
    return frame[name] if name in frame.columns else pd.Series(np.nan, index=frame.index, dtype=float)


def build(
    manifest_path: Path | None = None,
    ledger_path: Path | None = None,
    checkpoint_path: Path | None = None,
    master_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    master_path = master_path or ROOT / "data" / "historical_lines.csv"
    paid, _, _, checkpoint = load_paid_panel(
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        checkpoint_path=checkpoint_path,
        master_path=master_path,
    )
    main = choose_main_book_lines(paid)
    group = [
        "manifest_key", "nba_game_id", "game_date", "snapshot_kind",
        "requested_snapshot_ts", "player_norm", "prop",
    ]
    consensus = main.groupby(group, as_index=False).agg(
        player=("player", "first"),
        consensus_line=("line", "median"),
        consensus_p_over=("p_over_no_vig", "median"),
        book_count=("bookmaker", "nunique"),
        line_min=("line", "min"),
        line_max=("line", "max"),
        line_std=("line", lambda s: float(s.std(ddof=0))),
        p_over_std=("p_over_no_vig", lambda s: float(s.std(ddof=0))),
        paired_price_share=("p_over_no_vig", lambda s: float(s.notna().mean())),
    )
    consensus["line_range"] = consensus["line_max"] - consensus["line_min"]
    consensus = attach_latest_prizepicks(consensus, master_path)

    index = ["nba_game_id", "game_date", "player_norm", "prop", "player"]
    values = [
        "consensus_line", "consensus_p_over", "book_count", "line_range", "line_std",
        "p_over_std", "paired_price_share", "pp_line", "pp_snapshot_age_minutes",
        "pp_minus_consensus",
    ]
    wide = consensus.pivot_table(index=index, columns="snapshot_kind", values=values, aggfunc="first")
    wide.columns = [f"{metric}_{kind}" for metric, kind in wide.columns]
    wide = wide.reset_index()
    for kind in ["decision", "t_minus_60", "t_minus_5"]:
        line_col = f"consensus_line_{kind}"
        wide[f"available_{kind}"] = wide.get(line_col, pd.Series(index=wide.index, dtype=float)).notna()
    wide["line_move_60_to_5"] = _column_or_nan(wide, "consensus_line_t_minus_5") - _column_or_nan(wide, "consensus_line_t_minus_60")
    wide["p_over_move_60_to_5"] = _column_or_nan(wide, "consensus_p_over_t_minus_5") - _column_or_nan(wide, "consensus_p_over_t_minus_60")
    wide["dispersion_move_60_to_5"] = _column_or_nan(wide, "line_std_t_minus_5") - _column_or_nan(wide, "line_std_t_minus_60")
    wide["market_withdrawn_by_t5"] = wide["available_t_minus_60"] & ~wide["available_t_minus_5"]
    wide["line_move_decision_to_5"] = _column_or_nan(wide, "consensus_line_t_minus_5") - _column_or_nan(wide, "consensus_line_decision")
    wide["p_over_move_decision_to_5"] = _column_or_nan(wide, "consensus_p_over_t_minus_5") - _column_or_nan(wide, "consensus_p_over_decision")

    box = pd.read_csv(
        ROOT / "data" / "nba_data.csv",
        usecols=["player", "game_date", "mp", "pts", "trb", "ast", "stl", "blk", "tov"],
        low_memory=False,
    )
    box["game_date"] = pd.to_datetime(box["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    box["player_norm"] = box["player"].map(normalize_name)
    for prop, formula in PROP_FORMULAS.items():
        box[prop] = formula(box)
    box = box.drop_duplicates(["game_date", "player_norm"], keep="first")
    long_actual = box.melt(
        id_vars=["game_date", "player_norm", "mp"],
        value_vars=list(PROP_FORMULAS),
        var_name="prop",
        value_name="actual",
    )
    wide = wide.merge(long_actual, on=["game_date", "player_norm", "prop"], how="left", validate="many_to_one")
    for source_name in ["consensus", "pp"]:
        for kind in ["decision", "t_minus_60", "t_minus_5"]:
            line_name = f"{source_name}_line_{kind}" if source_name == "consensus" else f"pp_line_{kind}"
            if line_name not in wide.columns:
                continue
            stem = f"{source_name}_{kind}"
            valid = wide["actual"].notna() & wide[line_name].notna()
            wide[f"under_win_{stem}"] = np.where(valid, wide["actual"] < wide[line_name], np.nan)
            wide[f"push_{stem}"] = np.where(valid, wide["actual"] == wide[line_name], np.nan)

    summary = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z"),
        "credits_used": int(checkpoint["credits_used"]),
        "feature_rows": int(len(wide)),
        "games": int(wide["nba_game_id"].nunique()),
        "player_prop_date_keys": int(wide[["game_date", "player_norm", "prop"]].drop_duplicates().shape[0]),
        "actual_outcome_rows": int(wide["actual"].notna().sum()),
        "t60_to_t5_line_move_rows": int(wide["line_move_60_to_5"].notna().sum()),
        "decision_to_t5_line_move_rows": int(wide["line_move_decision_to_5"].notna().sum()),
        "pp_gap_t60_rows": int(wide.get("pp_minus_consensus_t_minus_60", pd.Series(dtype=float)).notna().sum()),
        "pp_gap_decision_rows": int(wide.get("pp_minus_consensus_decision", pd.Series(dtype=float)).notna().sum()),
        "market_withdrawal_rows": int(wide["market_withdrawn_by_t5"].sum()),
        "pre_tip_feature_columns": [column for column in wide.columns if column not in {"actual", "mp"} and not column.startswith(("under_win_", "push_"))],
        "label_columns": [column for column in wide.columns if column in {"actual", "mp"} or column.startswith(("under_win_", "push_"))],
    }
    return wide, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "clv_snapshot_manifest_2026_live.csv")
    parser.add_argument("--ledger", type=Path, default=ROOT / "data" / "clv_snapshot_execution_ledger.csv")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data" / "clv_snapshot_execution_checkpoint.json")
    parser.add_argument("--master", type=Path, default=ROOT / "data" / "historical_lines.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "clv_features_2026_live.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "output" / "clv_features_summary.json")
    args = parser.parse_args()
    features, summary = build(
        manifest_path=args.manifest,
        ledger_path=args.ledger,
        checkpoint_path=args.checkpoint,
        master_path=args.master,
    )
    atomic_write_csv(args.output, features)
    atomic_write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
