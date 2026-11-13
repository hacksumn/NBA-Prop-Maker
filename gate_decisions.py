"""Immutable run evidence for live gate evaluation, independent of pick histories."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scripts.build_release_manifest import PRODUCTION_MODEL_PATTERNS


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def start_run(config, policy):
    """Fingerprint effective settings and active model files before scoring."""
    models = Path(config["models_dir"])
    patterns = (*PRODUCTION_MODEL_PATTERNS, "xgb_*.pkl", "clf_*.pkl", "bias.json",
                "minutes_model.pkl", "sigma_context_analysis.json", "availability_minutes_policy.json")
    files = {p for pattern in patterns for p in models.glob(pattern) if p.is_file()}
    availability_policy = models / "availability_minutes_policy.json"
    if availability_policy.exists():
        policy_settings = json.loads(availability_policy.read_text(encoding="utf-8"))
        availability_model = models / "availability_minutes_model.pkl"
        if policy_settings.get("enabled_for_live") and availability_model.exists():
            files.add(availability_model)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()
                if "key" not in k.lower() and "token" not in k.lower()}
    settings["live_sportsbook_key"] = config.get("live_sportsbook_key")
    policy_json = json.dumps({"config": settings, "policy": policy}, sort_keys=True)
    return {"schema_version": 1, "run_id": uuid.uuid4().hex,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "policy_sha256": hashlib.sha256(policy_json.encode()).hexdigest(),
            "config": settings, "policy": policy,
            "model_sha256": {p.name: _digest(p) for p in sorted(files)},
            "source_sha256": _digest(Path(__file__).with_name("nba_props.py"))}


def record_generation_rejection(events, row, prop, reason):
    """Retain the first failed primary-generation rule and its scored inputs."""
    if events is None:
        return
    event = {"stage": "generation", "reason": reason, "player": row.get("player"),
             "team": row.get("team"), "prop": prop.upper()}
    for name in ["line", "pred", "market_edge_pred", "meta_prob", "p_over_cal", "sigma", "l5", "l10"]:
        event[name] = row.get(f"{prop}_{name}")
    line, prediction = event["line"], event["pred"]
    if pd.notna(line) and pd.notna(prediction):
        edge = float(prediction) - float(line)
        residual = event["market_edge_pred"]
        combined = .45 * edge + .55 * float(residual) if pd.notna(residual) else edge
        event["direction"] = "OVER" if combined > 0 else "UNDER"
    events.append(event)


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp, Path)):
        return str(value)
    return value


def save_run(run, output_dir, slate_date, generation_events, sanity_events,
             candidates, published, market_lines, injury_report=None):
    """Write one atomic run receipt after successful publication output writes."""
    receipt = {**run, "slate_date": slate_date,
               "publication_saved_at": datetime.now(timezone.utc).isoformat(),
               "generation_rejections": generation_events,
               "sanity_rejections": sanity_events,
               "final_candidates": candidates.to_dict("records"),
               "published": published.to_dict("records"),
               "market_offers": market_lines.to_dict("records"),
               "injury_report": injury_report,
               "coverage": {"generation": "first failure for primary-model path; fallback paths recorded only if final candidates",
                            "sanity": "first failure per generated candidate",
                            "final": "all final accuracy flags and exact publication rows"}}
    directory = Path(output_dir) / "gate_runs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slate_date}_{run['run_id']}.json"
    if path.exists():
        raise FileExistsError(f"Gate run already saved: {path.name}")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_clean(receipt), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)
    return path
