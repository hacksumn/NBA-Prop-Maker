#!/usr/bin/env python3
"""
Production Holdout Audit: Compare Model vs. Always-UNDER Baseline
==================================================================

This script evaluates prop performance from picks_history.csv and compares
model selection accuracy vs. the always-UNDER baseline.

Outputs:
  - output/production_holdout_audit.csv  (results by prop)
  - output/production_holdout_audit.json (summary stats)

Usage:
    python scripts/audit_production_holdout_simple.py
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.parent.resolve()

CONFIG = {
    'data_dir': SCRIPT_DIR / 'data',
    'output_dir': SCRIPT_DIR / 'output',
}

ALWAYS_UNDER_THRESHOLD = 0.52


def run_audit(picks_csv: Optional[Path] = None,
              output_dir: Optional[Path] = None,
              min_picks: int = 50) -> tuple:
    """Run audit on historical picks."""
    print("\n" + "="*70)
    print("PRODUCTION AUDIT: Model vs Always-UNDER (from picks_history.csv)")
    print("="*70 + "\n")
    
    # Load picks history
    picks_csv = Path(picks_csv) if picks_csv is not None else CONFIG['output_dir'] / 'picks_history.csv'
    output_dir = Path(output_dir) if output_dir is not None else CONFIG['output_dir']
    if not picks_csv.exists():
        print(f"ERROR: {picks_csv} not found")
        return None, None
    
    picks = pd.read_csv(picks_csv, low_memory=False)
    required_columns = {'game_date', 'prop', 'result', 'direction'}
    missing_columns = sorted(required_columns - set(picks.columns))
    if missing_columns:
        raise ValueError(
            "picks_history schema mismatch; missing required columns: "
            + ", ".join(missing_columns)
        )
    if picks.empty:
        raise ValueError("picks_history contains no rows")

    parsed_dates = pd.to_datetime(picks['game_date'], errors='coerce')
    if parsed_dates.isna().any():
        raise ValueError(
            f"picks_history contains {int(parsed_dates.isna().sum())} invalid game_date values"
        )
    picks['game_date'] = parsed_dates
    picks['result'] = picks['result'].fillna('').astype(str).str.strip().str.upper()
    picks['direction'] = picks['direction'].fillna('').astype(str).str.strip().str.upper()
    
    print(f"Loaded {len(picks):,} picks from {picks_csv}")
    print(f"Date range: {picks['game_date'].min().strftime('%Y-%m-%d')} to {picks['game_date'].max().strftime('%Y-%m-%d')}\n")
    
    # Only binary decisions belong in an accuracy comparison. PUSH, DNP, and
    # pending rows are neither wins nor losses and must not be scored as losses.
    graded = picks[picks['result'].isin({'WIN', 'LOSS'})].copy()
    print(f"Graded picks: {len(graded):,} ({len(graded)/len(picks)*100:.1f}%)\n")
    
    if len(graded) == 0:
        print("ERROR: No graded picks found")
        raise ValueError("No WIN/LOSS rows found in picks_history")

    invalid_directions = ~graded['direction'].isin({'OVER', 'UNDER'})
    if invalid_directions.any():
        examples = graded.loc[invalid_directions, 'direction'].drop_duplicates().head(5).tolist()
        raise ValueError(
            f"{int(invalid_directions.sum())} graded rows have invalid direction values: {examples}"
        )
    
    # Evaluate each prop
    results_rows = []
    
    for prop in sorted(graded['prop'].unique()):
        prop_picks = graded[graded['prop'] == prop].copy()
        n_picks = len(prop_picks)
        
        if n_picks < min_picks:
            print(f"{prop.upper()}: only {n_picks} graded picks (need {min_picks}+), skipping")
            continue
        
        # Model decision: OVER/UNDER based on pick
        # result: 'WIN' = model was correct, 'LOSS' = model was wrong
        # For each pick, model won if result=='WIN'
        model_correct = (prop_picks['result'] == 'WIN').astype(int)
        model_accuracy = model_correct.mean()
        
        # Always-UNDER baseline: assume all picks should have been UNDER
        # Count how many would have won if we always picked UNDER
        # Note: this assumes picks labeled 'OVER' should have been 'UNDER' to win
        # We need to infer which picks were OVER vs UNDER from the pick direction
        
        is_over = prop_picks['direction'].eq('OVER')
        
        # Always-UNDER would win if:
        # - The pick was UNDER and the model was correct, OR
        # - The pick was OVER and the model was wrong (opposite would have been right)
        always_under_correct = (~is_over & (model_correct == 1)) | (is_over & (model_correct == 0))
        always_under_accuracy = always_under_correct.mean()
        
        # Breakdown
        over_picks = is_over.sum()
        under_picks = (~is_over).sum()
        over_model_acc = model_correct[is_over].mean() if over_picks > 0 else np.nan
        under_model_acc = model_correct[~is_over].mean() if under_picks > 0 else np.nan
        
        # Edge calculation
        model_beats_baseline = model_accuracy > always_under_accuracy
        margin = model_accuracy - always_under_accuracy
        status = "PASS" if model_beats_baseline else "FAIL"
        
        # Check if even baseline is weak
        baseline_status = "OK" if always_under_accuracy >= ALWAYS_UNDER_THRESHOLD else f"WEAK"
        
        print(f"{prop.upper()}:")
        print(f"  Picks: {n_picks} (OVER {over_picks} | UNDER {under_picks})")
        print(f"  Model accuracy: {model_accuracy:.1%} (OVER {over_model_acc:.1%} | UNDER {under_model_acc:.1%})")
        print(f"  Always-UNDER:   {always_under_accuracy:.1%} [{baseline_status}]")
        print(f"  Edge vs baseline: {margin:+.1%} [{status}]\n")
        
        result_row = {
            'prop': prop.upper(),
            'graded_picks': n_picks,
            'over_picks': int(over_picks),
            'under_picks': int(under_picks),
            'model_accuracy': round(model_accuracy * 100, 1),
            'model_over_accuracy': round(over_model_acc * 100, 1) if not np.isnan(over_model_acc) else None,
            'model_under_accuracy': round(under_model_acc * 100, 1) if not np.isnan(under_model_acc) else None,
            'always_under_accuracy': round(always_under_accuracy * 100, 1),
            'edge_vs_always_under': round(margin * 100, 1),
            'beats_always_under': bool(model_beats_baseline),
            'baseline_status': baseline_status,
            'status': status,
        }
        results_rows.append(result_row)
    
    results_df = pd.DataFrame(results_rows)
    if results_df.empty:
        raise ValueError(
            f"No prop has the required minimum of {min_picks} graded WIN/LOSS rows"
        )
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / 'production_holdout_audit.csv'
    output_json = output_dir / 'production_holdout_audit.json'
    
    print("="*70)
    print("RESULTS")
    print("="*70 + "\n")
    print(results_df.to_string(index=False))
    print()
    
    results_df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")
    
    # Identify props to gate
    props_losing = results_df[~results_df['beats_always_under']]['prop'].tolist()
    props_weak_baseline = results_df[results_df['baseline_status'] == 'WEAK']['prop'].tolist()
    
    json_results = (
        results_df.astype(object)
        .where(pd.notna(results_df), None)
        .to_dict(orient='records')
    )
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_picks_audited': len(graded),
        'props_evaluated': len(results_df),
        'always_under_threshold': ALWAYS_UNDER_THRESHOLD,
        'props_losing_to_always_under': props_losing,
        'props_with_weak_baseline': props_weak_baseline,
        'recommendation': f"Gate/disable: {props_losing + props_weak_baseline}",
        'results': json_results,
    }
    
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(f"Saved to {output_json}")
    
    print("\n" + "="*70)
    print("ACTION ITEMS")
    print("="*70)
    print(f"\nProps that LOSE to always-UNDER: {props_losing}")
    print(f"Props with weak baseline (<{ALWAYS_UNDER_THRESHOLD:.1%}): {props_weak_baseline}")
    print("\nRecommendation: Update CONFIG['excluded_pick_props'] in nba_props.py")
    
    return results_df, summary


if __name__ == '__main__':
    try:
        results_df, summary = run_audit()
        if results_df is not None:
            print("\nAudit complete")
        else:
            print("\nAudit failed: no data")
            exit(1)
    except Exception as e:
        print(f"\nAudit failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
