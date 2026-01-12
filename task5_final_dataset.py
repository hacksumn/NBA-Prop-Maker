"""
Task 5: Remove Leaky Features & Create Final Dataset
NBA Player Props Betting System

CRITICAL: This script separates features (pre-tip-off only) from targets.
Any feature that uses information from the game being predicted = LEAKAGE.
"""

import pandas as pd
import numpy as np


def identify_feature_categories(df: pd.DataFrame) -> dict:
    """
    Categorize all columns into: identifiers, targets, valid features, and leaky columns.
    """
    all_cols = set(df.columns)
    
    # 1. Identifier columns (kept for tracking, not used as features)
    identifiers = {
        'player', 'tm', 'opp', 'game_date', 'game_id', 'matchup_id'
    }
    
    # 2. Target columns (what we're predicting)
    targets = {
        'pts', 'trb', 'ast'
    }
    
    # 3. Leaky columns (current game data - MUST BE REMOVED from features)
    leaky = {
        'mp',           # Current game minutes
        'fg', 'fga', 'fg_pct',    # Current game shooting
        '3p', '3pa', '3p_pct',    # Current game 3pt
        'ft', 'fta', 'ft_pct',    # Current game FT
        'orb', 'drb',   # Current game rebounds (components)
        'stl', 'blk', 'tov', 'pf',  # Current game stats
        'gmsc',         # Game score (derived from current game)
        'res', 'won',   # Game result (happens after tip-off)
        'is_home'       # We marked this as NaN anyway
    }
    
    # 4. Valid features (everything else - rolling averages, opponent stats, schedule)
    valid_features = all_cols - identifiers - targets - leaky
    
    # Verify we haven't missed anything
    accounted = identifiers | targets | leaky | valid_features
    unaccounted = all_cols - accounted
    
    return {
        'identifiers': identifiers & all_cols,  # Only keep if they exist
        'targets': targets & all_cols,
        'leaky': leaky & all_cols,
        'valid_features': valid_features,
        'unaccounted': unaccounted
    }


def validate_features_are_pre_tipoff(df: pd.DataFrame, feature_cols: list) -> bool:
    """
    Final validation that all features could be known before tip-off.
    
    Rules:
    1. No feature should have correlation > 0.85 with targets
    2. Feature names should indicate they're historical (_l5, _l10, _l20, _shifted, etc.)
    3. No raw current-game stats
    """
    print("\n" + "="*60)
    print("PRE-TIP-OFF FEATURE VALIDATION")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Check for suspiciously high correlations with targets
    print("\n[Test 1] Correlation check with targets:")
    targets = ['pts', 'trb', 'ast']
    
    suspicious_features = []
    for target in targets:
        if target not in df.columns:
            continue
        for feat in feature_cols:
            if feat in df.columns:
                corr = df[[target, feat]].dropna().corr().iloc[0, 1]
                if abs(corr) > 0.85:
                    suspicious_features.append((feat, target, corr))
                    print(f"  ⚠ HIGH CORRELATION: {feat} vs {target} = {corr:.4f}")
    
    if not suspicious_features:
        print(f"  ✓ PASS - No features with correlation > 0.85 to targets")
    else:
        print(f"  ✗ FAIL - {len(suspicious_features)} suspicious features found")
        all_passed = False
    
    # Test 2: Verify feature naming conventions suggest historical data
    print("\n[Test 2] Feature naming convention check:")
    historical_patterns = ['_l5', '_l10', '_l20', '_l3', '_std_', '_trend', 
                           '_allowed', '_rank', 'days_rest', 'is_b2b', 'games_',
                           'opp_', 'mp_avg', 'high_min', 'streak', 'player_days']
    
    non_historical = []
    for feat in feature_cols:
        if not any(pattern in feat for pattern in historical_patterns):
            # Check if it's a known valid feature
            known_valid = {'games_played'}
            if feat not in known_valid:
                non_historical.append(feat)
    
    if non_historical:
        print(f"  ⚠ Features without clear historical pattern: {non_historical}")
        print(f"    (Review these manually to confirm they're pre-tip-off)")
    else:
        print(f"  ✓ PASS - All features have historical naming patterns")
    
    # Test 3: Verify no current-game stat names in features
    print("\n[Test 3] Current-game stat name check:")
    current_game_names = ['pts', 'trb', 'ast', 'mp', 'fg', 'fga', 'ft', 'fta', 
                          '3p', '3pa', 'stl', 'blk', 'tov', 'orb', 'drb']
    
    exact_matches = [f for f in feature_cols if f in current_game_names]
    if exact_matches:
        print(f"  ✗ FAIL - Current game stats in features: {exact_matches}")
        all_passed = False
    else:
        print(f"  ✓ PASS - No current-game stat names in features")
    
    print("\n" + "="*60)
    if all_passed:
        print("STATUS: ALL FEATURES ARE PRE-TIP-OFF VALID ✓")
    else:
        print("STATUS: POTENTIAL LEAKAGE - FIX BEFORE MODELING")
    print("="*60)
    
    return all_passed


def create_modeling_dataset(df: pd.DataFrame, 
                            identifiers: set,
                            targets: set, 
                            valid_features: set) -> pd.DataFrame:
    """
    Create the final modeling dataset with only valid columns.
    """
    print("\nCreating modeling dataset...")
    
    # Select columns in specific order: identifiers, targets, features
    id_cols = sorted([c for c in identifiers if c in df.columns])
    target_cols = sorted([c for c in targets if c in df.columns])
    feature_cols = sorted([c for c in valid_features if c in df.columns])
    
    final_cols = id_cols + target_cols + feature_cols
    df_final = df[final_cols].copy()
    
    print(f"  Identifiers: {len(id_cols)}")
    print(f"  Targets: {len(target_cols)}")
    print(f"  Features: {len(feature_cols)}")
    
    return df_final, feature_cols


def filter_sufficient_history(df: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """
    Filter to only include player-games with sufficient history.
    
    Players with < min_games don't have reliable rolling averages.
    """
    print(f"\nFiltering to players with >= {min_games} games played...")
    
    before = len(df)
    df_filtered = df[df['games_played'] >= min_games].copy()
    after = len(df_filtered)
    
    print(f"  Before: {before:,} rows")
    print(f"  After: {after:,} rows")
    print(f"  Removed: {before - after:,} rows ({(before-after)/before*100:.1f}%)")
    
    return df_filtered


def main():
    # File paths
    input_path = 'player_games_schedule.csv'
    output_full = 'model_data_full.csv'       # All rows
    output_filtered = 'model_data_filtered.csv'  # Only rows with sufficient history
    
    print("Task 5: Remove Leaky Features & Create Final Dataset")
    print("-" * 40)
    
    # Load data
    print("Loading data with all features...")
    df = pd.read_csv(input_path, parse_dates=['game_date'])
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Identify column categories
    print("\nCategorizing columns...")
    categories = identify_feature_categories(df)
    
    print(f"\n  Identifiers: {len(categories['identifiers'])} columns")
    print(f"  Targets: {len(categories['targets'])} columns")
    print(f"  Valid Features: {len(categories['valid_features'])} columns")
    print(f"  Leaky (to remove): {len(categories['leaky'])} columns")
    
    if categories['unaccounted']:
        print(f"  ⚠ Unaccounted: {categories['unaccounted']}")
    
    # Show what's being removed
    print(f"\n[REMOVING LEAKY COLUMNS]:")
    for col in sorted(categories['leaky']):
        print(f"  - {col}")
    
    # Create modeling dataset
    df_model, feature_cols = create_modeling_dataset(
        df, 
        categories['identifiers'],
        categories['targets'],
        categories['valid_features']
    )
    
    # Validate features are pre-tip-off
    passed = validate_features_are_pre_tipoff(df_model, feature_cols)
    
    if not passed:
        print("\n⚠ WARNING: Potential leakage detected. Review before proceeding.")
    
    # Save full dataset
    df_model.to_csv(output_full, index=False)
    print(f"\n✓ Full dataset saved: {output_full}")
    print(f"  Rows: {len(df_model):,}")
    print(f"  Columns: {len(df_model.columns)}")
    
    # Create filtered version (sufficient history only)
    df_filtered = filter_sufficient_history(df_model, min_games=10)
    df_filtered.to_csv(output_filtered, index=False)
    print(f"\n✓ Filtered dataset saved: {output_filtered}")
    print(f"  Rows: {len(df_filtered):,}")
    print(f"  Columns: {len(df_filtered.columns)}")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL DATASET SUMMARY")
    print("="*60)
    print(f"\nTarget Variables (what we predict):")
    for t in ['pts', 'trb', 'ast']:
        if t in df_filtered.columns:
            print(f"  {t}: mean={df_filtered[t].mean():.1f}, std={df_filtered[t].std():.1f}")
    
    print(f"\nFeature Categories:")
    rolling_feats = [c for c in feature_cols if '_l5' in c or '_l10' in c or '_l20' in c]
    opp_feats = [c for c in feature_cols if c.startswith('opp_')]
    schedule_feats = [c for c in feature_cols if any(x in c for x in ['rest', 'b2b', 'games_', 'streak'])]
    
    print(f"  Rolling averages: {len(rolling_feats)}")
    print(f"  Opponent features: {len(opp_feats)}")
    print(f"  Schedule features: {len(schedule_feats)}")
    
    print(f"\n[SAMPLE - Ready for modeling]:")
    sample_cols = ['player', 'game_date', 'pts', 'pts_l10', 'opp_pts_allowed_l10', 'days_rest']
    print(df_filtered[sample_cols].head(10).to_string())
    
    print("\n" + "="*60)
    print("READY FOR TASK 6: Model Training")
    print("="*60)


if __name__ == "__main__":
    main()
