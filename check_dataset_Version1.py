import pandas as pd
import numpy as np
import sys

CSV = "model_data_filtered.csv"

LEAKY_COLS = {
    'pts','trb','ast','mp','fg','fga','fg_pct','3p','3pa','3p_pct',
    'ft','fta','ft_pct','orb','drb','stl','blk','tov','gmsc','res','won'
}

TARGETS = ['pts','trb','ast']

def main():
    try:
        df = pd.read_csv(CSV, parse_dates=['game_date'])
    except Exception as e:
        print(f"ERROR: could not read '{CSV}': {e}")
        sys.exit(1)

    print(f"\nLoaded: {CSV}")
    print(f"Rows: {len(df):,}")
    print(f"Cols: {len(df.columns)}\n")

    print("Columns:")
    print(df.columns.tolist())

    print("\nSample (first 3 rows, transposed):")
    with pd.option_context('display.max_rows', 200):
        print(df.head(3).T)

    # Numeric columns and leaky intersection
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    found_leaky = sorted(set(numeric) & LEAKY_COLS)
    print(f"\nLeaky numeric columns found (should be empty): {found_leaky}")

    # Correlation scan per target
    print("\nTop correlations per target (top 10). Flags > 0.85")
    for t in TARGETS:
        if t not in df.columns:
            print(f"  - target '{t}' not in dataset, skipping")
            continue
        cols = [c for c in numeric if c != t]
        mat = df[[t] + cols].dropna()
        if mat.shape[0] < 10:
            print(f"  - not enough rows with non-nulls for target '{t}' to compute correlations")
            continue
        corrs = mat.corr()[t].drop(t).abs().sort_values(ascending=False)
        print(f"\nTarget: {t}")
        print(corrs.head(10).to_string())
        high = corrs[corrs > 0.85]
        if not high.empty:
            print("  !!! Highly correlated features (>0.85):")
            for f,v in high.items():
                print(f"    - {f}: {v:.3f}")

    # Rolling-feature NaN sanity for first games
    if 'games_played' in df.columns:
        first_games = df[df['games_played'] == 0]
        print(f"\nFirst-game rows (games_played==0): {len(first_games):,}")
        if 'pts_l5' in df.columns:
            n_null = first_games['pts_l5'].isna().sum()
            print(f"First games with NaN pts_l5: {n_null}/{len(first_games):,}")
        else:
            print("No pts_l5 column found to check first-game NaNs.")
    else:
        print("\nNo 'games_played' column present; cannot run first-game sanity check.")

    # Duplicates check
    if set(['player','game_date','tm']).issubset(df.columns):
        dup_count = df.duplicated(subset=['player','game_date','tm']).sum()
        print(f"\nDuplicate rows (player+game_date+tm): {dup_count}")
    else:
        print("\nColumns for duplicate check (player, game_date, tm) not all present.")

    # Basic null counts for key columns
    key_cols = ['pts','trb','ast','pts_l10','trb_l10','ast_l10']
    existing = [c for c in key_cols if c in df.columns]
    print("\nNull counts for key columns:")
    for c in existing:
        print(f"  {c}: {df[c].isna().sum():,} nulls / {len(df):,} rows")

    # Show rows where any leaky column is non-null in numeric features (strong debug)
    if found_leaky:
        print("\nSample rows where leaky numeric columns are present (first 5 rows):")
        cols_to_show = ['player','game_date'] + found_leaky
        cols_to_show = [c for c in cols_to_show if c in df.columns]
        print(df[cols_to_show].head(5).to_string(index=False))
    else:
        print("\nNo canonical leaky numeric columns found in numeric columns set.")

    print("\nFinished checks. Paste this output back if you want me to interpret results or produce fixes.")

if __name__ == "__main__":
    main()