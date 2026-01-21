import json
import pandas as pd

CSV = "model_data_filtered.csv"

ID_COLS = {'player', 'tm', 'opp', 'game_date', 'game_id', 'matchup_id', 'matchup_raw', 'res'}
CURRENT_GAME = {
    'pts','trb','ast','stl','blk','tov','pf','mp','fg','fga','fg_pct','3p','3pa','3p_pct',
    'ft','fta','ft_pct','orb','drb','gmsc','won',
    # combo targets (if present)
    'pra','pr','pa','pra_l10','pr_l10','pa_l10'
}

df = pd.read_csv(CSV, parse_dates=['game_date'])

numeric_cols = set(df.select_dtypes(include=['number']).columns.tolist())
feature_set = numeric_cols - ID_COLS - CURRENT_GAME
feature_cols = sorted(list(feature_set))

print("Feature count:", len(feature_cols))
print("First 50 feature columns:", feature_cols[:50])
print("\nAny targets present in numeric columns (expected):", sorted(list(numeric_cols & {'pts','trb','ast'})))

intersection = sorted(list(feature_set & CURRENT_GAME))
print("\nIntersection with CURRENT_GAME (should be empty):", intersection)

# Save feature list for reproducibility
with open("feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)
print("\nSaved feature_cols.json")