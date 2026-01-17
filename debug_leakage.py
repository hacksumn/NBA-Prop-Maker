"""
Debug script to identify data leakage in predictions
"""
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Load historical data
print("Loading data...")
df = pd.read_csv('player_games_schedule.csv', parse_dates=['game_date'])
df = df.sort_values(['player', 'game_date'])

# Load a model
print("\nLoading PTS model...")
with open('model_pts.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
medians = model_data['medians']
features = model_data['features']

print(f"Model uses {len(features)} features")
print(f"\nFirst 20 features: {features[:20]}")

# Check if current game stats are in features
current_game_stats = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'mp', 'fg', 'fga', 'fg_pct']
leaky_features = [f for f in features if f in current_game_stats]

if leaky_features:
    print(f"\n⚠️ LEAKAGE DETECTED! Current game stats in features:")
    for f in leaky_features:
        print(f"  - {f}")
else:
    print("\n✓ No obvious current game stats in features")

# Check a specific player's prediction
print("\n" + "="*60)
print("TESTING PREDICTION FOR TREY MURPHY III")
print("="*60)

player = 'Trey Murphy III'
player_df = df[df['player'] == player].sort_values('game_date')

# Get the last game (this is what would be used for prediction)
latest = player_df.iloc[-1]
print(f"\nLatest game date: {latest['game_date']}")
print(f"Actual PTS in that game: {latest['pts']}")
print(f"pts_l10 (line proxy): {latest['pts_l10']}")
print(f"pts_l5: {latest['pts_l5']}")

# Extract features for this row
X_raw = latest[features].to_frame().T
print(f"\nFeature vector shape: {X_raw.shape}")

# Check for NaN values
nan_count = X_raw.isna().sum().sum()
print(f"NaN values in features: {nan_count}")

# Fill NaN with medians
X_filled = X_raw.fillna(medians)

# Scale
X_scaled = scaler.transform(X_filled)

# Predict
pred = model.predict(X_scaled)[0]
print(f"\nPrediction: {pred:.1f}")
print(f"Line (pts_l10): {latest['pts_l10']:.1f}")
print(f"Edge: {abs(pred - latest['pts_l10']):.1f}")

# Now check what happens if we use the PREVIOUS game's features
# (which is what we SHOULD be using for prediction)
if len(player_df) >= 2:
    print("\n" + "="*60)
    print("CORRECT APPROACH: Using PREVIOUS game's features")
    print("="*60)
    
    previous = player_df.iloc[-2]
    print(f"\nPrevious game date: {previous['game_date']}")
    print(f"Previous game's pts_l10: {previous['pts_l10']}")
    print(f"Previous game's pts_l5: {previous['pts_l5']}")
    
    # These rolling features from the previous game are what should be used
    # to predict the NEXT game (which is the latest game)
    X_raw_prev = previous[features].to_frame().T
    X_filled_prev = X_raw_prev.fillna(medians)
    X_scaled_prev = scaler.transform(X_filled_prev)
    pred_prev = model.predict(X_scaled_prev)[0]
    
    print(f"\nPrediction using previous features: {pred_prev:.1f}")
    print(f"Actual outcome: {latest['pts']:.1f}")
    print(f"Error: {abs(pred_prev - latest['pts']):.1f}")
    print(f"Line (previous pts_l10): {previous['pts_l10']:.1f}")
    print(f"Edge: {abs(pred_prev - previous['pts_l10']):.1f}")

# Check if the issue is in how we're getting "latest" features
print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

print("\nThe issue is likely in task7_daily_predictions.py:")
print("When making predictions for a FUTURE game, we should use:")
print("  1. The player's CURRENT rolling features (from their last game)")
print("  2. These features already exclude that last game (due to shift(1))")
print("  3. So they represent the player's state ENTERING the next game")
print("\nHowever, the rolling features include the CURRENT game's stats")
print("because they were calculated AFTER the game was played.")
print("\nFor FUTURE predictions (games not yet played), we need to:")
print("  1. Use the most recent game's rolling features")
print("  2. These are already properly shifted and ready to use")
print("\nThe problem might be in how opponent features or schedule features")
print("are being computed, or there's contamination somewhere else.")
