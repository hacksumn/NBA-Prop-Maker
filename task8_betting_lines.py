"""
Task 8: Compare Predictions to Real Betting Lines (IMPROVED OUTPUT)
Organizes recommendations by prop type for easier betting

Usage:
    python task8_betting_lines.py --input real_lines.csv
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import warnings
from pathlib import Path

# Suppress sklearn feature name warnings (harmless)
warnings.filterwarnings('ignore', message='X does not have valid feature names')


# Prop display names and order
PROP_ORDER = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pr', 'pa', 'pra']
PROP_NAMES = {
    'pts': 'POINTS',
    'trb': 'REBOUNDS', 
    'ast': 'ASSISTS',
    'stl': 'STEALS',
    'blk': 'BLOCKS',
    'tov': 'TURNOVERS',
    'pra': 'PTS+REB+AST',
    'pr': 'PTS+REB',
    'pa': 'PTS+AST'
}


def load_models():
    """Load all trained models."""
    models = {}
    for prop in PROP_ORDER:
        path = f'model_{prop}.pkl'
        if Path(path).exists():
            with open(path, 'rb') as f:
                models[prop] = pickle.load(f)
    return models


def load_player_data():
    """Load historical player data for feature generation."""
    df = pd.read_csv('player_games_schedule.csv', parse_dates=['game_date'])
    return df


def get_player_features(player_name: str, df: pd.DataFrame, feature_cols: list) -> pd.Series:
    """Get most recent features for a player."""
    player_df = df[df['player'] == player_name].sort_values('game_date')
    
    if len(player_df) == 0:
        return None
    
    # Get last row's features
    last_row = player_df.iloc[-1]
    features = last_row[feature_cols] if all(c in last_row.index for c in feature_cols) else None
    
    return features


def predict_player(player_name: str, prop: str, models: dict, df: pd.DataFrame) -> float:
    """Generate prediction for a player prop."""
    if prop not in models:
        return None
    
    model_data = models[prop]
    feature_cols = model_data['features']
    
    features = get_player_features(player_name, df, feature_cols)
    if features is None:
        return None
    
    # Prepare features as DataFrame to preserve column names
    X = pd.DataFrame([features.values], columns=feature_cols)
    
    # Fill NaN with medians from training
    for col in feature_cols:
        if col in X.columns and pd.isna(X[col].iloc[0]):
            if col in model_data['medians'].index:
                X[col] = model_data['medians'][col]
            else:
                X[col] = 0  # Fallback
    
    # Final NaN check - replace any remaining with 0
    X = X.fillna(0)
    
    # Scale and predict
    X_scaled = model_data['scaler'].transform(X)
    pred = model_data['model'].predict(X_scaled)[0]
    return max(0, pred)  # Floor at 0


def calculate_edge(pred: float, line: float, over_odds: int, under_odds: int):
    """Calculate betting edge and recommendation."""
    diff = pred - line
    
    if abs(diff) < 0.5:
        return None, None, 0, "NO EDGE"
    
    if diff > 0:  # Predict OVER
        direction = "OVER"
        odds = over_odds
    else:  # Predict UNDER
        direction = "UNDER"
        odds = under_odds
        diff = abs(diff)
    
    # Strength rating
    if diff >= 3.0:
        strength = 3
        label = "STRONG"
    elif diff >= 2.0:
        strength = 2
        label = "MODERATE"
    elif diff >= 1.5:
        strength = 1
        label = "SLIGHT"
    else:
        strength = 0
        label = "NO BET"
    
    return direction, odds, strength, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='real_lines.csv', help='Lines CSV file')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Task 8: Compare Predictions to Real Betting Lines")
    print("=" * 70)
    
    # Load models
    models = load_models()
    print(f"Loaded {len(models)} models: {', '.join(models.keys())}")
    
    # Load player data
    print("Loading historical data...")
    df = load_player_data()
    print(f"  Loaded {len(df):,} rows")
    
    # Load lines
    lines_df = pd.read_csv(args.input)
    players = lines_df['player'].unique()
    print(f"Loaded lines for {len(players)} players from {args.input}")
    
    # Generate all predictions
    all_bets = []
    
    for player in players:
        player_lines = lines_df[lines_df['player'] == player]
        
        for _, row in player_lines.iterrows():
            prop = row['prop']
            if prop not in models:
                continue
            
            pred = predict_player(player, prop, models, df)
            if pred is None:
                continue
            
            line = row['line']
            over_odds = row['over_odds']
            under_odds = row['under_odds']
            
            direction, odds, strength, label = calculate_edge(pred, line, over_odds, under_odds)
            
            if strength >= 1:  # At least slight edge
                all_bets.append({
                    'player': player,
                    'prop': prop,
                    'pred': pred,
                    'line': line,
                    'direction': direction,
                    'odds': odds,
                    'edge': abs(pred - line),
                    'strength': strength,
                    'label': label
                })
    
    # Convert to DataFrame
    bets_df = pd.DataFrame(all_bets)
    
    if len(bets_df) == 0:
        print("\nNo betting opportunities found.")
        return
    
    # Print organized by prop type
    print("\n" + "=" * 70)
    print("BETTING RECOMMENDATIONS BY PROP TYPE")
    print("=" * 70)
    
    total_strong = 0
    total_moderate = 0
    total_slight = 0
    
    for prop in PROP_ORDER:
        prop_bets = bets_df[bets_df['prop'] == prop].sort_values('edge', ascending=False)
        
        if len(prop_bets) == 0:
            continue
        
        strong = len(prop_bets[prop_bets['strength'] == 3])
        moderate = len(prop_bets[prop_bets['strength'] == 2])
        slight = len(prop_bets[prop_bets['strength'] == 1])
        
        total_strong += strong
        total_moderate += moderate
        total_slight += slight
        
        print(f"\n{'=' * 70}")
        print(f"  {PROP_NAMES[prop]} ({len(prop_bets)} bets: {strong} strong, {moderate} moderate, {slight} slight)")
        print(f"{'=' * 70}")
        
        # Strong bets first
        strong_bets = prop_bets[prop_bets['strength'] == 3]
        if len(strong_bets) > 0:
            print(f"\n  *** STRONG BETS ***")
            for _, bet in strong_bets.head(10).iterrows():
                odds_str = f"+{bet['odds']}" if bet['odds'] > 0 else str(int(bet['odds']))
                print(f"    {bet['player']:25} {bet['direction']:5} {bet['line']:5.1f} @ {odds_str:>5}")
                print(f"      Pred: {bet['pred']:.1f}, Edge: {bet['edge']:.1f} pts")
        
        # Moderate bets
        mod_bets = prop_bets[prop_bets['strength'] == 2]
        if len(mod_bets) > 0:
            print(f"\n  ** MODERATE BETS **")
            for _, bet in mod_bets.head(10).iterrows():
                odds_str = f"+{bet['odds']}" if bet['odds'] > 0 else str(int(bet['odds']))
                print(f"    {bet['player']:25} {bet['direction']:5} {bet['line']:5.1f} @ {odds_str:>5}")
                print(f"      Pred: {bet['pred']:.1f}, Edge: {bet['edge']:.1f} pts")
        
        # Slight bets (just count, don't list all)
        slight_bets = prop_bets[prop_bets['strength'] == 1]
        if len(slight_bets) > 0:
            print(f"\n  * SLIGHT EDGE: {len(slight_bets)} bets (not listed)")
    
    # Summary by direction
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    overs = len(bets_df[bets_df['direction'] == 'OVER'])
    unders = len(bets_df[bets_df['direction'] == 'UNDER'])
    
    print(f"\nTotal bets: {len(bets_df)}")
    print(f"  Strong:   {total_strong}")
    print(f"  Moderate: {total_moderate}")
    print(f"  Slight:   {total_slight}")
    print(f"\nDirection split:")
    print(f"  OVERS:  {overs} ({overs/len(bets_df)*100:.1f}%)")
    print(f"  UNDERS: {unders} ({unders/len(bets_df)*100:.1f}%)")
    
    # Top 10 overall
    print(f"\n{'=' * 70}")
    print("TOP 10 HIGHEST EDGE BETS (ALL PROPS)")
    print(f"{'=' * 70}")
    
    top10 = bets_df.sort_values('edge', ascending=False).head(10)
    for i, (_, bet) in enumerate(top10.iterrows(), 1):
        odds_str = f"+{bet['odds']}" if bet['odds'] > 0 else str(int(bet['odds']))
        print(f"\n  {i}. {bet['player']} - {PROP_NAMES[bet['prop']]}")
        print(f"     {bet['direction']} {bet['line']} @ {odds_str}")
        print(f"     Prediction: {bet['pred']:.1f}, Edge: {bet['edge']:.1f} pts")
    
    # Save to CSV for tracking
    bets_df['date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
    bets_df['actual'] = ''  # Fill in after games
    bets_df['result'] = ''  # WIN/LOSS
    bets_df['units_bet'] = 1.0  # Default 1 unit
    bets_df['profit'] = ''  # Calculate after
    
    # Reorder columns for Excel tracking
    export_cols = ['date', 'player', 'prop', 'direction', 'line', 'pred', 'edge', 
                   'odds', 'strength', 'label', 'actual', 'result', 'units_bet', 'profit']
    bets_df = bets_df[export_cols]
    
    bets_df.to_csv('betting_recommendations.csv', index=False)
    print(f"\n\nSaved all recommendations to: betting_recommendations.csv")
    print("  - Open in Excel to track results")
    print("  - Fill in 'actual' column after games")
    print("  - Mark 'result' as WIN or LOSS")
    print("  - profit = units_bet * (odds/100) for wins, -units_bet for losses")
    
    print("\n" + "=" * 70)
    print("DISCLAIMER: For educational purposes only.")
    print("Gambling involves risk. Never bet more than you can afford to lose.")
    print("=" * 70)


if __name__ == "__main__":
    main()
