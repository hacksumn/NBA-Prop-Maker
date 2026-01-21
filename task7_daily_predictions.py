"""
Task 7: Daily Predictions Workflow
NBA Player Props Betting System

Run this BEFORE games start to lock in tonight's predictions.

Usage:
    python task7_daily_predictions.py

Requirements:
    - Trained models (model_*.pkl) in working directory
    - Updated player_games_schedule.csv with recent data
    - pip install: pandas numpy scikit-learn requests
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROPS_TO_PREDICT = ['pts', 'trb', 'ast', 'pra', 'pr', 'pa', 'stl', 'blk', 'tov']
MIN_EDGE_STRONG = 3.0      # Points edge for "strong" bet
MIN_EDGE_MODERATE = 2.0    # Points edge for "moderate" bet
OUTPUT_DIR = '.'

def load_models():
    """Load all trained models."""
    models = {}
    for prop in PROPS_TO_PREDICT:
        model_path = f'model_{prop}.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[prop] = pickle.load(f)
            print(f"  ✓ Loaded {prop} model")
        else:
            print(f"  ✗ Missing {model_path}")
    return models


def load_latest_data():
    """Load the most recent player data."""
    if os.path.exists('player_games_schedule.csv'):
        df = pd.read_csv('player_games_schedule.csv', parse_dates=['game_date'])
        print(f"  ✓ Loaded {len(df):,} player-games")
        print(f"  Latest date: {df['game_date'].max().date()}")
        return df
    else:
        raise FileNotFoundError("player_games_schedule.csv not found!")


def get_latest_player_features(df, player_name):
    """Get the most recent features for a player."""
    player_df = df[df['player'] == player_name].sort_values('game_date')
    if len(player_df) == 0:
        return None
    return player_df.iloc[-1]


def get_players_for_prediction(df, min_games=10):
    """Get list of players with sufficient history."""
    player_games = df.groupby('player')['game_date'].count()
    valid_players = player_games[player_games >= min_games].index.tolist()
    return valid_players


def create_combo_features(row):
    """Create combination line proxies from component L10 averages."""
    row = row.copy()
    if 'pts_l10' in row and 'trb_l10' in row and 'ast_l10' in row:
        row['pra_l10'] = row['pts_l10'] + row['trb_l10'] + row['ast_l10']
        row['pr_l10'] = row['pts_l10'] + row['trb_l10']
        row['pa_l10'] = row['pts_l10'] + row['ast_l10']
    return row


def predict_player(player_name, df, models, feature_cols):
    """Generate predictions for a single player."""
    # Get latest features for this player
    latest = get_latest_player_features(df, player_name)
    if latest is None:
        return None
    
    # Add combo features
    latest = create_combo_features(latest)
    
    predictions = {'player': player_name}
    
    for prop, model_data in models.items():
        model = model_data['model']
        scaler = model_data['scaler']
        medians = model_data['medians']
        features = model_data['features']
        
        # Prepare features
        X = pd.DataFrame([latest[features]])
        X = X.fillna(medians)
        X_scaled = scaler.transform(X)
        
        # Predict
        pred = model.predict(X_scaled)[0]
        
        # Get line (L10 average)
        line_col = f'{prop}_l10'
        if line_col in latest.index:
            line = latest[line_col]
        else:
            line = pred  # Fallback
        
        predictions[f'{prop}_pred'] = round(pred, 1)
        predictions[f'{prop}_line'] = round(line, 1) if not pd.isna(line) else None
        predictions[f'{prop}_edge'] = round(pred - line, 1) if not pd.isna(line) else None
    
    return predictions


def generate_all_predictions(df, models, players=None):
    """Generate predictions for all valid players."""
    if players is None:
        players = get_players_for_prediction(df)
    
    # Get feature columns from first model
    first_model = list(models.values())[0]
    feature_cols = first_model['features']
    
    all_predictions = []
    for player in players:
        pred = predict_player(player, df, models, feature_cols)
        if pred:
            all_predictions.append(pred)
    
    return pd.DataFrame(all_predictions)


def create_bet_recommendations(predictions_df, lines_override=None):
    """
    Create betting recommendations from predictions.
    
    Args:
        predictions_df: DataFrame with predictions
        lines_override: Optional dict of {player: {prop: line}} for real sportsbook lines
    """
    bets = []
    
    for _, row in predictions_df.iterrows():
        player = row['player']
        
        for prop in PROPS_TO_PREDICT:
            pred_col = f'{prop}_pred'
            line_col = f'{prop}_line'
            edge_col = f'{prop}_edge'
            
            if pred_col not in row or pd.isna(row[pred_col]):
                continue
            
            pred = row[pred_col]
            
            # Use override line if provided, else use L10
            if lines_override and player in lines_override and prop in lines_override[player]:
                line = lines_override[player][prop]
            elif line_col in row and not pd.isna(row[line_col]):
                line = row[line_col]
            else:
                continue
            
            edge = pred - line
            direction = 'OVER' if edge > 0 else 'UNDER'
            abs_edge = abs(edge)
            
            # Classify bet strength
            if abs_edge >= MIN_EDGE_STRONG:
                strength = 'STRONG'
            elif abs_edge >= MIN_EDGE_MODERATE:
                strength = 'MODERATE'
            else:
                continue  # Skip small edges
            
            # Format prop name nicely
            prop_display = {
                'pts': 'Points',
                'trb': 'Rebounds',
                'ast': 'Assists',
                'pra': 'Pts+Reb+Ast',
                'pr': 'Pts+Reb',
                'pa': 'Pts+Ast',
                'stl': 'Steals',
                'blk': 'Blocks',
                'tov': 'Turnovers'
            }.get(prop, prop.upper())
            
            bets.append({
                'player': player,
                'prop': prop,
                'prop_display': prop_display,
                'direction': direction,
                'line': line,
                'prediction': pred,
                'edge': abs_edge,
                'strength': strength,
                'timestamp': datetime.now().isoformat()
            })
    
    return pd.DataFrame(bets)


def format_recommendations(bets_df):
    """Format betting recommendations for display."""
    if len(bets_df) == 0:
        return "No bets meeting criteria."
    
    # Sort by edge
    bets_df = bets_df.sort_values('edge', ascending=False)
    
    lines = []
    lines.append("=" * 70)
    lines.append("BET RECOMMENDATIONS")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    
    # Strong bets
    strong = bets_df[bets_df['strength'] == 'STRONG']
    if len(strong) > 0:
        lines.append(f"\n★★★ STRONG BETS ({len(strong)}):")
        for _, bet in strong.iterrows():
            lines.append(f"  {bet['player']:<25} {bet['prop_display']:<12} {bet['direction']:<5} {bet['line']:.1f}")
            lines.append(f"      Prediction: {bet['prediction']:.1f}, Edge: {'+' if bet['direction']=='OVER' else '-'}{bet['edge']:.1f} pts")
    
    # Moderate bets
    moderate = bets_df[bets_df['strength'] == 'MODERATE']
    if len(moderate) > 0:
        lines.append(f"\n★★ MODERATE BETS ({len(moderate)}):")
        for _, bet in moderate.iterrows():
            lines.append(f"  {bet['player']:<25} {bet['prop_display']:<12} {bet['direction']:<5} {bet['line']:.1f}")
            lines.append(f"      Prediction: {bet['prediction']:.1f}, Edge: {'+' if bet['direction']=='OVER' else '-'}{bet['edge']:.1f} pts")
    
    lines.append("\n" + "=" * 70)
    lines.append(f"Total bets: {len(bets_df)} (Strong: {len(strong)}, Moderate: {len(moderate)})")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def save_locked_predictions(bets_df, predictions_df, date_str=None):
    """Save predictions to timestamped files."""
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed predictions
    predictions_file = f'predictions_locked_{date_str}_{timestamp}.csv'
    predictions_df.to_csv(predictions_file, index=False)
    print(f"  ✓ Saved: {predictions_file}")
    
    # Save bet recommendations
    bets_file = f'bets_locked_{date_str}_{timestamp}.csv'
    bets_df.to_csv(bets_file, index=False)
    print(f"  ✓ Saved: {bets_file}")
    
    # Save formatted summary
    summary_file = f'bets_summary_{date_str}_{timestamp}.txt'
    summary = format_recommendations(bets_df)
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"  ✓ Saved: {summary_file}")
    
    return predictions_file, bets_file, summary_file


def main():
    print("=" * 60)
    print("Task 7: Daily NBA Props Predictions")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load models
    print("\n[1] Loading trained models...")
    models = load_models()
    if len(models) == 0:
        print("ERROR: No models found! Run task6_train_models.py first.")
        return
    
    # Step 2: Load data
    print("\n[2] Loading player data...")
    try:
        df = load_latest_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    # Step 3: Get valid players
    print("\n[3] Getting players with sufficient history...")
    players = get_players_for_prediction(df, min_games=10)
    print(f"  ✓ {len(players)} players eligible")
    
    # Step 4: Generate predictions
    print("\n[4] Generating predictions...")
    predictions_df = generate_all_predictions(df, models, players)
    print(f"  ✓ Generated predictions for {len(predictions_df)} players")
    
    # Step 5: Create bet recommendations
    print("\n[5] Creating bet recommendations...")
    bets_df = create_bet_recommendations(predictions_df)
    print(f"  ✓ Found {len(bets_df)} bets meeting criteria")
    
    # Step 6: Display summary
    print("\n" + format_recommendations(bets_df))
    
    # Step 7: Save locked predictions
    print("\n[6] Saving locked predictions...")
    save_locked_predictions(bets_df, predictions_df)
    
    print("\n" + "=" * 60)
    print("PREDICTIONS LOCKED! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
