"""
Task 7: Daily Prediction Script
NBA Player Props Betting System

This script generates predictions for today's games using trained models.

Usage:
1. Ensure models are trained (run task6_train_models.py first)
2. Provide today's matchups via:
   - Interactive input
   - CSV file (--input matchups.csv)
   - Command line (--players "LeBron James,LAL,GSW")

Output:
- Console predictions
- Optional CSV export (--output predictions_today.csv)
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

PROPS = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pra', 'pr', 'pa']

PROP_DISPLAY_NAMES = {
    'pts': 'Points',
    'trb': 'Rebounds', 
    'ast': 'Assists',
    'stl': 'Steals',
    'blk': 'Blocks',
    'tov': 'Turnovers',
    'pra': 'Pts+Reb+Ast',
    'pr': 'Pts+Reb',
    'pa': 'Pts+Ast'
}

# Feature configuration for computing new predictions
ROLLING_STATS = ['pts', 'trb', 'ast', 'mp', 'fga', 'fg_pct', '3pa', '3p_pct', 
                 'fta', 'ft_pct', 'tov', 'stl', 'blk']
VOLATILITY_STATS = ['pts', 'trb', 'ast', 'mp']
TREND_STATS = ['pts', 'trb', 'ast']


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_models(model_dir: Path = Path('.')) -> dict:
    """
    Load all trained models from pickle files.
    
    Returns:
        Dictionary of {prop_name: model_data}
    """
    models = {}
    missing = []
    
    for prop in PROPS:
        model_path = model_dir / f'model_{prop}.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[prop] = pickle.load(f)
        else:
            missing.append(prop)
    
    if missing:
        print(f"⚠  Missing models: {missing}")
        print("   Run task6_train_models.py first to train models.")
    
    if models:
        print(f"✓ Loaded {len(models)} models: {list(models.keys())}")
    
    return models


# =============================================================================
# HISTORICAL DATA LOADING
# =============================================================================

def load_historical_data(data_path: str = 'player_games_schedule.csv') -> pd.DataFrame:
    """Load historical player-game data with all features."""
    print(f"Loading historical data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['game_date'])
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    return df


def get_team_abbreviations(df: pd.DataFrame) -> dict:
    """Get mapping of team names to abbreviations."""
    teams = df['tm'].unique().tolist()
    return {t: t for t in teams}  # Identity mapping for now


def search_players(df: pd.DataFrame, query: str, limit: int = 10) -> list:
    """
    Search for players by partial name match.
    
    Args:
        df: Historical data
        query: Search string (case-insensitive)
        limit: Maximum results to return
        
    Returns:
        List of matching player names
    """
    import unicodedata
    
    def normalize(s):
        """Remove accents and lowercase."""
        return ''.join(
            c for c in unicodedata.normalize('NFD', s.lower())
            if unicodedata.category(c) != 'Mn'
        )
    
    players = df['player'].unique()
    query_norm = normalize(query)
    
    # Exact matches first (normalized)
    exact = [p for p in players if query_norm == normalize(p)]
    if exact:
        return exact[:limit]
    
    # Partial matches (normalized)
    partial = [p for p in players if query_norm in normalize(p)]
    
    return partial[:limit]


def get_player_with_fuzzy_match(df: pd.DataFrame, player_name: str) -> str:
    """
    Get exact player name, using fuzzy matching if needed.
    
    Returns:
        Exact player name from database, or None if not found
    """
    import unicodedata
    
    def normalize(s):
        """Remove accents and lowercase."""
        return ''.join(
            c for c in unicodedata.normalize('NFD', s.lower())
            if unicodedata.category(c) != 'Mn'
        )
    
    # Check exact match first
    if player_name in df['player'].unique():
        return player_name
    
    # Try normalized exact match
    query_norm = normalize(player_name)
    for p in df['player'].unique():
        if normalize(p) == query_norm:
            return p
    
    # Try partial match
    matches = search_players(df, player_name, limit=5)
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Check if first part of name matches exactly
        first_name = player_name.split()[0].lower()
        for m in matches:
            if m.lower().startswith(first_name):
                return m
        return None  # Multiple ambiguous matches
    else:
        return None


# =============================================================================
# FEATURE COMPUTATION FOR NEW GAMES
# =============================================================================

def get_player_latest_features(df: pd.DataFrame, player_name: str) -> pd.Series:
    """
    Get the most recent features for a player.
    
    These represent the player's rolling averages ENTERING their next game.
    """
    player_df = df[df['player'] == player_name].sort_values('game_date')
    
    if len(player_df) == 0:
        return None
    
    # Get the most recent row - its features are what we'd use for the NEXT game
    latest = player_df.iloc[-1].copy()
    
    # The rolling features in this row already exclude the current game
    # (they were calculated with shift(1)), so they're ready to use
    return latest


def get_opponent_features(df: pd.DataFrame, opponent: str) -> dict:
    """
    Get opponent's defensive features (what they typically allow).
    
    Returns the most recent defensive stats for the opponent team.
    """
    # Get all games where this team played (they are 'tm')
    opp_games = df[df['tm'] == opponent].sort_values('game_date')
    
    if len(opp_games) == 0:
        return {}
    
    # Get the most recent game for opponent-level features
    latest_game = opp_games.iloc[-1]
    
    # Extract opponent features (these are stored from opponent's perspective)
    opp_features = {
        'opp_pts_allowed_l10': latest_game.get('opp_pts_allowed_l10', np.nan),
        'opp_trb_allowed_l10': latest_game.get('opp_trb_allowed_l10', np.nan),
        'opp_ast_allowed_l10': latest_game.get('opp_ast_allowed_l10', np.nan),
        'opp_fga_allowed_l10': latest_game.get('opp_fga_allowed_l10', np.nan),
        'opp_pts_allowed_rank': latest_game.get('opp_pts_allowed_rank', np.nan),
        'opp_trb_allowed_rank': latest_game.get('opp_trb_allowed_rank', np.nan),
        'opp_ast_allowed_rank': latest_game.get('opp_ast_allowed_rank', np.nan),
        'opp_tm_pts_l10': latest_game.get('opp_tm_pts_l10', np.nan),
        'opp_tm_fga_l10': latest_game.get('opp_tm_fga_l10', np.nan),
    }
    
    return opp_features


def compute_schedule_features(df: pd.DataFrame, player_name: str, 
                              game_date: datetime, 
                              use_latest_features: bool = True) -> dict:
    """
    Compute schedule features for a player's upcoming game.
    
    Args:
        df: Historical data
        player_name: Player name
        game_date: Date of upcoming game
        use_latest_features: If True, use player's most recent schedule features
                            (recommended for predictions since they represent current state)
        
    Returns:
        Dictionary of schedule features
    """
    # Get player's recent games
    player_df = df[df['player'] == player_name].sort_values('game_date')
    
    if len(player_df) == 0:
        return {}
    
    latest = player_df.iloc[-1]
    latest_date = latest['game_date']
    
    # For predictions, use the player's most recent schedule features
    # This represents their current state and avoids issues with future dates
    if use_latest_features:
        # Use the schedule features from the player's last game
        # These represent realistic values
        return {
            'days_rest': latest.get('days_rest', 2.0),
            'player_days_rest': latest.get('player_days_rest', 2.0),
            'is_b2b': latest.get('is_b2b', 0),
            'is_b2b_second': latest.get('is_b2b_second', 0),
            'games_last_7d': latest.get('games_last_7d', 3.0),
            'games_last_14d': latest.get('games_last_14d', 6.0),
            'mp_l3': latest.get('mp_l3', np.nan),
            'mp_avg_l3': latest.get('mp_avg_l3', np.nan),
            'high_min_prev': latest.get('high_min_prev', 0.0),
            'games_streak': latest.get('games_streak', len(player_df)),
        }
    
    # Calculate fresh schedule features (for backtesting on historical dates)
    days_rest = (game_date - latest_date).days
    
    # Cap days rest to realistic range (0-10 days)
    days_rest = min(max(days_rest, 0), 10)
    
    # Back-to-back indicator
    is_b2b = 1 if days_rest <= 1 else 0
    is_b2b_second = 1 if days_rest == 1 else 0
    
    # Games in last 7/14 days (relative to game_date)
    cutoff_7d = game_date - timedelta(days=7)
    cutoff_14d = game_date - timedelta(days=14)
    games_last_7d = len(player_df[(player_df['game_date'] > cutoff_7d) & 
                                   (player_df['game_date'] < game_date)])
    games_last_14d = len(player_df[(player_df['game_date'] > cutoff_14d) & 
                                    (player_df['game_date'] < game_date)])
    
    # Minutes load (from most recent games before prediction date)
    recent_games = player_df[player_df['game_date'] < game_date].tail(3)
    mp_l3 = recent_games['mp'].sum() if len(recent_games) > 0 else np.nan
    mp_avg_l3 = recent_games['mp'].mean() if len(recent_games) > 0 else np.nan
    
    # High minutes previous game
    high_min_prev = 1.0 if latest['mp'] >= 36 else 0.0
    
    # Games streak
    games_streak = len(player_df)
    
    return {
        'days_rest': days_rest,
        'player_days_rest': days_rest,
        'is_b2b': is_b2b,
        'is_b2b_second': is_b2b_second,
        'games_last_7d': games_last_7d,
        'games_last_14d': games_last_14d,
        'mp_l3': mp_l3,
        'mp_avg_l3': mp_avg_l3,
        'high_min_prev': high_min_prev,
        'games_streak': games_streak,
    }


def prepare_features_for_prediction(df: pd.DataFrame, player_name: str, 
                                     team: str, opponent: str,
                                     game_date: datetime,
                                     model_features: list) -> pd.DataFrame:
    """
    Prepare a complete feature vector for prediction.
    
    Args:
        df: Historical data
        player_name: Player name
        team: Player's team abbreviation
        opponent: Opponent team abbreviation
        game_date: Date of the game
        model_features: List of features required by the model
        
    Returns:
        DataFrame with one row containing all features
    """
    # Try to find player with fuzzy matching
    exact_name = get_player_with_fuzzy_match(df, player_name)
    
    if exact_name is None:
        print(f"  ⚠ Player '{player_name}' not found in historical data")
        # Suggest similar names
        similar = search_players(df, player_name.split()[0], limit=3)
        if similar:
            print(f"    Did you mean: {similar}?")
        return None
    
    if exact_name != player_name:
        print(f"  → Matched '{player_name}' to '{exact_name}'")
    
    # Get player's latest rolling features
    player_latest = get_player_latest_features(df, exact_name)
    
    if player_latest is None:
        print(f"  ⚠ No historical data found for '{exact_name}'")
        return None
    
    # Start with player's rolling features
    features = player_latest.to_dict()
    
    # Update opponent features (the player's row has old opponent data)
    opp_features = get_opponent_features(df, opponent)
    features.update(opp_features)
    
    # Update schedule features for THIS game
    schedule_features = compute_schedule_features(df, exact_name, game_date)
    features.update(schedule_features)
    
    # Create DataFrame with just the model features
    feature_vector = {}
    for feat in model_features:
        if feat in features:
            feature_vector[feat] = features[feat]
        else:
            # Use NaN for missing features (model will use median)
            feature_vector[feat] = np.nan
    
    return pd.DataFrame([feature_vector])


# =============================================================================
# PREDICTION ENGINE
# =============================================================================

def predict_player_props(df: pd.DataFrame, models: dict, 
                         player_name: str, team: str, opponent: str,
                         game_date: datetime) -> dict:
    """
    Generate predictions for all props for a single player.
    
    Returns:
        Dictionary with predictions for each prop
    """
    # Resolve player name first (with fuzzy matching)
    exact_name = get_player_with_fuzzy_match(df, player_name)
    
    if exact_name is None:
        # Suggest similar names
        similar = search_players(df, player_name.split()[0] if ' ' in player_name else player_name, limit=3)
        if similar:
            print(f"  ⚠ Player '{player_name}' not found. Did you mean: {similar}?")
        else:
            print(f"  ⚠ Player '{player_name}' not found in historical data")
        return {'player': player_name, 'team': team, 'opponent': opponent, 
                'game_date': game_date.strftime('%Y-%m-%d')}
    
    if exact_name != player_name:
        print(f"  → Using '{exact_name}' for '{player_name}'")
    
    predictions = {
        'player': player_name,  # Keep original name for display
        'team': team,
        'opponent': opponent,
        'game_date': game_date.strftime('%Y-%m-%d')
    }
    
    # Get player's latest data once for line calculations
    player_latest = get_player_latest_features(df, exact_name)
    
    for prop, model_data in models.items():
        model = model_data['model']
        scaler = model_data['scaler']
        medians = model_data['medians']
        features = model_data['features']
        
        # Prepare features (pass exact_name directly since we already resolved it)
        X = prepare_features_for_prediction(
            df, exact_name, team, opponent, game_date, features
        )
        
        if X is None:
            predictions[prop] = None
            predictions[f'{prop}_line'] = None
            continue
        
        # Fill missing values with medians
        X = X.fillna(medians)
        
        # Scale
        X_scaled = scaler.transform(X)
        
        # Predict
        pred = model.predict(X_scaled)[0]
        
        # Floor predictions at 0 (counting stats can't be negative)
        pred = max(0, pred)
        
        predictions[prop] = round(pred, 1)
        
        # Get player's L10 as line proxy (using exact_name, already fetched)
        if player_latest is not None:
            if prop == 'pra':
                line = player_latest.get('pts_l10', 0) + player_latest.get('trb_l10', 0) + player_latest.get('ast_l10', 0)
            elif prop == 'pr':
                line = player_latest.get('pts_l10', 0) + player_latest.get('trb_l10', 0)
            elif prop == 'pa':
                line = player_latest.get('pts_l10', 0) + player_latest.get('ast_l10', 0)
            else:
                line = player_latest.get(f'{prop}_l10', np.nan)
            predictions[f'{prop}_line'] = round(line, 1) if pd.notna(line) else None
        else:
            predictions[f'{prop}_line'] = None
    
    return predictions


def find_edges(predictions: dict, min_edge: float = 1.0) -> list:
    """
    Identify predictions with significant edge over the line.
    
    Args:
        predictions: Dictionary of predictions
        min_edge: Minimum edge (prediction - line) to flag
        
    Returns:
        List of (prop, prediction, line, edge, direction) tuples
    """
    edges = []
    
    for prop in PROPS:
        pred = predictions.get(prop)
        line = predictions.get(f'{prop}_line')
        
        if pred is None or line is None:
            continue
        
        edge = pred - line
        direction = 'OVER' if edge > 0 else 'UNDER'
        
        if abs(edge) >= min_edge:
            edges.append({
                'prop': PROP_DISPLAY_NAMES.get(prop, prop),
                'prediction': pred,
                'line': line,
                'edge': abs(edge),
                'direction': direction
            })
    
    return sorted(edges, key=lambda x: -x['edge'])


# =============================================================================
# INPUT HANDLING
# =============================================================================

def parse_matchups_from_csv(filepath: str) -> list:
    """
    Parse matchups from a CSV file.
    
    Expected columns: player, team, opponent, game_date (optional)
    """
    df = pd.read_csv(filepath)
    
    required_cols = ['player', 'team', 'opponent']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    
    matchups = []
    for _, row in df.iterrows():
        game_date = pd.to_datetime(row.get('game_date', datetime.now().date()))
        matchups.append({
            'player': row['player'],
            'team': row['team'],
            'opponent': row['opponent'],
            'game_date': game_date
        })
    
    return matchups


def parse_matchups_from_string(matchup_str: str, game_date: datetime = None) -> list:
    """
    Parse matchups from command line string.
    
    Format: "Player Name,TEAM,OPP;Player2,TEAM2,OPP2"
    """
    if game_date is None:
        game_date = datetime.now()
    
    matchups = []
    for entry in matchup_str.split(';'):
        parts = entry.strip().split(',')
        if len(parts) >= 3:
            matchups.append({
                'player': parts[0].strip(),
                'team': parts[1].strip().upper(),
                'opponent': parts[2].strip().upper(),
                'game_date': game_date
            })
    
    return matchups


def interactive_input(df: pd.DataFrame) -> list:
    """Gather matchups interactively from user."""
    print("\n" + "="*60)
    print("INTERACTIVE INPUT MODE")
    print("="*60)
    
    # Show available teams
    teams = sorted(df['tm'].unique())
    print(f"\nAvailable teams: {', '.join(teams)}")
    
    # Get game date
    date_str = input("\nGame date (YYYY-MM-DD) or press Enter for today: ").strip()
    if date_str:
        game_date = pd.to_datetime(date_str)
    else:
        game_date = datetime.now()
    
    matchups = []
    print("\nEnter matchups (leave player blank to finish):")
    
    while True:
        print(f"\n--- Matchup {len(matchups) + 1} ---")
        player = input("Player name: ").strip()
        if not player:
            break
        
        team = input("Team (e.g., LAL): ").strip().upper()
        opponent = input("Opponent (e.g., GSW): ").strip().upper()
        
        if team and opponent:
            matchups.append({
                'player': player,
                'team': team,
                'opponent': opponent,
                'game_date': game_date
            })
            print(f"  ✓ Added: {player} ({team}) vs {opponent}")
    
    return matchups


def get_todays_players_from_data(df: pd.DataFrame, date: datetime = None,
                                  min_minutes: float = 0) -> list:
    """
    Get all players who played on a specific date.
    Useful for backtesting on historical dates.
    
    Args:
        df: Historical data
        date: Date to get players for
        min_minutes: Minimum average minutes (mp_l10) to include player
    """
    if date is None:
        date = df['game_date'].max()
    
    # Get games on that date
    date_games = df[df['game_date'].dt.date == date.date()].copy()
    
    if len(date_games) == 0:
        print(f"No games found for {date.date()}")
        return []
    
    # Filter by minutes if specified
    if min_minutes > 0:
        date_games = date_games[date_games['mp_l10'] >= min_minutes]
        if len(date_games) == 0:
            print(f"No players with {min_minutes}+ avg minutes found for {date.date()}")
            return []
    
    matchups = []
    for _, row in date_games.iterrows():
        matchups.append({
            'player': row['player'],
            'team': row['tm'],
            'opponent': row['opp'],
            'game_date': pd.to_datetime(date),
            'mp_l10': row.get('mp_l10', 0)  # Include for display
        })
    
    return matchups


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_predictions(predictions: dict, show_edges: bool = True):
    """Print formatted predictions for a player."""
    print(f"\n{'─'*60}")
    print(f"  {predictions['player']} ({predictions['team']} vs {predictions['opponent']})")
    print(f"  {predictions['game_date']}")
    print(f"{'─'*60}")
    
    # Main props table
    print(f"\n  {'Prop':<12} {'Pred':>8} {'Line':>8} {'Edge':>8}")
    print(f"  {'─'*40}")
    
    for prop in PROPS:
        pred = predictions.get(prop)
        line = predictions.get(f'{prop}_line')
        
        if pred is None:
            continue
        
        edge = pred - line if line else 0
        edge_str = f"{edge:+.1f}" if line else "N/A"
        line_str = f"{line:.1f}" if line else "N/A"
        
        # Highlight significant edges
        marker = ""
        if line and abs(edge) >= 2:
            marker = " **" if edge > 0 else " *"
        
        print(f"  {PROP_DISPLAY_NAMES.get(prop, prop):<12} {pred:>8.1f} {line_str:>8} {edge_str:>8}{marker}")
    
    # Show top edges
    if show_edges:
        edges = find_edges(predictions, min_edge=1.5)
        if edges:
            print(f"\n  [TOP EDGES]")
            for e in edges[:3]:
                print(f"  → {e['prop']}: {e['direction']} {e['line']:.1f} "
                      f"(pred: {e['prediction']:.1f}, edge: {e['edge']:.1f})")


def save_predictions_csv(all_predictions: list, output_path: str):
    """Save all predictions to CSV."""
    rows = []
    for pred in all_predictions:
        row = {
            'player': pred['player'],
            'team': pred['team'],
            'opponent': pred['opponent'],
            'game_date': pred['game_date']
        }
        for prop in PROPS:
            row[f'{prop}_pred'] = pred.get(prop)
            row[f'{prop}_line'] = pred.get(f'{prop}_line')
            if pred.get(prop) and pred.get(f'{prop}_line'):
                row[f'{prop}_edge'] = pred[prop] - pred[f'{prop}_line']
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate NBA player prop predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python task7_daily_predictions.py --interactive
  python task7_daily_predictions.py --input matchups.csv
  python task7_daily_predictions.py --players "LeBron James,LAL,GSW;Stephen Curry,GSW,LAL"
  python task7_daily_predictions.py --date 2025-02-07  # All players from that date
  python task7_daily_predictions.py --date 2025-02-07 --min-minutes 20  # Only rotation players
  python task7_daily_predictions.py --date 2025-02-07 --min-edge 2.0  # Only show big edges
        """
    )
    
    parser.add_argument('--input', '-i', help='CSV file with matchups')
    parser.add_argument('--players', '-p', help='Matchups string: "Player,TEAM,OPP;..."')
    parser.add_argument('--date', '-d', help='Get all players from this date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', help='Output CSV path')
    parser.add_argument('--interactive', action='store_true', help='Interactive input mode')
    parser.add_argument('--data', default='player_games_schedule.csv', help='Historical data file')
    parser.add_argument('--model-dir', default='.', help='Directory with model .pkl files')
    parser.add_argument('--min-minutes', type=float, default=0, 
                        help='Minimum average minutes (L10) to include player (default: 0)')
    parser.add_argument('--min-edge', type=float, default=0,
                        help='Only show players with at least this edge on any prop (default: 0)')
    parser.add_argument('--top', type=int, default=None,
                        help='Only show top N players by max edge')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Task 7: Daily Predictions")
    print("="*60)
    
    # Load models
    models = load_models(Path(args.model_dir))
    
    if not models:
        print("\n✗ No models loaded. Run task6_train_models.py first.")
        return
    
    # Load historical data
    df = load_historical_data(args.data)
    
    # Get matchups
    matchups = []
    
    if args.input:
        matchups = parse_matchups_from_csv(args.input)
        print(f"\nLoaded {len(matchups)} matchups from {args.input}")
    
    elif args.players:
        matchups = parse_matchups_from_string(args.players)
        print(f"\nParsed {len(matchups)} matchups from command line")
    
    elif args.date:
        game_date = pd.to_datetime(args.date)
        matchups = get_todays_players_from_data(df, game_date, min_minutes=args.min_minutes)
        print(f"\nFound {len(matchups)} players from {args.date}")
        if args.min_minutes > 0:
            print(f"  (filtered to {args.min_minutes}+ avg minutes)")
    
    elif args.interactive:
        matchups = interactive_input(df)
    
    else:
        # Default: show last date in data
        last_date = df['game_date'].max()
        print(f"\nNo input specified. Showing predictions for {last_date.date()}")
        print("Use --help for options.")
        matchups = get_todays_players_from_data(df, last_date, min_minutes=args.min_minutes)[:10]  # Limit to 10
    
    if not matchups:
        print("\nNo matchups to predict.")
        return
    
    # Generate predictions
    print(f"\n{'='*60}")
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    all_predictions = []
    
    for matchup in matchups:
        predictions = predict_player_props(
            df, models,
            matchup['player'],
            matchup['team'],
            matchup['opponent'],
            matchup['game_date']
        )
        
        if predictions.get('pts') is not None:
            # Calculate max edge for this player
            edges = find_edges(predictions, min_edge=0)
            max_edge = max([e['edge'] for e in edges]) if edges else 0
            predictions['_max_edge'] = max_edge
            
            all_predictions.append(predictions)
    
    # Apply edge filter if specified
    if args.min_edge > 0:
        all_predictions = [p for p in all_predictions if p.get('_max_edge', 0) >= args.min_edge]
        print(f"\nFiltered to {len(all_predictions)} players with edge >= {args.min_edge}")
    
    # Apply top N filter if specified
    if args.top:
        all_predictions = sorted(all_predictions, key=lambda x: -x.get('_max_edge', 0))[:args.top]
        print(f"\nShowing top {len(all_predictions)} players by max edge")
    
    # Print predictions
    for predictions in all_predictions:
        print_predictions(predictions)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"Players predicted: {len(all_predictions)}/{len(matchups)}")
    
    # Find best edges across all players
    all_edges = []
    for pred in all_predictions:
        edges = find_edges(pred, min_edge=2.0)
        for e in edges:
            e['player'] = pred['player']
            all_edges.append(e)
    
    all_edges = sorted(all_edges, key=lambda x: -x['edge'])
    
    if all_edges:
        print(f"\n[TOP EDGES ACROSS ALL PLAYERS (≥2.0)]")
        for e in all_edges[:10]:
            print(f"  {e['player']:<20} {e['prop']:<12} {e['direction']:<5} "
                  f"Line: {e['line']:.1f} → Pred: {e['prediction']:.1f} "
                  f"(+{e['edge']:.1f})")
    
    # Save if output specified
    if args.output:
        save_predictions_csv(all_predictions, args.output)
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
