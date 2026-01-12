"""
Task 8: Compare to Real Betting Lines
NBA Player Props Betting System

This script compares our model predictions to actual sportsbook lines
to identify +EV betting opportunities.

Features:
1. Integration with The Odds API for live lines
2. Manual line entry mode
3. Edge calculation and bet recommendations
4. Bankroll management suggestions

Usage:
  # With API (requires ODDS_API_KEY environment variable)
  python task8_betting_lines.py --api --date 2025-02-07

  # Manual entry mode
  python task8_betting_lines.py --manual --players "LeBron James,LAL,GSW"

  # From CSV with lines
  python task8_betting_lines.py --input lines.csv
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import requests for API calls
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


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

# The Odds API prop market mappings
ODDS_API_MARKETS = {
    'pts': 'player_points',
    'trb': 'player_rebounds',
    'ast': 'player_assists',
    'stl': 'player_steals',
    'blk': 'player_blocks',
    'tov': 'player_turnovers',
    'pra': 'player_points_rebounds_assists',
    'pr': 'player_points_rebounds',
    'pa': 'player_points_assists'
}

# Minimum edge thresholds for bet recommendations
EDGE_THRESHOLDS = {
    'strong': 3.0,    # Strong bet
    'moderate': 2.0,  # Moderate bet
    'slight': 1.0     # Slight edge (caution)
}

# Kelly Criterion fraction (conservative)
KELLY_FRACTION = 0.25  # Quarter Kelly


# =============================================================================
# MODEL & DATA LOADING (from Task 7)
# =============================================================================

def load_models(model_dir: Path = Path('.')) -> dict:
    """Load all trained models from pickle files."""
    models = {}
    for prop in PROPS:
        model_path = model_dir / f'model_{prop}.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[prop] = pickle.load(f)
    
    if models:
        print(f"✓ Loaded {len(models)} models")
    else:
        print("✗ No models found")
    
    return models


def load_historical_data(data_path: str = 'player_games_schedule.csv') -> pd.DataFrame:
    """Load historical player-game data."""
    print(f"Loading historical data...")
    df = pd.read_csv(data_path, parse_dates=['game_date'])
    print(f"  Loaded {len(df):,} rows")
    return df


# =============================================================================
# PLAYER NAME MATCHING (from Task 7)
# =============================================================================

def normalize_name(s):
    """Remove accents and lowercase."""
    import unicodedata
    return ''.join(
        c for c in unicodedata.normalize('NFD', s.lower())
        if unicodedata.category(c) != 'Mn'
    )


def get_player_match(df: pd.DataFrame, player_name: str) -> str:
    """Get exact player name using fuzzy matching."""
    if player_name in df['player'].unique():
        return player_name
    
    query_norm = normalize_name(player_name)
    for p in df['player'].unique():
        if normalize_name(p) == query_norm:
            return p
    
    # Partial match
    for p in df['player'].unique():
        if query_norm in normalize_name(p):
            return p
    
    return None


# =============================================================================
# PREDICTION ENGINE (simplified from Task 7)
# =============================================================================

def get_player_features(df: pd.DataFrame, player_name: str) -> pd.Series:
    """Get most recent features for a player."""
    player_df = df[df['player'] == player_name].sort_values('game_date')
    if len(player_df) == 0:
        return None
    return player_df.iloc[-1]


def predict_prop(df: pd.DataFrame, model_data: dict, player_name: str) -> float:
    """Predict a single prop for a player."""
    exact_name = get_player_match(df, player_name)
    if exact_name is None:
        return None
    
    player_latest = get_player_features(df, exact_name)
    if player_latest is None:
        return None
    
    # Build feature vector
    features = model_data['features']
    feature_vector = {}
    for feat in features:
        if feat in player_latest.index:
            feature_vector[feat] = player_latest[feat]
        else:
            feature_vector[feat] = np.nan
    
    X = pd.DataFrame([feature_vector])
    X = X.fillna(model_data['medians'])
    X_scaled = model_data['scaler'].transform(X)
    
    pred = model_data['model'].predict(X_scaled)[0]
    return max(0, pred)  # Floor at 0


def get_all_predictions(df: pd.DataFrame, models: dict, player_name: str) -> dict:
    """Get predictions for all props for a player."""
    predictions = {}
    for prop, model_data in models.items():
        pred = predict_prop(df, model_data, player_name)
        if pred is not None:
            predictions[prop] = round(pred, 1)
    return predictions


# =============================================================================
# THE ODDS API INTEGRATION
# =============================================================================

def get_odds_api_key() -> str:
    """Get API key from environment."""
    key = os.environ.get('ODDS_API_KEY')
    if not key:
        print("⚠ ODDS_API_KEY not found in environment")
        print("  Get a free key at: https://the-odds-api.com/")
        print("  Set it: export ODDS_API_KEY=your_key_here")
    return key


def fetch_nba_games(api_key: str) -> list:
    """Fetch upcoming NBA games from The Odds API."""
    if not HAS_REQUESTS:
        print("⚠ requests library not installed. Run: pip install requests")
        return []
    
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    params = {
        'apiKey': api_key,
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        games = response.json()
        print(f"✓ Found {len(games)} upcoming NBA games")
        return games
    except Exception as e:
        print(f"✗ API error: {e}")
        return []


def fetch_player_props(api_key: str, event_id: str, market: str) -> dict:
    """Fetch player props for a specific game and market."""
    if not HAS_REQUESTS:
        return {}
    
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': market,
        'oddsFormat': 'american'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  ⚠ Could not fetch {market}: {e}")
        return {}


def parse_prop_lines(odds_data: dict, prop_type: str) -> dict:
    """Parse player prop lines from API response."""
    lines = {}
    
    if not odds_data or 'bookmakers' not in odds_data:
        return lines
    
    for bookmaker in odds_data.get('bookmakers', []):
        book_name = bookmaker.get('key', 'unknown')
        
        for market in bookmaker.get('markets', []):
            for outcome in market.get('outcomes', []):
                player = outcome.get('description', '')
                point = outcome.get('point')
                price = outcome.get('price')
                name = outcome.get('name')  # Over or Under
                
                if player and point is not None:
                    if player not in lines:
                        lines[player] = {
                            'line': point,
                            'over_odds': None,
                            'under_odds': None,
                            'book': book_name
                        }
                    
                    if name == 'Over':
                        lines[player]['over_odds'] = price
                    elif name == 'Under':
                        lines[player]['under_odds'] = price
    
    return lines


def fetch_all_props_for_games(api_key: str, games: list, props: list = None) -> dict:
    """Fetch all player props for all games."""
    if props is None:
        props = ['pts', 'trb', 'ast']  # Main props only to save API calls
    
    all_lines = {}
    
    for game in games:
        event_id = game.get('id')
        home = game.get('home_team')
        away = game.get('away_team')
        
        print(f"\nFetching props for {away} @ {home}...")
        
        for prop in props:
            market = ODDS_API_MARKETS.get(prop)
            if not market:
                continue
            
            odds_data = fetch_player_props(api_key, event_id, market)
            lines = parse_prop_lines(odds_data, prop)
            
            for player, line_data in lines.items():
                if player not in all_lines:
                    all_lines[player] = {}
                all_lines[player][prop] = line_data
    
    return all_lines


# =============================================================================
# MANUAL LINE ENTRY
# =============================================================================

def manual_line_entry(df: pd.DataFrame, players: list) -> dict:
    """Interactive manual line entry."""
    print("\n" + "="*60)
    print("MANUAL LINE ENTRY")
    print("="*60)
    print("Enter betting lines for each player/prop.")
    print("Press Enter to skip, or 'q' to finish early.\n")
    
    all_lines = {}
    main_props = ['pts', 'trb', 'ast', 'pra']
    
    for player_info in players:
        player = player_info if isinstance(player_info, str) else player_info.get('player', '')
        exact_name = get_player_match(df, player)
        
        if exact_name is None:
            print(f"⚠ Player '{player}' not found, skipping")
            continue
        
        print(f"\n--- {exact_name} ---")
        all_lines[exact_name] = {}
        
        for prop in main_props:
            prop_name = PROP_DISPLAY_NAMES.get(prop, prop)
            line_input = input(f"  {prop_name} line (or Enter to skip): ").strip()
            
            if line_input.lower() == 'q':
                return all_lines
            
            if line_input:
                try:
                    line = float(line_input)
                    
                    # Optionally get odds
                    over_odds = input(f"    Over odds (default -110): ").strip()
                    under_odds = input(f"    Under odds (default -110): ").strip()
                    
                    all_lines[exact_name][prop] = {
                        'line': line,
                        'over_odds': int(over_odds) if over_odds else -110,
                        'under_odds': int(under_odds) if under_odds else -110,
                        'book': 'manual'
                    }
                except ValueError:
                    print(f"    ⚠ Invalid number, skipping")
    
    return all_lines


def load_lines_from_csv(filepath: str) -> dict:
    """
    Load betting lines from CSV file.
    
    Expected columns: player, prop, line, over_odds, under_odds
    """
    df = pd.read_csv(filepath)
    
    required = ['player', 'prop', 'line']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    
    all_lines = {}
    
    for _, row in df.iterrows():
        player = row['player']
        prop = row['prop'].lower()
        
        if player not in all_lines:
            all_lines[player] = {}
        
        all_lines[player][prop] = {
            'line': float(row['line']),
            'over_odds': int(row.get('over_odds', -110)),
            'under_odds': int(row.get('under_odds', -110)),
            'book': row.get('book', 'csv')
        }
    
    print(f"✓ Loaded lines for {len(all_lines)} players from {filepath}")
    return all_lines


# =============================================================================
# EDGE CALCULATION & BET RECOMMENDATIONS
# =============================================================================

def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1


def implied_probability(odds: int) -> float:
    """Calculate implied probability from American odds."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def calculate_edge(prediction: float, line: float, odds: int, direction: str) -> dict:
    """
    Calculate betting edge.
    
    Args:
        prediction: Our model's prediction
        line: Sportsbook line
        odds: American odds for this bet
        direction: 'over' or 'under'
    
    Returns:
        Dictionary with edge metrics
    """
    diff = prediction - line
    
    if direction == 'over':
        edge_direction = 'OVER' if diff > 0 else None
        edge_magnitude = diff if diff > 0 else 0
    else:  # under
        edge_direction = 'UNDER' if diff < 0 else None
        edge_magnitude = abs(diff) if diff < 0 else 0
    
    # Calculate implied probability and our estimated probability
    imp_prob = implied_probability(odds)
    
    # Simple edge estimate based on line difference
    # Rough approximation: each point of edge ≈ 2-3% probability shift
    prob_shift = edge_magnitude * 0.025  # Conservative estimate
    our_prob = 0.5 + prob_shift  # Assuming 50/50 at line
    
    # Edge = our probability - implied probability
    edge_pct = (our_prob - imp_prob) * 100 if edge_direction else 0
    
    # Kelly criterion bet sizing
    if edge_pct > 0:
        decimal_odds = american_to_decimal(odds)
        kelly_full = (our_prob * decimal_odds - 1) / (decimal_odds - 1)
        kelly_frac = kelly_full * KELLY_FRACTION
    else:
        kelly_frac = 0
    
    return {
        'direction': edge_direction,
        'edge_pts': edge_magnitude,
        'edge_pct': edge_pct,
        'our_prob': our_prob,
        'implied_prob': imp_prob,
        'kelly_pct': max(0, kelly_frac * 100),
        'odds': odds
    }


def analyze_bet(prediction: float, line_data: dict) -> dict:
    """Analyze both over and under for a prop."""
    line = line_data['line']
    over_odds = line_data.get('over_odds', -110)
    under_odds = line_data.get('under_odds', -110)
    
    over_edge = calculate_edge(prediction, line, over_odds, 'over')
    under_edge = calculate_edge(prediction, line, under_odds, 'under')
    
    # Return the better edge
    if over_edge['edge_pts'] > under_edge['edge_pts']:
        return {**over_edge, 'line': line, 'prediction': prediction}
    elif under_edge['edge_pts'] > 0:
        return {**under_edge, 'line': line, 'prediction': prediction}
    else:
        return {
            'direction': None,
            'edge_pts': 0,
            'edge_pct': 0,
            'line': line,
            'prediction': prediction,
            'our_prob': 0.5,
            'implied_prob': 0.5,
            'kelly_pct': 0,
            'odds': -110
        }


def get_bet_recommendation(edge_pts: float, edge_pct: float) -> str:
    """Get bet recommendation based on edge."""
    if edge_pts >= EDGE_THRESHOLDS['strong'] and edge_pct > 5:
        return "★★★ STRONG BET"
    elif edge_pts >= EDGE_THRESHOLDS['moderate'] and edge_pct > 2:
        return "★★ MODERATE BET"
    elif edge_pts >= EDGE_THRESHOLDS['slight'] and edge_pct > 0:
        return "★ SLIGHT EDGE"
    else:
        return "NO BET"


# =============================================================================
# OUTPUT & REPORTING
# =============================================================================

def print_player_analysis(player: str, predictions: dict, lines: dict):
    """Print analysis for a single player."""
    print(f"\n{'━'*70}")
    print(f"  {player}")
    print(f"{'━'*70}")
    
    print(f"\n  {'Prop':<12} {'Pred':>7} {'Line':>7} {'Edge':>7} {'Odds':>7} {'Rec':<18}")
    print(f"  {'─'*60}")
    
    bets = []
    
    for prop in PROPS:
        if prop not in predictions or prop not in lines:
            continue
        
        pred = predictions[prop]
        line_data = lines[prop]
        
        analysis = analyze_bet(pred, line_data)
        
        if analysis['direction']:
            rec = get_bet_recommendation(analysis['edge_pts'], analysis['edge_pct'])
            direction = analysis['direction']
            edge_str = f"+{analysis['edge_pts']:.1f}"
            odds_str = f"{analysis['odds']:+d}"
            
            print(f"  {PROP_DISPLAY_NAMES.get(prop, prop):<12} {pred:>7.1f} {analysis['line']:>7.1f} "
                  f"{edge_str:>7} {odds_str:>7} {direction:<6} {rec}")
            
            if "BET" in rec:
                bets.append({
                    'prop': prop,
                    'direction': direction,
                    'line': analysis['line'],
                    'prediction': pred,
                    'edge_pts': analysis['edge_pts'],
                    'edge_pct': analysis['edge_pct'],
                    'odds': analysis['odds'],
                    'kelly_pct': analysis['kelly_pct'],
                    'rec': rec
                })
        else:
            print(f"  {PROP_DISPLAY_NAMES.get(prop, prop):<12} {pred:>7.1f} {line_data['line']:>7.1f} "
                  f"{'—':>7} {'—':>7} {'—':<6} NO EDGE")
    
    return bets


def print_summary(all_bets: list, bankroll: float = None):
    """Print summary of all betting recommendations."""
    print(f"\n{'='*70}")
    print("BET RECOMMENDATIONS SUMMARY")
    print(f"{'='*70}")
    
    if not all_bets:
        print("\nNo +EV bets identified.")
        return
    
    # Sort by edge
    all_bets = sorted(all_bets, key=lambda x: -x['edge_pts'])
    
    # Strong bets
    strong = [b for b in all_bets if "STRONG" in b['rec']]
    moderate = [b for b in all_bets if "MODERATE" in b['rec']]
    slight = [b for b in all_bets if "SLIGHT" in b['rec']]
    
    if strong:
        print(f"\n★★★ STRONG BETS ({len(strong)}):")
        for b in strong:
            print(f"  {b['player']:<20} {PROP_DISPLAY_NAMES.get(b['prop'], b['prop']):<12} "
                  f"{b['direction']:<6} {b['line']:.1f} @ {b['odds']:+d}")
            print(f"      Prediction: {b['prediction']:.1f}, Edge: +{b['edge_pts']:.1f} pts")
            if bankroll and b['kelly_pct'] > 0:
                bet_size = bankroll * b['kelly_pct'] / 100
                print(f"      Suggested bet: ${bet_size:.0f} ({b['kelly_pct']:.1f}% of bankroll)")
    
    if moderate:
        print(f"\n★★ MODERATE BETS ({len(moderate)}):")
        for b in moderate:
            print(f"  {b['player']:<20} {PROP_DISPLAY_NAMES.get(b['prop'], b['prop']):<12} "
                  f"{b['direction']:<6} {b['line']:.1f} @ {b['odds']:+d}")
            print(f"      Prediction: {b['prediction']:.1f}, Edge: +{b['edge_pts']:.1f} pts")
    
    if slight:
        print(f"\n★ SLIGHT EDGES ({len(slight)}):")
        for b in slight[:5]:  # Top 5 only
            print(f"  {b['player']:<20} {PROP_DISPLAY_NAMES.get(b['prop'], b['prop']):<12} "
                  f"{b['direction']:<6} {b['line']:.1f} (edge: +{b['edge_pts']:.1f})")
    
    # Stats
    print(f"\n{'─'*70}")
    print(f"Total bets identified: {len(all_bets)}")
    print(f"  Strong: {len(strong)}, Moderate: {len(moderate)}, Slight: {len(slight)}")
    
    if bankroll:
        total_kelly = sum(b['kelly_pct'] for b in all_bets if b['kelly_pct'] > 0)
        print(f"\nTotal suggested allocation: {total_kelly:.1f}% of bankroll (${bankroll * total_kelly / 100:.0f})")


def save_bets_csv(all_bets: list, output_path: str):
    """Save bet recommendations to CSV."""
    if not all_bets:
        print("No bets to save.")
        return
    
    df = pd.DataFrame(all_bets)
    df['prop_name'] = df['prop'].map(PROP_DISPLAY_NAMES)
    df = df[['player', 'prop_name', 'direction', 'line', 'prediction', 
             'edge_pts', 'edge_pct', 'odds', 'kelly_pct', 'rec']]
    df = df.sort_values('edge_pts', ascending=False)
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df)} bet recommendations to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare predictions to betting lines and find +EV bets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using The Odds API
  export ODDS_API_KEY=your_key_here
  python task8_betting_lines.py --api

  # Manual line entry
  python task8_betting_lines.py --manual --players "LeBron James;Nikola Jokic"

  # From CSV file with lines
  python task8_betting_lines.py --input lines.csv

  # Save recommendations
  python task8_betting_lines.py --api --output bets.csv --bankroll 1000
        """
    )
    
    parser.add_argument('--api', action='store_true', help='Fetch lines from The Odds API')
    parser.add_argument('--manual', action='store_true', help='Manual line entry mode')
    parser.add_argument('--input', '-i', help='CSV file with betting lines')
    parser.add_argument('--players', '-p', help='Player names for manual entry (semicolon separated)')
    parser.add_argument('--output', '-o', help='Output CSV path for recommendations')
    parser.add_argument('--bankroll', type=float, help='Bankroll for Kelly bet sizing')
    parser.add_argument('--data', default='player_games_schedule.csv', help='Historical data file')
    parser.add_argument('--model-dir', default='.', help='Directory with model .pkl files')
    parser.add_argument('--props', help='Props to analyze (comma separated, default: pts,trb,ast,pra)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Task 8: Compare to Real Betting Lines")
    print("="*70)
    
    # Load models
    models = load_models(Path(args.model_dir))
    if not models:
        print("\n✗ No models found. Run task6_train_models.py first.")
        return
    
    # Load historical data
    df = load_historical_data(args.data)
    
    # Get betting lines
    all_lines = {}
    
    if args.api:
        api_key = get_odds_api_key()
        if api_key:
            games = fetch_nba_games(api_key)
            props_to_fetch = args.props.split(',') if args.props else ['pts', 'trb', 'ast']
            all_lines = fetch_all_props_for_games(api_key, games, props_to_fetch)
    
    elif args.input:
        all_lines = load_lines_from_csv(args.input)
    
    elif args.manual:
        if args.players:
            players = [p.strip() for p in args.players.split(';')]
        else:
            print("Enter player names (one per line, empty line to finish):")
            players = []
            while True:
                p = input().strip()
                if not p:
                    break
                players.append(p)
        
        all_lines = manual_line_entry(df, players)
    
    else:
        print("\nNo input mode specified. Use --api, --manual, or --input")
        print("Run with --help for usage examples.")
        return
    
    if not all_lines:
        print("\nNo betting lines available.")
        return
    
    # Analyze each player
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    all_bets = []
    
    for player, lines in all_lines.items():
        # Get predictions for this player
        predictions = get_all_predictions(df, models, player)
        
        if not predictions:
            print(f"\n⚠ Could not generate predictions for {player}")
            continue
        
        # Analyze bets
        bets = print_player_analysis(player, predictions, lines)
        
        for bet in bets:
            bet['player'] = player
            all_bets.append(bet)
    
    # Summary
    print_summary(all_bets, args.bankroll)
    
    # Save if output specified
    if args.output:
        save_bets_csv(all_bets, args.output)
    
    print("\n" + "="*70)
    print("DISCLAIMER: This is for educational purposes only.")
    print("Gambling involves risk. Never bet more than you can afford to lose.")
    print("="*70)


if __name__ == "__main__":
    main()
