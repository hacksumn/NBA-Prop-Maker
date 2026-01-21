"""
Generate Picks Using Classification Model + Real Vegas Lines
NBA Player Props v2

Strategy:
1. Fetch real Vegas lines from The Odds API
2. For each line, calculate features
3. Run classification model to get probability
4. Only bet when confidence > 60%
"""

import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# API Configuration
ODDS_API_KEY = "8269e5e566cd8d8f8c294b85244ffab7"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Confidence thresholds
MIN_CONFIDENCE = 0.60  # Only bet when 60%+ confident
HIGH_CONFIDENCE = 0.65  # Flag these as strongest plays


def load_player_features():
    """Load latest player features including matchup history."""
    print("Loading player features...")
    
    # Try matchup-enhanced data first
    try:
        df = pd.read_csv('player_games_with_matchups.csv', parse_dates=['game_date'])
        has_matchups = 'pts_matchup_edge' in df.columns
        print(f"  Loaded with matchup features: {'✓' if has_matchups else '✗'}")
    except:
        df = pd.read_csv('player_games_schedule.csv', parse_dates=['game_date'])
        print(f"  Loaded base features (no matchup history)")
    
    # Get most recent features for each player
    latest = df.sort_values('game_date').groupby('player').last().reset_index()
    print(f"  {len(latest)} players with features")
    
    return latest


def load_classifiers():
    """Load trained classification models."""
    print("Loading classification models...")
    models = {}
    
    props = ['pts', 'trb', 'ast', 'pra', 'pr', 'pa']
    for prop in props:
        try:
            with open(f'classifier_{prop}.pkl', 'rb') as f:
                models[prop] = pickle.load(f)
            print(f"  ✓ {prop.upper()}")
        except FileNotFoundError:
            print(f"  ✗ {prop} not found")
    
    return models


def fetch_vegas_lines():
    """Fetch real Vegas lines from The Odds API."""
    print("\nFetching Vegas lines...")
    
    # Map of prop markets
    markets = [
        'player_points',
        'player_rebounds', 
        'player_assists',
        'player_points_rebounds_assists',
        'player_points_rebounds',
        'player_points_assists'
    ]
    
    all_lines = []
    
    # Get today's NBA games
    games_url = f"{ODDS_API_BASE}/sports/basketball_nba/events"
    params = {'apiKey': ODDS_API_KEY}
    
    try:
        resp = requests.get(games_url, params=params)
        games = resp.json()
        print(f"  Found {len(games)} games")
    except Exception as e:
        print(f"  Error fetching games: {e}")
        return pd.DataFrame()
    
    # Get props for each game
    for game in games:
        game_id = game['id']
        game_time = game.get('commence_time', '')
        
        for market in markets:
            props_url = f"{ODDS_API_BASE}/sports/basketball_nba/events/{game_id}/odds"
            params = {
                'apiKey': ODDS_API_KEY,
                'regions': 'us',
                'markets': market,
                'oddsFormat': 'american'
            }
            
            try:
                resp = requests.get(props_url, params=params)
                if resp.status_code != 200:
                    continue
                    
                data = resp.json()
                
                for book in data.get('bookmakers', []):
                    for mkt in book.get('markets', []):
                        for outcome in mkt.get('outcomes', []):
                            player = outcome.get('description', '')
                            line = outcome.get('point', 0)
                            over_under = outcome.get('name', '')
                            odds = outcome.get('price', -110)
                            
                            if over_under == 'Over' and player and line:
                                all_lines.append({
                                    'player': player,
                                    'market': market,
                                    'line': line,
                                    'odds': odds,
                                    'book': book['key'],
                                    'game_time': game_time,
                                    'home': game.get('home_team', ''),
                                    'away': game.get('away_team', '')
                                })
            except:
                continue
    
    if not all_lines:
        print("  No lines found!")
        return pd.DataFrame()
    
    lines_df = pd.DataFrame(all_lines)
    
    # Get best line for each player/market
    best_lines = lines_df.groupby(['player', 'market']).agg({
        'line': 'mean',  # Average across books
        'odds': 'max',   # Best odds
        'game_time': 'first',
        'home': 'first',
        'away': 'first'
    }).reset_index()
    
    print(f"  Got {len(best_lines)} unique player props")
    
    return best_lines


def create_features_for_prediction(player_features, vegas_line, prop, matchup_history=None):
    """
    Create features for classification model prediction.
    
    Must match the features used in training (with matchup features).
    """
    # Map market names to our prop names
    market_to_prop = {
        'player_points': 'pts',
        'player_rebounds': 'trb',
        'player_assists': 'ast',
        'player_points_rebounds_assists': 'pra',
        'player_points_rebounds': 'pr',
        'player_points_assists': 'pa'
    }
    
    prop_name = market_to_prop.get(vegas_line['market'], prop)
    
    # Get player's rolling stats
    if prop_name in ['pts', 'trb', 'ast']:
        l5 = player_features.get(f'{prop_name}_l5', np.nan)
        l10 = player_features.get(f'{prop_name}_l10', np.nan)
        l20 = player_features.get(f'{prop_name}_l20', np.nan)
        variance = player_features.get(f'{prop_name}_std_l10', 5.0)
        # Matchup columns
        matchup_edge_col = f'{prop_name}_matchup_edge'
        recent_matchup_edge_col = f'{prop_name}_recent_matchup_edge'
        favorable_col = f'{prop_name}_favorable_matchup'
        unfavorable_col = f'{prop_name}_unfavorable_matchup'
    else:
        l5 = player_features.get('pts_l5', np.nan)
        l10 = player_features.get('pts_l10', np.nan)
        l20 = player_features.get('pts_l20', np.nan)
        variance = player_features.get('pts_std_l10', 5.0)
        # Use pts matchup for combo props
        matchup_edge_col = 'pts_matchup_edge'
        recent_matchup_edge_col = 'pts_recent_matchup_edge'
        favorable_col = 'pts_favorable_matchup'
        unfavorable_col = 'pts_unfavorable_matchup'
    
    line = vegas_line['line']
    
    # Fill NaN with defaults
    l5 = l5 if pd.notna(l5) else line
    l10 = l10 if pd.notna(l10) else line
    l20 = l20 if pd.notna(l20) else line
    variance = variance if pd.notna(variance) else 5.0
    
    # Base features
    avg_vs_line = l10 - line
    l5_vs_line = l5 - line
    l20_vs_line = l20 - line
    trend = l5_vs_line - l20_vs_line
    recent_vs_season = l5 / (l20 + 0.1)
    
    # Schedule features
    is_b2b = player_features.get('is_b2b_second', 0) or 0
    high_mins = player_features.get('high_min_prev', 0) or 0
    days_rest = player_features.get('days_rest', 1) or 1
    games_density = (player_features.get('games_last_7d', 2) or 2) / 4
    mins_load = (player_features.get('mp_l10', 25) or 25) / 36
    
    # Matchup (team defense)
    opp_rank = player_features.get('opp_pts_allowed_rank', 15) or 15
    easy_matchup = opp_rank / 30
    
    # PLAYER VS OPPONENT MATCHUP HISTORY
    if matchup_history:
        matchup_edge = matchup_history.get('matchup_edge', 0)
        recent_matchup_edge = matchup_history.get('recent_matchup_edge', matchup_edge)
        matchup_games = matchup_history.get('games', 0)
        favorable_vs_opp = matchup_history.get('favorable', 0)
        unfavorable_vs_opp = matchup_history.get('unfavorable', 0)
    else:
        matchup_edge = player_features.get(matchup_edge_col, 0) or 0
        recent_matchup_edge = player_features.get(recent_matchup_edge_col, matchup_edge) or matchup_edge
        matchup_games = player_features.get('n_prev_matchups', 0) or 0
        favorable_vs_opp = player_features.get(favorable_col, 0) or 0
        unfavorable_vs_opp = player_features.get(unfavorable_col, 0) or 0
    
    has_matchup_history = 1 if matchup_games >= 2 else 0
    strong_matchup_history = 1 if matchup_games >= 3 else 0
    matchup_edge_vs_line = matchup_edge - avg_vs_line
    strong_matchup_over = 1 if (strong_matchup_history and matchup_edge > 3) else 0
    strong_matchup_under = 1 if (strong_matchup_history and matchup_edge < -3) else 0
    well_rested = 1 if days_rest >= 2 else 0
    
    # Build feature dict matching training order
    features = {
        # Base features
        'avg_vs_line': avg_vs_line,
        'l5_vs_line': l5_vs_line,
        'l20_vs_line': l20_vs_line,
        'variance': variance,
        'line_percentile': avg_vs_line / (variance + 0.1),
        'trend': trend,
        'recent_vs_season': recent_vs_season,
        # Schedule
        'is_b2b': is_b2b,
        'high_mins': high_mins,
        'fatigue_risk': is_b2b * high_mins,
        'days_rest_capped': min(max(days_rest, 0), 5),
        'well_rested': well_rested,
        # Matchup (team defense)
        'easy_matchup': easy_matchup,
        'hard_matchup': 1 - easy_matchup,
        # PLAYER VS OPPONENT HISTORY
        'matchup_edge': matchup_edge,
        'recent_matchup_edge': recent_matchup_edge,
        'has_matchup_history': has_matchup_history,
        'strong_matchup_history': strong_matchup_history,
        'matchup_games': matchup_games,
        'matchup_edge_vs_line': matchup_edge_vs_line,
        'strong_matchup_over': strong_matchup_over,
        'strong_matchup_under': strong_matchup_under,
        'favorable_vs_opp': favorable_vs_opp,
        'unfavorable_vs_opp': unfavorable_vs_opp,
        # Interactions
        'fatigue_x_cold': is_b2b * (1 if trend < 0 else 0),
        'fatigue_x_hot': is_b2b * (1 if trend > 0 else 0),
        'easy_x_hot': easy_matchup * (1 if recent_vs_season > 1 else 0),
        'hard_x_cold': (1 - easy_matchup) * (1 if recent_vs_season < 1 else 0),
        'good_matchup_x_hot': strong_matchup_over * (1 if trend > 0 else 0),
        'bad_matchup_x_cold': strong_matchup_under * (1 if trend < 0 else 0),
        'good_matchup_rested': strong_matchup_over * well_rested,
        'bad_matchup_tired': strong_matchup_under * is_b2b,
        'edge_reliability': avg_vs_line / (variance + 1),
        'matchup_reliability': matchup_edge / (variance + 1),
        'line_extreme_low': 1 if (line - l20) < -3 else 0,
        'line_extreme_high': 1 if (line - l20) > 3 else 0,
        'is_consistent': 1 if variance < 5 else 0,
        'consistent_above_line': (1 if variance < 5 else 0) * (1 if avg_vs_line > 0 else 0),
        # Usage/load
        'mins_load': mins_load,
        'high_usage': 1 if mins_load > 0.9 else 0,
        'games_density': games_density,
        'heavy_schedule': 1 if games_density > 0.75 else 0,
        'compound_fatigue': is_b2b * (1 if games_density > 0.75 else 0) * (1 if mins_load > 0.9 else 0),
        # Non-linear
        'trend_sq': trend ** 2,
        'variance_sq': variance ** 2,
        'edge_sq': avg_vs_line ** 2,
        'matchup_edge_sq': matchup_edge ** 2
    }
    
    return features


def generate_picks(player_features_df, vegas_lines, models):
    """Generate picks using classification models."""
    print("\nGenerating picks...")
    
    picks = []
    
    # Map market to model prop
    market_to_prop = {
        'player_points': 'pts',
        'player_rebounds': 'trb',
        'player_assists': 'ast',
        'player_points_rebounds_assists': 'pra',
        'player_points_rebounds': 'pr',
        'player_points_assists': 'pa'
    }
    
    for _, line in vegas_lines.iterrows():
        player = line['player']
        market = line['market']
        prop = market_to_prop.get(market)
        
        if prop not in models:
            continue
        
        # Find player features
        player_row = player_features_df[player_features_df['player'] == player]
        
        if len(player_row) == 0:
            # Try fuzzy matching
            matches = player_features_df[player_features_df['player'].str.contains(player.split()[-1], case=False, na=False)]
            if len(matches) == 1:
                player_row = matches
            else:
                continue
        
        player_features = player_row.iloc[0].to_dict()
        
        # Create features
        features = create_features_for_prediction(player_features, line, prop)
        
        # Convert to array in correct order (must match training)
        feature_order = [
            # Base features
            'avg_vs_line', 'l5_vs_line', 'l20_vs_line',
            'variance', 'line_percentile', 'trend', 'recent_vs_season',
            # Schedule
            'is_b2b', 'high_mins', 'fatigue_risk', 'days_rest_capped', 'well_rested',
            # Matchup (team defense)
            'easy_matchup', 'hard_matchup',
            # PLAYER VS OPPONENT HISTORY
            'matchup_edge', 'has_matchup_history', 'matchup_games',
            'matchup_edge_vs_line', 'strong_matchup_over', 'strong_matchup_under',
            # Interactions
            'fatigue_x_cold', 'fatigue_x_hot',
            'easy_x_hot', 'hard_x_cold',
            'good_matchup_x_hot', 'bad_matchup_x_cold',
            'edge_reliability',
            'line_extreme_low', 'line_extreme_high',
            'is_consistent', 'consistent_above_line',
            # Usage/load
            'mins_load', 'high_usage', 'games_density', 'heavy_schedule',
            'compound_fatigue',
            # Non-linear
            'trend_sq', 'variance_sq', 'edge_sq', 'matchup_edge_sq'
        ]
        
        X = np.array([[features[f] for f in feature_order]])
        
        # Scale features
        model_bundle = models[prop]
        scaler = model_bundle['scaler']
        model = model_bundle['model']
        
        try:
            X_scaled = scaler.transform(X)
            prob_over = model.predict_proba(X_scaled)[0][1]
        except Exception as e:
            continue
        
        # Determine bet direction and confidence
        if prob_over >= MIN_CONFIDENCE:
            direction = 'OVER'
            confidence = prob_over
        elif prob_over <= (1 - MIN_CONFIDENCE):
            direction = 'UNDER'
            confidence = 1 - prob_over
        else:
            continue  # Not confident enough
        
        picks.append({
            'player': player,
            'prop': prop.upper(),
            'market': market,
            'direction': direction,
            'line': line['line'],
            'confidence': confidence,
            'prob_over': prob_over,
            'odds': line['odds'],
            'game_time': line['game_time'],
            'l10_avg': player_features.get(f'{prop}_l10', player_features.get('pts_l10', 0)),
            'l5_avg': player_features.get(f'{prop}_l5', player_features.get('pts_l5', 0)),
            'variance': features['variance']
        })
    
    picks_df = pd.DataFrame(picks)
    
    if len(picks_df) > 0:
        picks_df = picks_df.sort_values('confidence', ascending=False)
    
    return picks_df


def display_picks(picks_df):
    """Display picks nicely formatted."""
    if len(picks_df) == 0:
        print("\nNo picks meet confidence threshold!")
        return
    
    # High confidence picks
    high_conf = picks_df[picks_df['confidence'] >= HIGH_CONFIDENCE]
    med_conf = picks_df[(picks_df['confidence'] >= MIN_CONFIDENCE) & (picks_df['confidence'] < HIGH_CONFIDENCE)]
    
    print(f"\n{'='*70}")
    print(f"  CLASSIFICATION MODEL PICKS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")
    
    if len(high_conf) > 0:
        print(f"\n⭐⭐⭐ HIGH CONFIDENCE ({len(high_conf)} picks) - 65%+ probability")
        print("-"*70)
        for _, pick in high_conf.iterrows():
            arrow = "🔼" if pick['direction'] == 'OVER' else "🔽"
            print(f"  {arrow} {pick['player']:<22} {pick['prop']:<6} {pick['direction']:<5} "
                  f"{pick['line']:<6} (conf: {pick['confidence']:.1%}, odds: {pick['odds']:+d})")
            print(f"     L5: {pick['l5_avg']:.1f}, L10: {pick['l10_avg']:.1f}, "
                  f"Var: {pick['variance']:.1f}")
    
    if len(med_conf) > 0:
        print(f"\n⭐⭐ MEDIUM CONFIDENCE ({len(med_conf)} picks) - 60-65% probability")
        print("-"*70)
        for _, pick in med_conf.head(15).iterrows():
            arrow = "🔼" if pick['direction'] == 'OVER' else "🔽"
            print(f"  {arrow} {pick['player']:<22} {pick['prop']:<6} {pick['direction']:<5} "
                  f"{pick['line']:<6} (conf: {pick['confidence']:.1%}, odds: {pick['odds']:+d})")
    
    # Summary stats
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Total picks: {len(picks_df)}")
    print(f"  High confidence (65%+): {len(high_conf)}")
    print(f"  Medium confidence (60-65%): {len(med_conf)}")
    
    # Expected value calculation
    if len(picks_df) > 0:
        avg_conf = picks_df['confidence'].mean()
        expected_win_rate = avg_conf
        # At -110 odds, break even is 52.38%
        if expected_win_rate > 0.5238:
            expected_roi = (expected_win_rate * 1.0 - (1 - expected_win_rate) * 1.1) * 100
            print(f"  Average confidence: {avg_conf:.1%}")
            print(f"  Expected ROI: {expected_roi:+.1f}% (if calibrated correctly)")
        else:
            print(f"  Average confidence: {avg_conf:.1%} (near break-even)")


def save_picks(picks_df):
    """Save picks to CSV."""
    if len(picks_df) == 0:
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'picks_v2_{timestamp}.csv'
    picks_df.to_csv(filename, index=False)
    print(f"\n✓ Saved: {filename}")
    
    # Also save as latest
    picks_df.to_csv('picks_v2_latest.csv', index=False)
    print(f"✓ Saved: picks_v2_latest.csv")


def main():
    print("="*70)
    print("  NBA PROPS v2 - Classification Model Picks")
    print("="*70)
    print(f"\nStrategy: Only bet when model is {MIN_CONFIDENCE:.0%}+ confident")
    print(f"High confidence threshold: {HIGH_CONFIDENCE:.0%}+\n")
    
    # Load player features
    player_features = load_player_features()
    
    # Load models
    models = load_classifiers()
    
    if not models:
        print("ERROR: No models loaded!")
        return
    
    # Fetch real Vegas lines
    vegas_lines = fetch_vegas_lines()
    
    if len(vegas_lines) == 0:
        print("ERROR: No Vegas lines fetched!")
        return
    
    # Generate picks
    picks = generate_picks(player_features, vegas_lines, models)
    
    # Display
    display_picks(picks)
    
    # Save
    save_picks(picks)


if __name__ == "__main__":
    main()
