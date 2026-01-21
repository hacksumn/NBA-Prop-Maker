"""
GET REAL VEGAS LINES - The Odds API
NBA Player Props Betting System

Fetches ACTUAL sportsbook lines to compare against model predictions.
This is the ONLY way to know if the model actually works.

Usage:
    python get_vegas_lines.py
"""

import requests
import pandas as pd
import json
from datetime import datetime
import os

# Your API Key
API_KEY = "8269e5e566cd8d8f8c294b85244ffab7"
BASE_URL = "https://api.the-odds-api.com/v4"

# Prop markets to fetch
PROP_MARKETS = [
    'player_points',
    'player_rebounds', 
    'player_assists',
    'player_points_rebounds_assists',
    'player_points_rebounds',
    'player_points_assists',
    'player_steals',
    'player_blocks',
    'player_turnovers'
]


def check_api_usage():
    """Check remaining API requests"""
    url = f"{BASE_URL}/sports?apiKey={API_KEY}"
    response = requests.get(url)
    remaining = response.headers.get('x-requests-remaining', 'Unknown')
    used = response.headers.get('x-requests-used', 'Unknown')
    print(f"API Usage: {used} used, {remaining} remaining")
    return response.status_code == 200


def get_todays_games():
    """Get today's NBA games"""
    url = f"{BASE_URL}/sports/basketball_nba/events"
    params = {'apiKey': API_KEY}
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return []
    
    events = response.json()
    print(f"\nFound {len(events)} upcoming NBA games:")
    for event in events:
        print(f"  {event['away_team']} @ {event['home_team']} - {event['commence_time']}")
    
    return events


def get_player_props(event_id, market='player_points'):
    """Get player props for a specific game and market"""
    url = f"{BASE_URL}/sports/basketball_nba/events/{event_id}/odds"
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': market,
        'oddsFormat': 'american'
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        return []
    
    return response.json()


def parse_player_props(odds_data, market_name):
    """Parse odds data into clean DataFrame rows"""
    props = []
    
    if not odds_data or 'bookmakers' not in odds_data:
        return props
    
    event_id = odds_data.get('id', '')
    home_team = odds_data.get('home_team', '')
    away_team = odds_data.get('away_team', '')
    commence_time = odds_data.get('commence_time', '')
    
    for bookmaker in odds_data.get('bookmakers', []):
        book_name = bookmaker.get('key', '')
        
        for market in bookmaker.get('markets', []):
            if market.get('key') != market_name:
                continue
                
            for outcome in market.get('outcomes', []):
                player = outcome.get('description', '')
                name = outcome.get('name', '')  # Over or Under
                line = outcome.get('point', 0)
                odds = outcome.get('price', 0)
                
                props.append({
                    'event_id': event_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'commence_time': commence_time,
                    'bookmaker': book_name,
                    'market': market_name,
                    'player': player,
                    'direction': name,
                    'line': line,
                    'odds': odds,
                    'timestamp': datetime.now().isoformat()
                })
    
    return props


def fetch_all_props():
    """Fetch all player props for today's games"""
    print("="*60)
    print("FETCHING REAL VEGAS LINES")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check API
    if not check_api_usage():
        print("API check failed!")
        return None
    
    # Get games
    games = get_todays_games()
    if not games:
        print("No games found!")
        return None
    
    # Fetch props for each game and market
    all_props = []
    
    for game in games:
        event_id = game['id']
        print(f"\nFetching props for: {game['away_team']} @ {game['home_team']}")
        
        for market in PROP_MARKETS:
            print(f"  - {market}...", end=" ")
            odds_data = get_player_props(event_id, market)
            props = parse_player_props(odds_data, market)
            all_props.extend(props)
            print(f"{len(props)} lines")
            
            # Small delay to be nice to API
            import time
            time.sleep(0.5)
    
    # Create DataFrame
    df = pd.DataFrame(all_props)
    
    if len(df) == 0:
        print("\nNo props found!")
        return None
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {len(df)} lines fetched")
    print(f"Players: {df['player'].nunique()}")
    print(f"Markets: {df['market'].unique().tolist()}")
    
    return df


def consolidate_lines(df):
    """
    Consolidate multiple bookmaker lines into consensus.
    Take average line across books, best odds for over/under.
    """
    # Get unique player/market combinations
    consolidated = []
    
    for (player, market), group in df.groupby(['player', 'market']):
        # Get over lines
        overs = group[group['direction'] == 'Over']
        unders = group[group['direction'] == 'Under']
        
        if len(overs) == 0 or len(unders) == 0:
            continue
        
        # Consensus line (average)
        avg_line = group['line'].mean()
        
        # Best odds
        best_over_odds = overs['odds'].max()
        best_under_odds = unders['odds'].max()
        
        consolidated.append({
            'player': player,
            'market': market,
            'line': round(avg_line, 1),
            'over_odds': best_over_odds,
            'under_odds': best_under_odds,
            'num_books': group['bookmaker'].nunique()
        })
    
    return pd.DataFrame(consolidated)


def compare_to_model(vegas_df, model_predictions_file='bets_TONIGHT_CLEAN.csv'):
    """Compare Vegas lines to model predictions"""
    
    if not os.path.exists(model_predictions_file):
        print(f"\nModel predictions file not found: {model_predictions_file}")
        return None
    
    model_df = pd.read_csv(model_predictions_file)
    
    # Map market names
    market_map = {
        'player_points': 'Points',
        'player_rebounds': 'Rebounds',
        'player_assists': 'Assists',
        'player_points_rebounds_assists': 'Pts+Reb+Ast',
        'player_points_rebounds': 'Pts+Reb',
        'player_points_assists': 'Pts+Ast',
        'player_steals': 'Steals',
        'player_blocks': 'Blocks',
        'player_turnovers': 'Turnovers'
    }
    
    vegas_df['prop_display'] = vegas_df['market'].map(market_map)
    
    # Merge
    comparison = model_df.merge(
        vegas_df[['player', 'prop_display', 'line', 'over_odds', 'under_odds']],
        on=['player', 'prop_display'],
        how='inner',
        suffixes=('_model', '_vegas')
    )
    
    if len(comparison) == 0:
        print("\nNo matching players/props found between model and Vegas!")
        return None
    
    # Calculate real edge
    comparison['vegas_line'] = comparison['line_vegas']
    comparison['model_line'] = comparison['line_model']  # This is actually L10, rename for clarity
    comparison['real_edge'] = comparison['prediction'] - comparison['vegas_line']
    comparison['real_direction'] = comparison['real_edge'].apply(lambda x: 'OVER' if x > 0 else 'UNDER')
    comparison['real_edge_abs'] = comparison['real_edge'].abs()
    
    # Get appropriate odds
    comparison['bet_odds'] = comparison.apply(
        lambda x: x['over_odds'] if x['real_direction'] == 'OVER' else x['under_odds'],
        axis=1
    )
    
    return comparison


def main():
    # Fetch Vegas lines
    raw_df = fetch_all_props()
    
    if raw_df is None:
        return
    
    # Save raw data
    raw_df.to_csv('vegas_lines_raw.csv', index=False)
    print(f"\n✓ Saved: vegas_lines_raw.csv")
    
    # Consolidate
    consolidated = consolidate_lines(raw_df)
    consolidated.to_csv('vegas_lines_consolidated.csv', index=False)
    print(f"✓ Saved: vegas_lines_consolidated.csv")
    
    # Compare to model
    print("\n" + "="*60)
    print("COMPARING TO MODEL PREDICTIONS")
    print("="*60)
    
    comparison = compare_to_model(consolidated)
    
    if comparison is not None:
        # Sort by real edge
        comparison = comparison.sort_values('real_edge_abs', ascending=False)
        
        print(f"\nMatched {len(comparison)} predictions to Vegas lines!")
        print("\nTOP REAL-EDGE PLAYS:")
        print(comparison[['player', 'prop_display', 'vegas_line', 'prediction', 'real_edge', 'bet_odds']].head(20).to_string())
        
        # Save
        comparison.to_csv('model_vs_vegas.csv', index=False)
        print(f"\n✓ Saved: model_vs_vegas.csv")
        
        # Summary
        print("\n" + "="*60)
        print("EDGE DISTRIBUTION (vs REAL VEGAS LINES)")
        print("="*60)
        for low, high in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 99)]:
            count = len(comparison[(comparison['real_edge_abs'] >= low) & (comparison['real_edge_abs'] < high)])
            print(f"  Edge {low}-{high}: {count} bets")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
