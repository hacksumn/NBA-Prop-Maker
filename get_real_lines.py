"""
REAL BETTING LINES FETCHER
Gets actual player prop lines from sportsbooks via The Odds API

FREE API KEY: https://the-odds-api.com/ (500 requests/month free)

Usage:
    python get_real_lines.py --api-key YOUR_KEY
    python get_real_lines.py  # Uses ODDS_API_KEY environment variable
"""

import requests
import pandas as pd
import argparse
import os
from datetime import datetime
import json

# The Odds API endpoints
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# Player prop markets we care about
PROP_MARKETS = [
    'player_points',
    'player_rebounds', 
    'player_assists',
    'player_steals',
    'player_blocks',
    'player_turnovers',
    'player_points_rebounds_assists',
    'player_points_rebounds',
    'player_points_assists',
]

PROP_MAP = {
    'player_points': 'pts',
    'player_rebounds': 'trb',
    'player_assists': 'ast',
    'player_steals': 'stl',
    'player_blocks': 'blk',
    'player_turnovers': 'tov',
    'player_points_rebounds_assists': 'pra',
    'player_points_rebounds': 'pr',
    'player_points_assists': 'pa',
}


def get_events(api_key: str) -> list:
    """Get upcoming NBA games."""
    url = f"{BASE_URL}/sports/{SPORT}/events"
    params = {
        'apiKey': api_key,
        'dateFormat': 'iso',
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    events = response.json()
    print(f"Found {len(events)} upcoming NBA games")
    
    # Show the games
    for event in events[:10]:
        game_time = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
        print(f"  {event['away_team']} @ {event['home_team']} - {game_time.strftime('%m/%d %I:%M %p')}")
    
    return events


def get_player_props(api_key: str, event_id: str, markets: list) -> list:
    """Get player props for a specific game."""
    url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': ','.join(markets),
        'oddsFormat': 'american',
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    return response.json()


def parse_props(odds_data: dict) -> list:
    """Parse player props from odds data."""
    lines = []
    
    bookmakers = odds_data.get('bookmakers', [])
    
    for book in bookmakers:
        book_name = book.get('title', 'Unknown')
        
        for market in book.get('markets', []):
            market_key = market.get('key', '')
            prop_type = PROP_MAP.get(market_key)
            
            if not prop_type:
                continue
            
            for outcome in market.get('outcomes', []):
                player = outcome.get('description', '')
                line = outcome.get('point')
                price = outcome.get('price')
                over_under = outcome.get('name', '').lower()
                
                if player and line is not None:
                    lines.append({
                        'player': player,
                        'prop': prop_type,
                        'line': line,
                        'over_odds' if over_under == 'over' else 'under_odds': price,
                        'book': book_name,
                    })
    
    return lines


def consolidate_lines(all_lines: list) -> pd.DataFrame:
    """Consolidate lines, combining over/under odds."""
    # Group by player, prop, line, book
    df = pd.DataFrame(all_lines)
    
    if len(df) == 0:
        return df
    
    # Pivot to get over and under odds in same row
    consolidated = []
    
    for (player, prop, line, book), group in df.groupby(['player', 'prop', 'line', 'book']):
        row = {
            'player': player,
            'prop': prop,
            'line': line,
            'book': book,
            'over_odds': None,
            'under_odds': None,
        }
        
        for _, r in group.iterrows():
            if 'over_odds' in r and pd.notna(r.get('over_odds')):
                row['over_odds'] = r['over_odds']
            if 'under_odds' in r and pd.notna(r.get('under_odds')):
                row['under_odds'] = r['under_odds']
        
        consolidated.append(row)
    
    return pd.DataFrame(consolidated)


def get_best_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Get best available line for each player/prop."""
    if len(df) == 0:
        return df
    
    best = []
    
    for (player, prop), group in df.groupby(['player', 'prop']):
        # Get the most common line (consensus)
        consensus_line = group['line'].mode().iloc[0] if len(group['line'].mode()) > 0 else group['line'].iloc[0]
        
        # Get best odds for this line
        at_line = group[group['line'] == consensus_line]
        
        if len(at_line) > 0:
            # Find best over and under odds
            best_over = at_line['over_odds'].max() if at_line['over_odds'].notna().any() else -110
            best_under = at_line['under_odds'].max() if at_line['under_odds'].notna().any() else -110
            
            best.append({
                'player': player,
                'prop': prop,
                'line': consensus_line,
                'over_odds': int(best_over) if pd.notna(best_over) else -110,
                'under_odds': int(best_under) if pd.notna(best_under) else -110,
                'book': 'Best Available',
            })
    
    return pd.DataFrame(best)


def main():
    parser = argparse.ArgumentParser(description='Fetch real NBA player prop betting lines')
    parser.add_argument('--api-key', help='The Odds API key (or set ODDS_API_KEY env var)')
    parser.add_argument('--output', default='real_lines.csv', help='Output CSV file')
    parser.add_argument('--all-books', action='store_true', help='Include all sportsbooks (not just best)')
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get('ODDS_API_KEY')
    
    if not api_key:
        print("="*60)
        print("ERROR: No API key provided!")
        print("="*60)
        print("\nGet a FREE API key at: https://the-odds-api.com/")
        print("  - 500 requests/month free")
        print("  - Sign up takes 30 seconds")
        print("\nThen run:")
        print(f"  python {__file__} --api-key YOUR_API_KEY")
        print("\nOr set environment variable:")
        print("  $env:ODDS_API_KEY = 'YOUR_API_KEY'  # PowerShell")
        return
    
    print("="*60)
    print("FETCHING REAL BETTING LINES")
    print("="*60)
    
    # Get upcoming games (next 24-48 hours)
    print("\nStep 1: Getting upcoming games...")
    try:
        events = get_events(api_key)
    except requests.exceptions.HTTPError as e:
        print(f"API Error: {e}")
        print("Check your API key is valid")
        return
    
    if not events:
        print("No upcoming NBA games found!")
        return
    
    # Use first 5 games (or all if less)
    events_to_use = events[:5]
    print(f"Using {len(events_to_use)} games for prop lines")
    
    # Get props for each game
    print(f"\nStep 2: Fetching player props...")
    all_lines = []
    
    for event in events_to_use:
        print(f"  {event['away_team']} @ {event['home_team']}...")
        try:
            odds_data = get_player_props(api_key, event['id'], PROP_MARKETS)
            lines = parse_props(odds_data)
            all_lines.extend(lines)
            print(f"    Found {len(lines)} prop lines")
        except Exception as e:
            print(f"    Error: {e}")
    
    if not all_lines:
        print("\nNo player props found!")
        print("Props may not be available yet (usually posted 1-2 hours before games)")
        return
    
    # Process lines
    print(f"\nStep 3: Processing {len(all_lines)} lines...")
    df = consolidate_lines(all_lines)
    
    if not args.all_books:
        df = get_best_lines(df)
    
    # Save
    df.to_csv(args.output, index=False)
    
    print(f"\n{'='*60}")
    print("LINES SAVED")
    print(f"{'='*60}")
    print(f"File: {args.output}")
    print(f"Players: {df['player'].nunique()}")
    print(f"Total lines: {len(df)}")
    
    # Show sample
    print(f"\nSample lines:")
    print(df.head(15).to_string(index=False))
    
    print(f"\nNow run:")
    print(f"  python task8_betting_lines.py --input {args.output}")


if __name__ == "__main__":
    main()
