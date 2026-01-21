"""
Task 10: Deployment Pipeline
NBA Player Props Betting System

This script automates the daily prediction workflow:
1. Scrape/update today's games and player data
2. Generate predictions for all players
3. Compare to betting lines (if available)
4. Send alerts for high-edge opportunities
5. Log everything for tracking

Can be run manually or scheduled via cron/Task Scheduler.

Usage:
  python task10_deployment.py                    # Full daily run
  python task10_deployment.py --dry-run          # Test without saving
  python task10_deployment.py --email alerts@example.com  # With email alerts
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    HAS_EMAIL = True
except ImportError:
    HAS_EMAIL = False


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Betting thresholds (from backtest findings)
    'min_edge': 1.5,              # Minimum edge to recommend
    'min_minutes': 15,            # Minimum avg minutes to include player
    'strong_edge': 3.0,           # Edge for "strong" recommendation
    
    # Props to predict (exclude blocks based on backtest)
    'active_props': ['pts', 'trb', 'ast', 'stl', 'tov', 'pra', 'pr', 'pa'],
    
    # Preferred direction (unders performed better)
    'prefer_unders': True,
    
    # File paths
    'data_file': 'player_games_schedule.csv',
    'predictions_dir': 'daily_predictions',
    'log_file': 'deployment.log',
    
    # API keys (set via environment variables)
    'odds_api_key': os.environ.get('ODDS_API_KEY'),
    'email_user': os.environ.get('EMAIL_USER'),
    'email_pass': os.environ.get('EMAIL_PASS'),
}

PROPS = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pra', 'pr', 'pa']

PROP_DISPLAY = {
    'pts': 'Points', 'trb': 'Rebounds', 'ast': 'Assists',
    'stl': 'Steals', 'blk': 'Blocks', 'tov': 'Turnovers',
    'pra': 'Pts+Reb+Ast', 'pr': 'Pts+Reb', 'pa': 'Pts+Ast'
}


# =============================================================================
# LOGGING
# =============================================================================

class Logger:
    """Simple logger that writes to file and console."""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self.start_time = datetime.now()
    
    def log(self, message: str, level: str = 'INFO'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted = f"[{timestamp}] [{level}] {message}"
        print(formatted)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted + '\n')
    
    def info(self, msg): self.log(msg, 'INFO')
    def warn(self, msg): self.log(msg, 'WARN')
    def error(self, msg): self.log(msg, 'ERROR')
    def success(self, msg): self.log(msg, 'SUCCESS')


# =============================================================================
# DATA UPDATES
# =============================================================================

def check_data_freshness(data_path: str, logger: Logger) -> dict:
    """Check if data needs updating."""
    if not Path(data_path).exists():
        return {'exists': False, 'needs_update': True, 'last_date': None}
    
    df = pd.read_csv(data_path, parse_dates=['game_date'])
    last_date = df['game_date'].max()
    today = datetime.now().date()
    days_old = (today - last_date.date()).days
    
    logger.info(f"Data last updated: {last_date.date()} ({days_old} days ago)")
    
    return {
        'exists': True,
        'needs_update': days_old > 1,
        'last_date': last_date,
        'days_old': days_old,
        'rows': len(df)
    }


def update_data_manual_instructions(logger: Logger):
    """Print instructions for manual data update."""
    logger.warn("Data needs updating. Manual steps required:")
    logger.info("  1. Go to Basketball-Reference.com")
    logger.info("  2. Download recent game logs")
    logger.info("  3. Append to database_24_25.csv")
    logger.info("  4. Re-run tasks 1-4 to regenerate features")
    logger.info("  Or use --skip-update to continue with existing data")


def scrape_todays_games(logger: Logger) -> list:
    """
    Get today's NBA games.
    Returns list of {'home': team, 'away': team, 'time': str}
    """
    if not HAS_REQUESTS:
        logger.warn("requests library not installed, can't fetch games")
        return []
    
    # Try The Odds API first
    api_key = CONFIG.get('odds_api_key')
    if api_key:
        try:
            url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
            params = {'apiKey': api_key, 'dateFormat': 'iso'}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            games = []
            for event in response.json():
                commence = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
                if commence.date() == datetime.now().date():
                    games.append({
                        'home': event['home_team'],
                        'away': event['away_team'],
                        'time': commence.strftime('%I:%M %p'),
                        'event_id': event['id']
                    })
            
            logger.info(f"Found {len(games)} games today via API")
            return games
        except Exception as e:
            logger.warn(f"API error: {e}")
    
    logger.warn("Could not fetch today's games automatically")
    return []


# =============================================================================
# MODEL LOADING & PREDICTION
# =============================================================================

def load_models(model_dir: Path, logger: Logger) -> dict:
    """Load all trained models."""
    models = {}
    for prop in CONFIG['active_props']:
        model_path = model_dir / f'model_{prop}.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[prop] = pickle.load(f)
    
    logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
    return models


def load_historical_data(data_path: str, logger: Logger) -> pd.DataFrame:
    """Load historical player data."""
    df = pd.read_csv(data_path, parse_dates=['game_date'])
    logger.info(f"Loaded {len(df):,} rows, date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    return df


def get_players_for_games(df: pd.DataFrame, games: list, logger: Logger) -> list:
    """
    Get players expected to play in today's games.
    Uses most recent roster from historical data.
    """
    # Team name mapping (API names to abbreviations)
    team_map = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BRK',
        'Charlotte Hornets': 'CHO', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHO',
        'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }
    
    matchups = []
    
    for game in games:
        home_abbr = team_map.get(game['home'], game['home'])
        away_abbr = team_map.get(game['away'], game['away'])
        
        # Get recent players for each team
        for team, opp in [(home_abbr, away_abbr), (away_abbr, home_abbr)]:
            # Players who played in last 5 games for this team
            team_recent = df[df['tm'] == team].sort_values('game_date')
            last_5_dates = team_recent['game_date'].unique()[-5:]
            recent_players = team_recent[team_recent['game_date'].isin(last_5_dates)]
            
            # Filter by minutes
            player_mins = recent_players.groupby('player')['mp_l10'].last()
            active_players = player_mins[player_mins >= CONFIG['min_minutes']].index.tolist()
            
            for player in active_players:
                matchups.append({
                    'player': player,
                    'team': team,
                    'opponent': opp,
                    'game_date': datetime.now()
                })
    
    logger.info(f"Found {len(matchups)} players for {len(games)} games")
    return matchups


def get_all_active_players(df: pd.DataFrame, logger: Logger) -> list:
    """Get all players with recent activity (fallback if no games found)."""
    # Players active in last 7 days of data
    last_date = df['game_date'].max()
    cutoff = last_date - timedelta(days=7)
    recent = df[df['game_date'] >= cutoff]
    
    # Filter by minutes
    player_stats = recent.groupby('player').agg({
        'tm': 'last',
        'opp': 'last', 
        'mp_l10': 'last',
        'game_date': 'last'
    }).reset_index()
    
    active = player_stats[player_stats['mp_l10'] >= CONFIG['min_minutes']]
    
    matchups = []
    for _, row in active.iterrows():
        matchups.append({
            'player': row['player'],
            'team': row['tm'],
            'opponent': row['opp'],  # Use last opponent as placeholder
            'game_date': datetime.now()
        })
    
    logger.info(f"Found {len(matchups)} active players (fallback mode)")
    return matchups


def predict_player(df: pd.DataFrame, models: dict, player: str, 
                   team: str, opp: str) -> dict:
    """Generate predictions for a single player."""
    # Get player's latest features
    player_df = df[df['player'] == player].sort_values('game_date')
    if len(player_df) == 0:
        return None
    
    latest = player_df.iloc[-1]
    
    predictions = {'player': player, 'team': team, 'opponent': opp}
    
    for prop, model_data in models.items():
        features = model_data['features']
        
        # Build feature vector
        feature_vals = {}
        for feat in features:
            if feat in latest.index:
                feature_vals[feat] = latest[feat]
            else:
                feature_vals[feat] = np.nan
        
        X = pd.DataFrame([feature_vals])
        X = X.fillna(model_data['medians'])
        X_scaled = model_data['scaler'].transform(X)
        
        pred = model_data['model'].predict(X_scaled)[0]
        pred = max(0, pred)  # Floor at 0
        
        # Get line proxy (L10 average)
        line_col = f'{prop}_l10'
        if prop in ['pra', 'pr', 'pa']:
            # Combo stats
            if prop == 'pra':
                line = latest.get('pts_l10', 0) + latest.get('trb_l10', 0) + latest.get('ast_l10', 0)
            elif prop == 'pr':
                line = latest.get('pts_l10', 0) + latest.get('trb_l10', 0)
            else:  # pa
                line = latest.get('pts_l10', 0) + latest.get('ast_l10', 0)
        else:
            line = latest.get(line_col, pred)
        
        predictions[prop] = round(pred, 1)
        predictions[f'{prop}_line'] = round(line, 1) if pd.notna(line) else None
        
        # Calculate edge
        if predictions[f'{prop}_line']:
            edge = pred - predictions[f'{prop}_line']
            predictions[f'{prop}_edge'] = round(edge, 1)
            predictions[f'{prop}_direction'] = 'OVER' if edge > 0 else 'UNDER'
    
    return predictions


def run_predictions(df: pd.DataFrame, models: dict, matchups: list, 
                    logger: Logger) -> list:
    """Run predictions for all matchups."""
    all_predictions = []
    
    for matchup in matchups:
        pred = predict_player(
            df, models,
            matchup['player'],
            matchup['team'],
            matchup['opponent']
        )
        
        if pred and pred.get('pts') is not None:
            all_predictions.append(pred)
    
    logger.info(f"Generated predictions for {len(all_predictions)} players")
    return all_predictions


# =============================================================================
# EDGE DETECTION & FILTERING
# =============================================================================

def find_betting_opportunities(predictions: list, logger: Logger) -> list:
    """Find high-edge betting opportunities."""
    opportunities = []
    
    for pred in predictions:
        for prop in CONFIG['active_props']:
            edge_key = f'{prop}_edge'
            dir_key = f'{prop}_direction'
            line_key = f'{prop}_line'
            
            if edge_key not in pred or pred[edge_key] is None:
                continue
            
            edge = abs(pred[edge_key])
            direction = pred[dir_key]
            
            # Apply filters from backtest findings
            if edge < CONFIG['min_edge']:
                continue
            
            # Prefer unders (they performed better)
            if CONFIG['prefer_unders'] and direction == 'OVER' and edge < CONFIG['strong_edge']:
                continue
            
            strength = 'STRONG' if edge >= CONFIG['strong_edge'] else 'MODERATE'
            
            opportunities.append({
                'player': pred['player'],
                'team': pred['team'],
                'opponent': pred['opponent'],
                'prop': prop,
                'prop_name': PROP_DISPLAY.get(prop, prop),
                'prediction': pred[prop],
                'line': pred[line_key],
                'edge': edge,
                'direction': direction,
                'strength': strength
            })
    
    # Sort by edge
    opportunities = sorted(opportunities, key=lambda x: -x['edge'])
    
    logger.info(f"Found {len(opportunities)} betting opportunities")
    return opportunities


# =============================================================================
# OUTPUT & ALERTS
# =============================================================================

def save_predictions(predictions: list, opportunities: list, 
                     output_dir: str, logger: Logger):
    """Save predictions and opportunities to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Save all predictions
    pred_df = pd.DataFrame(predictions)
    pred_file = output_path / f'predictions_{today}.csv'
    pred_df.to_csv(pred_file, index=False)
    logger.info(f"Saved predictions to: {pred_file}")
    
    # Save opportunities
    if opportunities:
        opp_df = pd.DataFrame(opportunities)
        opp_file = output_path / f'opportunities_{today}.csv'
        opp_df.to_csv(opp_file, index=False)
        logger.info(f"Saved opportunities to: {opp_file}")
    
    # Save summary JSON
    summary = {
        'date': today,
        'timestamp': datetime.now().isoformat(),
        'total_players': len(predictions),
        'total_opportunities': len(opportunities),
        'strong_bets': len([o for o in opportunities if o['strength'] == 'STRONG']),
        'moderate_bets': len([o for o in opportunities if o['strength'] == 'MODERATE']),
    }
    
    summary_file = output_path / f'summary_{today}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return pred_file, summary


def format_alert_message(opportunities: list) -> str:
    """Format opportunities as alert message."""
    if not opportunities:
        return "No betting opportunities found today."
    
    lines = [
        "🏀 NBA Props Alert",
        f"📅 {datetime.now().strftime('%A, %B %d, %Y')}",
        f"📊 {len(opportunities)} opportunities found",
        ""
    ]
    
    # Strong bets first
    strong = [o for o in opportunities if o['strength'] == 'STRONG']
    if strong:
        lines.append("⭐ STRONG BETS:")
        for o in strong[:5]:
            lines.append(f"  {o['player']} {o['prop_name']}")
            lines.append(f"    {o['direction']} {o['line']} (pred: {o['prediction']}, edge: +{o['edge']:.1f})")
        lines.append("")
    
    # Moderate bets
    moderate = [o for o in opportunities if o['strength'] == 'MODERATE']
    if moderate:
        lines.append(f"📈 MODERATE BETS ({len(moderate)}):")
        for o in moderate[:10]:
            lines.append(f"  {o['player']} {o['prop_name']} {o['direction']} {o['line']} (+{o['edge']:.1f})")
        lines.append("")
    
    lines.append("Good luck! 🍀")
    
    return '\n'.join(lines)


def send_email_alert(message: str, recipient: str, logger: Logger):
    """Send email alert."""
    if not HAS_EMAIL:
        logger.warn("Email libraries not available")
        return False
    
    email_user = CONFIG.get('email_user')
    email_pass = CONFIG.get('email_pass')
    
    if not email_user or not email_pass:
        logger.warn("Email credentials not configured")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = email_user
        msg['To'] = recipient
        msg['Subject'] = f"🏀 NBA Props Alert - {datetime.now().strftime('%m/%d/%Y')}"
        msg.attach(MIMEText(message, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_user, email_pass)
        server.send_message(msg)
        server.quit()
        
        logger.success(f"Email sent to {recipient}")
        return True
    except Exception as e:
        logger.error(f"Email failed: {e}")
        return False


def print_opportunities(opportunities: list, logger: Logger):
    """Print opportunities to console."""
    if not opportunities:
        logger.info("No betting opportunities found")
        return
    
    print("\n" + "="*70)
    print("TODAY'S BETTING OPPORTUNITIES")
    print("="*70)
    
    strong = [o for o in opportunities if o['strength'] == 'STRONG']
    moderate = [o for o in opportunities if o['strength'] == 'MODERATE']
    
    if strong:
        print("\n⭐ STRONG BETS (edge 3+ pts):")
        print("-"*70)
        for o in strong:
            print(f"  {o['player']:<22} {o['prop_name']:<12} {o['direction']:<6} "
                  f"{o['line']:>5.1f} → {o['prediction']:>5.1f} (edge: +{o['edge']:.1f})")
    
    if moderate:
        print(f"\n📈 MODERATE BETS ({len(moderate)}):")
        print("-"*70)
        for o in moderate[:15]:
            print(f"  {o['player']:<22} {o['prop_name']:<12} {o['direction']:<6} "
                  f"{o['line']:>5.1f} → {o['prediction']:>5.1f} (edge: +{o['edge']:.1f})")
    
    print("\n" + "="*70)
    print(f"Total: {len(strong)} strong, {len(moderate)} moderate")
    print("="*70)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(args, logger: Logger) -> dict:
    """Run the full prediction pipeline."""
    results = {
        'success': False,
        'predictions': 0,
        'opportunities': 0,
        'errors': []
    }
    
    try:
        # Step 1: Check data freshness
        logger.info("="*60)
        logger.info("STEP 1: Checking data")
        logger.info("="*60)
        
        data_status = check_data_freshness(CONFIG['data_file'], logger)
        
        if not data_status['exists']:
            logger.error(f"Data file not found: {CONFIG['data_file']}")
            results['errors'].append("Data file missing")
            return results
        
        if data_status['needs_update'] and not args.skip_update:
            update_data_manual_instructions(logger)
            if not args.force:
                logger.warn("Use --force to continue with stale data, or --skip-update")
                return results
        
        # Step 2: Load models
        logger.info("="*60)
        logger.info("STEP 2: Loading models")
        logger.info("="*60)
        
        models = load_models(Path(args.model_dir), logger)
        if not models:
            logger.error("No models found")
            results['errors'].append("No models")
            return results
        
        # Step 3: Load data
        logger.info("="*60)
        logger.info("STEP 3: Loading historical data")
        logger.info("="*60)
        
        df = load_historical_data(CONFIG['data_file'], logger)
        
        # Step 4: Get today's matchups
        logger.info("="*60)
        logger.info("STEP 4: Getting matchups")
        logger.info("="*60)
        
        games = scrape_todays_games(logger)
        
        if games:
            matchups = get_players_for_games(df, games, logger)
        else:
            logger.warn("No games found, using recent players")
            matchups = get_all_active_players(df, logger)
        
        if not matchups:
            logger.warn("No players to predict")
            results['success'] = True  # Not an error, just no games
            return results
        
        # Step 5: Generate predictions
        logger.info("="*60)
        logger.info("STEP 5: Generating predictions")
        logger.info("="*60)
        
        predictions = run_predictions(df, models, matchups, logger)
        results['predictions'] = len(predictions)
        
        # Step 6: Find opportunities
        logger.info("="*60)
        logger.info("STEP 6: Finding betting opportunities")
        logger.info("="*60)
        
        opportunities = find_betting_opportunities(predictions, logger)
        results['opportunities'] = len(opportunities)
        
        # Step 7: Output results
        logger.info("="*60)
        logger.info("STEP 7: Saving results")
        logger.info("="*60)
        
        if not args.dry_run:
            save_predictions(predictions, opportunities, 
                           CONFIG['predictions_dir'], logger)
        else:
            logger.info("Dry run - skipping file saves")
        
        # Print to console
        print_opportunities(opportunities, logger)
        
        # Step 8: Send alerts
        if args.email and opportunities:
            logger.info("="*60)
            logger.info("STEP 8: Sending alerts")
            logger.info("="*60)
            
            message = format_alert_message(opportunities)
            send_email_alert(message, args.email, logger)
        
        results['success'] = True
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        results['errors'].append(str(e))
        import traceback
        traceback.print_exc()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='NBA Props Daily Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python task10_deployment.py                     # Full daily run
  python task10_deployment.py --dry-run           # Test without saving
  python task10_deployment.py --force             # Run even with stale data
  python task10_deployment.py --email you@email.com  # With email alerts
  
Schedule with cron (Linux):
  0 9 * * * cd /path/to/nba && python task10_deployment.py >> cron.log 2>&1

Schedule with Task Scheduler (Windows):
  - Action: Start a program
  - Program: python
  - Arguments: C:\\path\\to\\task10_deployment.py
  - Start in: C:\\path\\to\\nba
        """
    )
    
    parser.add_argument('--dry-run', action='store_true', 
                        help="Test run without saving files")
    parser.add_argument('--force', action='store_true',
                        help="Run even if data is stale")
    parser.add_argument('--skip-update', action='store_true',
                        help="Skip data freshness check")
    parser.add_argument('--email', help="Email address for alerts")
    parser.add_argument('--model-dir', default='.', 
                        help="Directory with model files")
    parser.add_argument('--min-edge', type=float, default=1.5,
                        help="Minimum edge threshold (default: 1.5)")
    parser.add_argument('--min-minutes', type=float, default=15,
                        help="Minimum avg minutes (default: 15)")
    
    args = parser.parse_args()
    
    # Update config from args
    CONFIG['min_edge'] = args.min_edge
    CONFIG['min_minutes'] = args.min_minutes
    
    # Setup logger
    log_file = None if args.dry_run else CONFIG['log_file']
    logger = Logger(log_file)
    
    # Banner
    print("\n" + "="*60)
    print("🏀 NBA PROPS PREDICTION PIPELINE")
    print(f"📅 {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}")
    print("="*60)
    
    # Run pipeline
    results = run_pipeline(args, logger)
    
    # Final summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"  Status:        {'✓ SUCCESS' if results['success'] else '✗ FAILED'}")
    print(f"  Predictions:   {results['predictions']}")
    print(f"  Opportunities: {results['opportunities']}")
    
    if results['errors']:
        print(f"  Errors:        {len(results['errors'])}")
        for err in results['errors']:
            print(f"    - {err}")
    
    print("="*60 + "\n")
    
    # Exit code
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()
