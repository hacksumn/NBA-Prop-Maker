"""
BET TRACKER & GRADER
NBA Player Props Betting System

Creates tracking sheet from model_vs_vegas.csv
Grades results after games finish

Usage:
    # Before games: Lock in picks
    python grade_bets.py lock
    
    # After games: Grade results  
    python grade_bets.py grade
    
    # View summary
    python grade_bets.py summary
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os


def calc_profit(odds, result, stake=100):
    """Calculate profit/loss from a bet"""
    if result == 'WIN':
        if odds > 0:
            return stake * (odds / 100)
        else:
            return stake * (100 / abs(odds))
    elif result == 'LOSS':
        return -stake
    elif result == 'PUSH':
        return 0
    return None


def lock_picks(min_edge=3.0):
    """Create tracking sheet from model_vs_vegas.csv"""
    
    if not os.path.exists('model_vs_vegas.csv'):
        print("ERROR: model_vs_vegas.csv not found!")
        print("Run get_vegas_lines.py first.")
        return
    
    df = pd.read_csv('model_vs_vegas.csv')
    
    # Filter to minimum edge
    df['edge_abs'] = df['real_edge'].abs()
    df = df[df['edge_abs'] >= min_edge].copy()
    
    # Create tracking sheet
    tracking = pd.DataFrame({
        'date': datetime.now().strftime('%Y-%m-%d'),
        'locked_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'player': df['player'],
        'prop': df['prop_display'],
        'bet_direction': df['real_direction'],
        'vegas_line': df['vegas_line'],
        'prediction': df['prediction'],
        'edge': df['real_edge'].round(1),
        'edge_abs': df['edge_abs'].round(1),
        'confidence': df['edge_abs'].apply(lambda x: 'HIGH' if x >= 5 else 'MED'),
        'odds': df['bet_odds'],
        'actual': '',
        'result': '',
        'profit': ''
    })
    
    # Sort by edge
    tracking = tracking.sort_values('edge_abs', ascending=False).reset_index(drop=True)
    
    # Save
    filename = f"tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    tracking.to_csv(filename, index=False)
    
    # Also save as "latest" for easy grading
    tracking.to_csv('tracking_latest.csv', index=False)
    
    print("="*70)
    print("🔒 PICKS LOCKED")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Minimum edge: {min_edge} points")
    print(f"Total bets: {len(tracking)}")
    print(f"HIGH confidence (5+): {len(tracking[tracking['confidence']=='HIGH'])}")
    print(f"MED confidence (3-5): {len(tracking[tracking['confidence']=='MED'])}")
    
    print("\n" + "="*70)
    print("TONIGHT'S PLAYS")
    print("="*70)
    
    for conf in ['HIGH', 'MED']:
        subset = tracking[tracking['confidence'] == conf]
        if len(subset) > 0:
            stars = '⭐⭐⭐' if conf == 'HIGH' else '⭐⭐'
            print(f"\n{stars} {conf} CONFIDENCE ({len(subset)} bets):")
            for _, row in subset.iterrows():
                direction = "🔼" if row['bet_direction'] == 'OVER' else "🔽"
                print(f"  {direction} {row['player']:<22} {row['prop']:<14} {row['bet_direction']:<5} {row['vegas_line']:>5.1f}  (pred: {row['prediction']:.1f}, edge: {row['edge']:+.1f}, odds: {row['odds']:+d})")
    
    print("\n" + "="*70)
    print(f"✓ Saved: {filename}")
    print(f"✓ Saved: tracking_latest.csv")
    print("="*70)
    print("\nAfter games finish, run: python grade_bets.py grade")


def grade_picks():
    """Grade picks by entering actual results"""
    
    if not os.path.exists('tracking_latest.csv'):
        print("ERROR: tracking_latest.csv not found!")
        print("Run 'python grade_bets.py lock' first.")
        return
    
    df = pd.read_csv('tracking_latest.csv')
    
    print("="*70)
    print("GRADING PICKS")
    print("="*70)
    print("Enter actual stats for each player (or 'skip' to skip, 'done' to finish)\n")
    
    for idx, row in df.iterrows():
        if pd.notna(row['actual']) and row['actual'] != '':
            continue  # Already graded
            
        print(f"\n{row['player']} - {row['prop']}")
        print(f"  Bet: {row['bet_direction']} {row['vegas_line']} @ {row['odds']:+d}")
        print(f"  Model predicted: {row['prediction']:.1f}")
        
        actual = input("  Actual stat: ").strip()
        
        if actual.lower() == 'done':
            break
        if actual.lower() == 'skip':
            continue
            
        try:
            actual_val = float(actual)
            df.at[idx, 'actual'] = actual_val
            
            # Determine result
            if row['bet_direction'] == 'OVER':
                if actual_val > row['vegas_line']:
                    result = 'WIN'
                elif actual_val < row['vegas_line']:
                    result = 'LOSS'
                else:
                    result = 'PUSH'
            else:  # UNDER
                if actual_val < row['vegas_line']:
                    result = 'WIN'
                elif actual_val > row['vegas_line']:
                    result = 'LOSS'
                else:
                    result = 'PUSH'
            
            df.at[idx, 'result'] = result
            df.at[idx, 'profit'] = calc_profit(row['odds'], result)
            
            emoji = '✅' if result == 'WIN' else ('❌' if result == 'LOSS' else '➖')
            print(f"  {emoji} {result} (profit: {df.at[idx, 'profit']:+.2f})")
            
        except ValueError:
            print("  Invalid input, skipping...")
    
    # Save updated
    df.to_csv('tracking_latest.csv', index=False)
    df.to_csv(f"tracking_graded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    
    print("\n✓ Progress saved to tracking_latest.csv")
    print("Run 'python grade_bets.py summary' to see results")


def auto_grade(box_scores_file=None):
    """Auto-grade from a box scores CSV file"""
    
    if not os.path.exists('tracking_latest.csv'):
        print("ERROR: tracking_latest.csv not found!")
        return
    
    df = pd.read_csv('tracking_latest.csv')
    
    if box_scores_file and os.path.exists(box_scores_file):
        # Load box scores and auto-match
        box = pd.read_csv(box_scores_file)
        print(f"Loaded {len(box)} box score records")
        # Would need to implement matching logic based on file format
    else:
        print("Provide a box scores file or use manual grading:")
        print("  python grade_bets.py grade")


def show_summary():
    """Show summary of graded bets"""
    
    if not os.path.exists('tracking_latest.csv'):
        print("ERROR: tracking_latest.csv not found!")
        return
    
    df = pd.read_csv('tracking_latest.csv')
    
    # Filter to graded only
    graded = df[df['result'].isin(['WIN', 'LOSS', 'PUSH'])].copy()
    
    if len(graded) == 0:
        print("No graded bets yet. Run 'python grade_bets.py grade' first.")
        return
    
    print("="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    print(f"Date: {df['date'].iloc[0]}")
    print(f"Graded: {len(graded)} / {len(df)} bets")
    
    # Overall stats
    wins = len(graded[graded['result'] == 'WIN'])
    losses = len(graded[graded['result'] == 'LOSS'])
    pushes = len(graded[graded['result'] == 'PUSH'])
    total_profit = graded['profit'].sum()
    
    win_pct = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    
    print(f"\n{'='*70}")
    print("OVERALL")
    print(f"{'='*70}")
    print(f"  Record: {wins}-{losses}-{pushes}")
    print(f"  Win Rate: {win_pct:.1f}%")
    print(f"  Total Profit: {total_profit:+.2f} units")
    print(f"  ROI: {total_profit / len(graded):+.2f} units/bet")
    
    # By confidence
    print(f"\n{'='*70}")
    print("BY CONFIDENCE")
    print(f"{'='*70}")
    
    for conf in ['HIGH', 'MED']:
        subset = graded[graded['confidence'] == conf]
        if len(subset) > 0:
            w = len(subset[subset['result'] == 'WIN'])
            l = len(subset[subset['result'] == 'LOSS'])
            p = subset['profit'].sum()
            pct = w / (w + l) * 100 if (w + l) > 0 else 0
            print(f"  {conf}: {w}-{l} ({pct:.1f}%), Profit: {p:+.2f}")
    
    # By prop type
    print(f"\n{'='*70}")
    print("BY PROP TYPE")
    print(f"{'='*70}")
    
    for prop in graded['prop'].unique():
        subset = graded[graded['prop'] == prop]
        if len(subset) > 0:
            w = len(subset[subset['result'] == 'WIN'])
            l = len(subset[subset['result'] == 'LOSS'])
            p = subset['profit'].sum()
            pct = w / (w + l) * 100 if (w + l) > 0 else 0
            print(f"  {prop:<14}: {w}-{l} ({pct:.1f}%), Profit: {p:+.2f}")
    
    # Individual results
    print(f"\n{'='*70}")
    print("ALL BETS")
    print(f"{'='*70}")
    
    for _, row in graded.iterrows():
        emoji = '✅' if row['result'] == 'WIN' else ('❌' if row['result'] == 'LOSS' else '➖')
        print(f"  {emoji} {row['player']:<20} {row['prop']:<14} {row['bet_direction']:<5} {row['vegas_line']:>5.1f} | Actual: {row['actual']:>5.1f} | {row['profit']:+.2f}")
    
    print("="*70)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python grade_bets.py lock     # Lock in tonight's picks")
        print("  python grade_bets.py grade    # Grade results manually")
        print("  python grade_bets.py summary  # Show results summary")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'lock':
        min_edge = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
        lock_picks(min_edge)
    elif command == 'grade':
        grade_picks()
    elif command == 'summary':
        show_summary()
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
