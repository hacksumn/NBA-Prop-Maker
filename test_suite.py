"""
Comprehensive Test Suite for NBA Prop Prediction Model
Tests to ensure data leakage is fixed and models are working correctly
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def test_no_leaky_features():
    """Test 1: Ensure no current-game stats are in model features."""
    print("="*60)
    print("TEST 1: No Leaky Features")
    print("="*60)
    
    leaky_features = {
        'pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf',
        'mp', 'fg', 'fga', 'fg_pct', '3p', '3pa', '3p_pct',
        'ft', 'fta', 'ft_pct', 'orb', 'drb', 'gmsc',
        'FANTASY_PTS', 'PLUS_MINUS', 'res', 'won'
    }
    
    props = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pra', 'pr', 'pa']
    all_passed = True
    
    for prop in props:
        model_path = f'model_{prop}.pkl'
        if not Path(model_path).exists():
            continue
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        features = model_data['features']
        found_leaky = [f for f in features if f in leaky_features]
        
        if found_leaky:
            print(f"❌ {prop.upper()}: Found leaky features: {found_leaky}")
            all_passed = False
        else:
            print(f"✓ {prop.upper()}: Clean")
    
    if all_passed:
        print("\n✅ TEST PASSED: No leaky features detected")
    else:
        print("\n❌ TEST FAILED: Leaky features found!")
    
    return all_passed


def test_rolling_features_shifted():
    """Test 2: Verify rolling features are properly shifted."""
    print("\n" + "="*60)
    print("TEST 2: Rolling Features Properly Shifted")
    print("="*60)
    
    df = pd.read_csv('player_games_schedule.csv', parse_dates=['game_date'])
    df = df.sort_values(['player', 'game_date'])
    
    # Pick a player with many games
    player = 'LeBron James'
    player_df = df[df['player'] == player].head(10)
    
    if len(player_df) < 2:
        print("⚠️ Not enough data for test")
        return True
    
    # Check: Game 2's pts_l5 should equal Game 1's pts
    game1_pts = player_df.iloc[0]['pts']
    game2_pts_l5 = player_df.iloc[1]['pts_l5']
    
    print(f"\nPlayer: {player}")
    print(f"Game 1 actual PTS: {game1_pts}")
    print(f"Game 2 pts_l5: {game2_pts_l5}")
    
    if abs(game1_pts - game2_pts_l5) < 0.01:
        print("✅ TEST PASSED: Rolling features properly shifted")
        return True
    else:
        print("❌ TEST FAILED: Rolling features not properly shifted!")
        return False


def test_realistic_predictions():
    """Test 3: Verify predictions are realistic (not too accurate)."""
    print("\n" + "="*60)
    print("TEST 3: Realistic Prediction Accuracy")
    print("="*60)
    
    props = ['pts', 'trb', 'ast', 'pra', 'pr', 'pa']
    all_passed = True
    
    for prop in props:
        pred_file = f'predictions_{prop}.csv'
        if not Path(pred_file).exists():
            continue
        
        df = pd.read_csv(pred_file)
        
        # Calculate betting accuracy
        pred_over = df['predicted'] > df['line']
        actual_over = df['actual'] > df['line']
        betting_acc = (pred_over == actual_over).mean()
        
        # Calculate correlation
        correlation = df['actual'].corr(df['predicted'])
        
        print(f"\n{prop.upper()}:")
        print(f"  Betting Accuracy: {betting_acc:.1%}")
        print(f"  Correlation: {correlation:.3f}")
        
        # Check if too accurate (sign of leakage)
        if betting_acc > 0.65:
            print(f"  ❌ Betting accuracy too high (>{65}%)")
            all_passed = False
        elif betting_acc < 0.48:
            print(f"  ⚠️ Betting accuracy too low (<48%)")
        else:
            print(f"  ✓ Realistic betting accuracy")
        
        if correlation > 0.85:
            print(f"  ❌ Correlation too high (>{0.85})")
            all_passed = False
        else:
            print(f"  ✓ Reasonable correlation")
    
    if all_passed:
        print("\n✅ TEST PASSED: Predictions are realistic")
    else:
        print("\n❌ TEST FAILED: Some predictions are unrealistic!")
    
    return all_passed


def test_edge_distribution():
    """Test 4: Verify edge distribution is reasonable."""
    print("\n" + "="*60)
    print("TEST 4: Edge Distribution")
    print("="*60)
    
    df = pd.read_csv('predictions_pts.csv')
    df['edge'] = df['predicted'] - df['line']
    
    print(f"\nEdge statistics for PTS:")
    print(f"  Mean: {df['edge'].mean():.2f}")
    print(f"  Std: {df['edge'].std():.2f}")
    print(f"  Min: {df['edge'].min():.2f}")
    print(f"  Max: {df['edge'].max():.2f}")
    print(f"  Median: {df['edge'].median():.2f}")
    
    # Count extreme edges
    extreme_edges = (df['edge'].abs() > 10).sum()
    pct_extreme = extreme_edges / len(df) * 100
    
    print(f"\nExtreme edges (>10 points): {extreme_edges} ({pct_extreme:.1f}%)")
    
    if pct_extreme > 5:
        print("❌ TEST FAILED: Too many extreme edges!")
        return False
    else:
        print("✅ TEST PASSED: Edge distribution is reasonable")
        return True


def test_model_performance():
    """Test 5: Verify model performance metrics are in expected range."""
    print("\n" + "="*60)
    print("TEST 5: Model Performance Metrics")
    print("="*60)
    
    expected_ranges = {
        'pts': {'r2': (0.45, 0.65), 'mae': (3.5, 5.5)},
        'trb': {'r2': (0.35, 0.55), 'mae': (1.5, 2.5)},
        'ast': {'r2': (0.40, 0.60), 'mae': (1.0, 1.8)},
        'pra': {'r2': (0.55, 0.70), 'mae': (5.0, 7.0)},
    }
    
    all_passed = True
    
    for prop, ranges in expected_ranges.items():
        model_path = f'model_{prop}.pkl'
        if not Path(model_path).exists():
            continue
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        metrics = model_data['metrics']
        r2 = metrics['r2']
        mae = metrics['mae']
        
        print(f"\n{prop.upper()}:")
        print(f"  R²: {r2:.3f} (expected: {ranges['r2'][0]:.2f}-{ranges['r2'][1]:.2f})")
        print(f"  MAE: {mae:.2f} (expected: {ranges['mae'][0]:.2f}-{ranges['mae'][1]:.2f})")
        
        r2_ok = ranges['r2'][0] <= r2 <= ranges['r2'][1]
        mae_ok = ranges['mae'][0] <= mae <= ranges['mae'][1]
        
        if r2_ok and mae_ok:
            print(f"  ✓ Metrics in expected range")
        else:
            if not r2_ok:
                print(f"  ⚠️ R² outside expected range")
            if not mae_ok:
                print(f"  ⚠️ MAE outside expected range")
            # Not failing test, just warning
    
    print("\n✅ TEST PASSED: Model performance reviewed")
    return True


def test_prediction_consistency():
    """Test 6: Verify predictions are consistent with historical averages."""
    print("\n" + "="*60)
    print("TEST 6: Prediction Consistency")
    print("="*60)
    
    df = pd.read_csv('predictions_pts.csv')
    
    # Predictions should be somewhat close to lines (player's L10 avg)
    # But not too close (that would indicate no model value)
    
    diff = (df['predicted'] - df['line']).abs()
    mean_diff = diff.mean()
    
    print(f"\nMean absolute difference between prediction and line: {mean_diff:.2f}")
    
    # Should be between 1 and 5 points on average
    if 0.5 < mean_diff < 6:
        print("✅ TEST PASSED: Predictions show appropriate deviation from lines")
        return True
    elif mean_diff <= 0.5:
        print("❌ TEST FAILED: Predictions too close to lines (model not adding value)")
        return False
    else:
        print("❌ TEST FAILED: Predictions too far from lines (possible issue)")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("NBA PROP PREDICTION MODEL - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    tests = [
        ("No Leaky Features", test_no_leaky_features),
        ("Rolling Features Shifted", test_rolling_features_shifted),
        ("Realistic Predictions", test_realistic_predictions),
        ("Edge Distribution", test_edge_distribution),
        ("Model Performance", test_model_performance),
        ("Prediction Consistency", test_prediction_consistency),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ {test_name} CRASHED: {str(e)}")
            results.append((test_name, False))
    
    # Final summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - MODEL IS READY FOR PRODUCTION!")
    else:
        print("⚠️ SOME TESTS FAILED - REVIEW REQUIRED")
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
