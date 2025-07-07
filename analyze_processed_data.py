#!/usr/bin/env python3
"""Analyze processed poker data for training viability."""

import pandas as pd
import glob
import numpy as np
from pathlib import Path

def analyze_processed_data():
    """Analyze all processed chunks for training viability."""
    
    # Get all chunk files
    files = sorted(glob.glob('data/processed/focused/chunk_*.parquet'))
    print(f"Found {len(files)} chunk files")
    
    total_rows = 0
    all_columns = None
    
    # Analyze each file
    for f in files:
        df = pd.read_parquet(f)
        total_rows += len(df)
        print(f"{f}: {len(df):,} rows")
        
        if all_columns is None:
            all_columns = set(df.columns)
        else:
            all_columns = all_columns.intersection(set(df.columns))
    
    print(f"\nTotal dataset: {total_rows:,} rows")
    print(f"Common columns across all files: {len(all_columns)}")
    
    # Load first file for detailed analysis
    df_sample = pd.read_parquet(files[0])
    
    print(f"\n=== DETAILED ANALYSIS ===")
    print(f"Sample file shape: {df_sample.shape}")
    print(f"Data types: {df_sample.dtypes.value_counts()}")
    print(f"Missing values: {df_sample.isnull().sum().sum()}")
    
    # Target analysis
    print(f"\n=== TARGET ANALYSIS ===")
    print(f"Winner distribution: {df_sample['is_winner'].value_counts().to_dict()}")
    print(f"Win rate: {df_sample['is_winner'].mean():.3f}")
    
    # Feature analysis
    print(f"\n=== FEATURE ANALYSIS ===")
    
    # Player features
    player_features = [col for col in df_sample.columns if col.startswith('player_') or col in ['position', 'is_button', 'is_small_blind', 'is_big_blind', 'is_early_position', 'is_middle_position', 'is_late_position']]
    print(f"Player features: {len(player_features)}")
    
    # Hole card features
    hole_features = [col for col in df_sample.columns if col.startswith('hole_')]
    print(f"Hole card features: {len(hole_features)}")
    
    # Board features
    board_features = [col for col in df_sample.columns if col.startswith('board_')]
    print(f"Board features: {len(board_features)}")
    
    # Opponent features
    opp_features = [col for col in df_sample.columns if col.startswith('opp_')]
    print(f"Opponent features: {len(opp_features)}")
    
    # Game context features
    game_features = [col for col in df_sample.columns if col in ['seat_count', 'pot_size', 'min_bet', 'table']]
    print(f"Game context features: {len(game_features)}")
    
    # Categorical features
    categorical_features = df_sample.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Categorical features: {categorical_features}")
    
    # Numerical features
    numerical_features = df_sample.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_features = [f for f in numerical_features if f not in ['is_winner', 'winnings']]
    print(f"Numerical features: {len(numerical_features)}")
    
    # Check for potential issues
    print(f"\n=== POTENTIAL ISSUES ===")
    
    # Check for constant features
    constant_features = []
    for col in numerical_features:
        if df_sample[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        print(f"Constant features: {constant_features}")
    else:
        print("No constant features found")
    
    # Check for highly correlated features
    print(f"\n=== CORRELATION ANALYSIS ===")
    corr_matrix = df_sample[numerical_features].corr()
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr_pairs:
        print(f"Highly correlated feature pairs (|r| > 0.95): {len(high_corr_pairs)}")
        for pair in high_corr_pairs[:5]:  # Show first 5
            print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
    else:
        print("No highly correlated features found")
    
    # Training viability assessment
    print(f"\n=== TRAINING VIABILITY ASSESSMENT ===")
    
    issues = []
    strengths = []
    
    # Check dataset size
    if total_rows < 10000:
        issues.append(f"Small dataset: {total_rows:,} rows")
    else:
        strengths.append(f"Good dataset size: {total_rows:,} rows")
    
    # Check class balance
    win_rate = df_sample['is_winner'].mean()
    if win_rate < 0.01 or win_rate > 0.99:
        issues.append(f"Severe class imbalance: {win_rate:.3f} win rate")
    elif win_rate < 0.05 or win_rate > 0.95:
        issues.append(f"Class imbalance: {win_rate:.3f} win rate")
    else:
        strengths.append(f"Reasonable class balance: {win_rate:.3f} win rate")
    
    # Check feature count
    if len(numerical_features) < 10:
        issues.append(f"Few features: {len(numerical_features)} numerical features")
    else:
        strengths.append(f"Good feature count: {len(numerical_features)} numerical features")
    
    # Check missing values
    if df_sample.isnull().sum().sum() > 0:
        issues.append("Missing values present")
    else:
        strengths.append("No missing values")
    
    # Check opponent features
    if len(opp_features) > 0:
        strengths.append(f"Rich opponent information: {len(opp_features)} opponent features")
    else:
        issues.append("No opponent features")
    
    print("Strengths:")
    for strength in strengths:
        print(f"  ✅ {strength}")
    
    print("Issues:")
    for issue in issues:
        print(f"  ⚠️ {issue}")
    
    if len(issues) == 0:
        print("\n🎉 Dataset is ready for training!")
    else:
        print(f"\n⚠️ {len(issues)} issues to address before training")
    
    return {
        'total_rows': total_rows,
        'features': len(numerical_features),
        'win_rate': win_rate,
        'issues': issues,
        'strengths': strengths
    }

if __name__ == "__main__":
    analyze_processed_data() 