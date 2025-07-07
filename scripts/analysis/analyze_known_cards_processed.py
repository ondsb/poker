#!/usr/bin/env python3
"""
Analyze processed known cards data to verify hole card extraction.
"""

import pandas as pd
import os
from glob import glob
from pathlib import Path

def analyze_known_cards_processed():
    """Analyze the processed known cards data."""
    print("🔍 Analyzing processed known cards data")
    print("-" * 50)
    
    processed_dir = "data/processed/known_cards_focused"
    parquet_files = sorted(glob(os.path.join(processed_dir, "chunk_*.parquet")))
    
    if not parquet_files:
        print("❌ No processed files found!")
        return
    
    total_rows = 0
    known_hole_rows = 0
    unknown_hole_rows = 0
    example_rows = []
    
    for file_path in parquet_files:
        print(f"📁 Processing: {os.path.basename(file_path)}")
        df = pd.read_parquet(file_path)
        
        file_rows = len(df)
        total_rows += file_rows
        
        # Count rows with known vs unknown hole cards
        mask_known = (df['hole_card1'] != 'unknown') & (df['hole_card2'] != 'unknown')
        file_known = mask_known.sum()
        file_unknown = (~mask_known).sum()
        
        known_hole_rows += file_known
        unknown_hole_rows += file_unknown
        
        print(f"  📊 Rows: {file_rows:,}")
        print(f"  🃏 Known hole cards: {file_known:,} ({100*file_known/file_rows:.1f}%)")
        print(f"  🃏 Unknown hole cards: {file_unknown:,} ({100*file_unknown/file_rows:.1f}%)")
        
        # Collect examples of known cards
        if file_known > 0 and len(example_rows) < 10:
            known_examples = df[mask_known].head(10 - len(example_rows))
            for _, row in known_examples.iterrows():
                example_rows.append({
                    'action_idx': row['action_idx'],
                    'player_idx': row['player_idx'],
                    'hole_card1': row['hole_card1'],
                    'hole_card2': row['hole_card2'],
                    'has_hole_cards': row['has_hole_cards'],
                    'is_winner': row['is_winner']
                })
    
    print(f"\n📊 Overall Summary:")
    print(f"  📁 Total files: {len(parquet_files)}")
    print(f"  📊 Total rows: {total_rows:,}")
    print(f"  🃏 Known hole cards: {known_hole_rows:,} ({100*known_hole_rows/total_rows:.1f}%)")
    print(f"  🃏 Unknown hole cards: {unknown_hole_rows:,} ({100*unknown_hole_rows/total_rows:.1f}%)")
    
    if example_rows:
        print(f"\n🃏 Example rows with known hole cards:")
        for i, row in enumerate(example_rows[:5]):
            print(f"  {i+1}. Action {row['action_idx']}, Player {row['player_idx']}: {row['hole_card1']} {row['hole_card2']} (Winner: {row['is_winner']})")
    
    # Check for action progression
    if example_rows:
        print(f"\n📈 Action progression analysis:")
        sample_df = pd.read_parquet(parquet_files[0])
        if 'action_idx' in sample_df.columns:
            action_progression = sample_df.groupby('action_idx')['has_hole_cards'].mean()
            print(f"  📈 Hole card visibility by action:")
            for action_idx, visibility_rate in action_progression.head(10).items():
                print(f"    Action {action_idx}: {100*visibility_rate:.1f}% known")
    
    # Save analysis summary
    summary = {
        'total_files': len(parquet_files),
        'total_rows': int(total_rows),
        'known_hole_rows': int(known_hole_rows),
        'unknown_hole_rows': int(unknown_hole_rows),
        'known_rate': float(known_hole_rows / total_rows),
        'example_rows': example_rows[:5]
    }
    
    summary_file = Path(processed_dir) / "analysis_summary.json"
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Analysis summary saved to: {summary_file}")

if __name__ == "__main__":
    analyze_known_cards_processed() 