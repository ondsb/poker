#!/usr/bin/env python3
"""
Analyze the known cards data to understand the issue with 0 samples.
"""

import pandas as pd
import glob
import os

def analyze_known_cards_data():
    """Analyze the processed known cards data."""
    print("🔍 Analyzing known cards data")
    print("-" * 50)
    
    data_dir = "data/processed/known_cards_focused"
    files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.parquet")))
    
    if not files:
        print("❌ No files found!")
        return
    
    total_known = 0
    total_unknown = 0
    total_winners = 0
    total_rows = 0
    
    for f in files:
        df = pd.read_parquet(f)
        known = sum(df["hole_card1"] != "unknown")
        unknown = sum(df["hole_card1"] == "unknown")
        winners = sum(df["is_winner"])
        rows = len(df)
        
        print(f"{os.path.basename(f)}: Known={known}, Unknown={unknown}, Winners={winners}, Total={rows}")
        
        total_known += known
        total_unknown += unknown
        total_winners += winners
        total_rows += rows
    
    print(f"\n📊 TOTAL SUMMARY:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Known hole cards: {total_known:,} ({100*total_known/total_rows:.1f}%)")
    print(f"  Unknown hole cards: {total_unknown:,} ({100*total_unknown/total_rows:.1f}%)")
    print(f"  Winners: {total_winners:,} ({100*total_winners/total_rows:.1f}%)")
    
    # Check if we have any rows with known cards
    if total_known > 0:
        print(f"\n✅ Found {total_known} rows with known hole cards!")
        
        # Load a sample with known cards
        for f in files:
            df = pd.read_parquet(f)
            known_mask = df["hole_card1"] != "unknown"
            if known_mask.sum() > 0:
                known_sample = df[known_mask].head(3)
                print(f"\n📋 Sample rows with known cards from {os.path.basename(f)}:")
                for _, row in known_sample.iterrows():
                    print(f"  Player {row['player_idx']}: {row['hole_card1']} {row['hole_card2']} (Winner: {row['is_winner']})")
                break
    else:
        print(f"\n❌ NO ROWS WITH KNOWN HOLE CARDS FOUND!")
        print(f"   This explains why the model shows 0 samples.")
        
        # Check what's in the hole card columns
        df_sample = pd.read_parquet(files[0])
        print(f"\n🔍 Sample hole card values:")
        print(f"  hole_card1 unique values: {df_sample['hole_card1'].value_counts().head()}")
        print(f"  hole_card2 unique values: {df_sample['hole_card2'].value_counts().head()}")
        
        # Check if there are any non-unknown values
        non_unknown_1 = df_sample[df_sample['hole_card1'] != 'unknown']['hole_card1'].unique()
        non_unknown_2 = df_sample[df_sample['hole_card2'] != 'unknown']['hole_card2'].unique()
        print(f"  Non-unknown hole_card1 values: {non_unknown_1}")
        print(f"  Non-unknown hole_card2 values: {non_unknown_2}")

if __name__ == "__main__":
    analyze_known_cards_data() 