#!/usr/bin/env python3
"""
Analyze processed/focused parquet files for hole card extraction quality.
"""
import os
import pandas as pd
from glob import glob

FOCUSED_DIR = 'data/processed/focused'

all_files = sorted(glob(os.path.join(FOCUSED_DIR, 'chunk_*.parquet')))

total_rows = 0
known_hole_rows = 0
known_opp_rows = 0
example_rows = []

for f in all_files:
    df = pd.read_parquet(f)
    total_rows += len(df)
    # Rows where both hole cards are known
    mask_known = (df['hole_card1'] != 'unknown') & (df['hole_card2'] != 'unknown')
    known_hole_rows += mask_known.sum()
    # Rows where at least one opponent has known hole cards
    opp_cols = [f'opp_{i}_has_hole_cards' for i in range(1, 6)]
    mask_opp = df[opp_cols].sum(axis=1) > 0
    known_opp_rows += mask_opp.sum()
    # Collect up to 5 example rows with known hole cards
    if len(example_rows) < 5:
        example_rows.extend(df[mask_known].head(5 - len(example_rows)).to_dict('records'))

print(f"Total rows: {total_rows}")
print(f"Rows with known hole cards: {known_hole_rows} ({100*known_hole_rows/total_rows:.2f}%)")
print(f"Rows with at least one opponent known: {known_opp_rows} ({100*known_opp_rows/total_rows:.2f}%)")
if example_rows:
    print("\nExample rows with known hole cards:")
    for row in example_rows:
        print({k: row[k] for k in ['hole_card1', 'hole_card2', 'player_idx', 'table', 'is_winner', 'winnings']})
if known_hole_rows / total_rows < 0.05:
    print("\n⚠️  WARNING: Very low rate of known hole cards! Check extraction logic or data quality.")

# Save summary
summary = pd.DataFrame({
    'total_rows': [total_rows],
    'known_hole_rows': [known_hole_rows],
    'known_opp_rows': [known_opp_rows],
    'known_hole_rate': [known_hole_rows/total_rows],
    'known_opp_rate': [known_opp_rows/total_rows],
})
summary.to_csv(os.path.join(FOCUSED_DIR, 'hole_card_analysis_summary.csv'), index=False)
print(f"\nSummary saved to {FOCUSED_DIR}/hole_card_analysis_summary.csv") 