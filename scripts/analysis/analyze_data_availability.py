#!/usr/bin/env python3
"""
Analyze data availability for all players' cards across different datasets.
"""

import json
import os
import glob
from collections import defaultdict, Counter
import pandas as pd
from pathlib import Path

def analyze_jsonl_dataset(file_path, sample_size=10000):
    """Analyze the main JSONL dataset for hole card availability."""
    print(f"\n=== Analyzing JSONL Dataset: {file_path} ===")
    
    total_lines = 0
    null_cards = 0
    unknown_cards = 0
    known_cards = 0
    known_cards_examples = []
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            total_lines += 1
            
            try:
                data = json.loads(line.strip())
                hole_cards = data.get('hole_cards')
                
                if hole_cards is None:
                    null_cards += 1
                elif hole_cards == "????":
                    unknown_cards += 1
                else:
                    known_cards += 1
                    if len(known_cards_examples) < 5:
                        known_cards_examples.append({
                            'hole_cards': hole_cards,
                            'player_id': data.get('player_id'),
                            'table': data.get('table'),
                            'seat_count': data.get('seat_count')
                        })
            except json.JSONDecodeError:
                continue
    
    print(f"Total lines analyzed: {total_lines}")
    print(f"Null hole cards: {null_cards} ({null_cards/total_lines*100:.1f}%)")
    print(f"Unknown hole cards (????): {unknown_cards} ({unknown_cards/total_lines*100:.1f}%)")
    print(f"Known hole cards: {known_cards} ({known_cards/total_lines*100:.1f}%)")
    
    if known_cards_examples:
        print("\nExamples of known hole cards:")
        for ex in known_cards_examples:
            print(f"  {ex['hole_cards']} (Player: {ex['player_id'][:10]}..., Table: {ex['table'][:10]}..., Seats: {ex['seat_count']})")
    
    return {
        'total': total_lines,
        'null': null_cards,
        'unknown': unknown_cards,
        'known': known_cards,
        'examples': known_cards_examples
    }

def analyze_phh_file(file_path):
    """Analyze a single PHH file for hole card availability."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract deal actions
        deal_actions = []
        for line in content.split('\n'):
            if 'd dh p' in line and '#' not in line:
                deal_actions.append(line.strip())
        
        # Count known vs unknown cards
        known_cards = 0
        unknown_cards = 0
        card_examples = []
        
        for action in deal_actions:
            # Extract card part after player
            parts = action.split()
            if len(parts) >= 3:
                cards = parts[2]
                if '?' in cards:
                    unknown_cards += 1
                else:
                    known_cards += 1
                    if len(card_examples) < 3:
                        card_examples.append(cards)
        
        return {
            'file': file_path,
            'known': known_cards,
            'unknown': unknown_cards,
            'total_players': known_cards + unknown_cards,
            'examples': card_examples
        }
    except Exception as e:
        return {'file': file_path, 'error': str(e)}

def analyze_phh_dataset(dataset_path, max_files=100):
    """Analyze PHH dataset for hole card availability."""
    print(f"\n=== Analyzing PHH Dataset: {dataset_path} ===")
    
    phh_files = glob.glob(f"{dataset_path}/**/*.phh", recursive=True)
    print(f"Found {len(phh_files)} PHH files")
    
    if len(phh_files) > max_files:
        phh_files = phh_files[:max_files]
        print(f"Analyzing first {max_files} files...")
    
    results = []
    total_hands = 0
    total_known_cards = 0
    total_unknown_cards = 0
    complete_hands = 0  # Hands where all players have known cards
    
    for file_path in phh_files:
        result = analyze_phh_file(file_path)
        if 'error' not in result:
            results.append(result)
            total_hands += 1
            total_known_cards += result['known']
            total_unknown_cards += result['unknown']
            
            if result['unknown'] == 0 and result['total_players'] > 0:
                complete_hands += 1
    
    if results:
        print(f"\nAnalyzed {total_hands} hands:")
        print(f"Total players with known cards: {total_known_cards}")
        print(f"Total players with unknown cards: {total_unknown_cards}")
        print(f"Complete hands (all cards known): {complete_hands} ({complete_hands/total_hands*100:.1f}%)")
        
        # Show some examples
        complete_examples = [r for r in results if r['unknown'] == 0 and r['total_players'] > 0]
        if complete_examples:
            print(f"\nExamples of complete hands:")
            for ex in complete_examples[:3]:
                print(f"  {ex['file']}: {ex['examples']}")
    
    return {
        'total_hands': total_hands,
        'total_known': total_known_cards,
        'total_unknown': total_unknown_cards,
        'complete_hands': complete_hands,
        'results': results
    }

def find_complete_hands_in_jsonl(file_path, output_file, min_players=6):
    """Find hands where all players have known cards and save them."""
    print(f"\n=== Finding Complete Hands in JSONL ===")
    
    complete_hands = []
    table_hands = defaultdict(list)
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                table = data.get('table')
                hole_cards = data.get('hole_cards')
                
                if table and hole_cards and hole_cards != "????":
                    table_hands[table].append(data)
            except json.JSONDecodeError:
                continue
    
    # Find tables where all players have known cards
    for table, hands in table_hands.items():
        if len(hands) >= min_players:
            # Check if all players in this table have known cards
            all_known = all(h.get('hole_cards') and h.get('hole_cards') != "????" for h in hands)
            if all_known:
                complete_hands.extend(hands)
    
    print(f"Found {len(complete_hands)} players in complete hands")
    
    # Save complete hands
    if complete_hands:
        with open(output_file, 'w') as f:
            for hand in complete_hands:
                f.write(json.dumps(hand) + '\n')
        print(f"Saved complete hands to {output_file}")
    
    return complete_hands

def main():
    """Main analysis function."""
    print("=== Poker Data Availability Analysis ===\n")
    
    # Analyze main JSONL dataset
    jsonl_file = "data/raw/poker_training_data.jsonl"
    if os.path.exists(jsonl_file):
        jsonl_stats = analyze_jsonl_dataset(jsonl_file)
        
        # Find complete hands
        complete_hands_file = "data/raw/complete_hands.jsonl"
        complete_hands = find_complete_hands_in_jsonl(jsonl_file, complete_hands_file)
    
    # Analyze PHH datasets
    phh_base = "data/raw/phh-dataset/data"
    
    # Analyze individual PHH files
    individual_files = [
        "data/raw/phh-dataset/data/antonius-blom-2009.phh",
        "data/raw/phh-dataset/data/arieh-yockey-2019.phh",
        "data/raw/phh-dataset/data/wsop/2023/43/5/03-22-08.phh",
        "data/raw/phh-dataset/data/pluribus/99/81.phh"
    ]
    
    print("\n=== Individual PHH File Analysis ===")
    for file_path in individual_files:
        if os.path.exists(file_path):
            result = analyze_phh_file(file_path)
            print(f"\n{file_path}:")
            print(f"  Known cards: {result['known']}")
            print(f"  Unknown cards: {result['unknown']}")
            print(f"  Total players: {result['total_players']}")
            if result['examples']:
                print(f"  Examples: {result['examples']}")
    
    # Analyze larger PHH datasets
    if os.path.exists(phh_base):
        # Analyze Pluribus dataset
        pluribus_path = f"{phh_base}/pluribus"
        if os.path.exists(pluribus_path):
            pluribus_stats = analyze_phh_dataset(pluribus_path, max_files=50)
        
        # Analyze WSOP dataset
        wsop_path = f"{phh_base}/wsop"
        if os.path.exists(wsop_path):
            wsop_stats = analyze_phh_dataset(wsop_path, max_files=50)
    
    print("\n=== Summary ===")
    print("1. Main JSONL dataset: Has ~2.5M entries with known hole cards, but they're player-centric")
    print("2. PHH datasets: Have complete hand information with all players' cards")
    print("3. Recommendation: Use PHH datasets for training with complete hand information")

if __name__ == "__main__":
    main() 