#!/usr/bin/env python3
"""
Analyze the actual hand structure in the JSONL dataset to understand if we can get all players' cards.
"""

import json
import os
from collections import defaultdict
import re

def analyze_hand_structure(file_path, sample_size=1000):
    """Analyze how hands are structured in the JSONL dataset."""
    print(f"=== Analyzing Hand Structure in {file_path} ===")
    
    table_hands = defaultdict(list)
    hand_count = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            if hand_count >= sample_size:
                break
            
            try:
                data = json.loads(line.strip())
                table = data.get('table')
                if table:
                    table_hands[table].append(data)
                    hand_count += 1
            except json.JSONDecodeError:
                continue
    
    print(f"Analyzed {hand_count} entries across {len(table_hands)} unique tables")
    
    # Analyze a few complete tables
    complete_tables = 0
    examples = []
    
    for table, hands in table_hands.items():
        if len(hands) >= 6:  # 6-player hands
            # Check if this table has all players with known cards
            known_cards_count = sum(1 for h in hands if h.get('hole_cards') and h.get('hole_cards') != "????")
            
            if known_cards_count >= 2:  # At least 2 players with known cards
                complete_tables += 1
                if len(examples) < 3:
                    examples.append({
                        'table': table,
                        'total_players': len(hands),
                        'known_cards': known_cards_count,
                        'hands': hands[:3]  # First 3 hands
                    })
    
    print(f"Found {complete_tables} tables with at least 2 players having known cards")
    
    if examples:
        print("\nExample tables:")
        for ex in examples:
            print(f"\nTable: {ex['table'][:20]}...")
            print(f"  Total players: {ex['total_players']}")
            print(f"  Players with known cards: {ex['known_cards']}")
            print("  Sample hands:")
            for hand in ex['hands']:
                print(f"    Player {hand.get('player_idx')}: {hand.get('hole_cards')} (seat {hand.get('seat')})")
    
    return table_hands

def analyze_actions_for_cards(file_path, sample_size=100):
    """Analyze actions to see if we can extract all players' cards from showdown."""
    print(f"\n=== Analyzing Actions for Card Information ===")
    
    showdown_hands = []
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            
            try:
                data = json.loads(line.strip())
                actions = data.get('actions', [])
                
                # Look for showdown actions (sm = show)
                showdown_actions = [a for a in actions if a.startswith('p') and 'sm' in a]
                
                if showdown_actions:
                    showdown_hands.append({
                        'table': data.get('table'),
                        'player_idx': data.get('player_idx'),
                        'hole_cards': data.get('hole_cards'),
                        'showdown_actions': showdown_actions,
                        'all_actions': actions
                    })
            except json.JSONDecodeError:
                continue
    
    print(f"Found {len(showdown_hands)} hands with showdown actions")
    
    if showdown_hands:
        print("\nExample showdown hands:")
        for hand in showdown_hands[:3]:
            print(f"\nTable: {hand['table'][:20]}...")
            print(f"Player {hand['player_idx']}: {hand['hole_cards']}")
            print(f"Showdown actions: {hand['showdown_actions']}")
    
    return showdown_hands

def check_phh_format_correctly():
    """Check PHH format more carefully."""
    print(f"\n=== Checking PHH Format ===")
    
    phh_files = [
        "data/raw/phh-dataset/data/antonius-blom-2009.phh",
        "data/raw/phh-dataset/data/pluribus/99/81.phh"
    ]
    
    for file_path in phh_files:
        if os.path.exists(file_path):
            print(f"\nAnalyzing {file_path}:")
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Find all deal actions
            lines = content.split('\n')
            deal_actions = []
            for line in lines:
                if 'd dh p' in line and '#' not in line:
                    # Extract the action part
                    action_match = re.search(r"'([^']+)'", line)
                    if action_match:
                        deal_actions.append(action_match.group(1))
            
            print(f"  Deal actions found: {deal_actions}")
            
            # Count known vs unknown
            known = sum(1 for a in deal_actions if '?' not in a)
            unknown = sum(1 for a in deal_actions if '?' in a)
            print(f"  Known cards: {known}")
            print(f"  Unknown cards: {unknown}")

def main():
    """Main analysis function."""
    print("=== Detailed Hand Structure Analysis ===\n")
    
    # Analyze the main dataset structure
    jsonl_file = "data/raw/poker_training_data.jsonl"
    if os.path.exists(jsonl_file):
        table_hands = analyze_hand_structure(jsonl_file, sample_size=5000)
        showdown_hands = analyze_actions_for_cards(jsonl_file, sample_size=500)
    
    # Check PHH format
    check_phh_format_correctly()
    
    print("\n=== Key Findings ===")
    print("1. JSONL dataset is player-centric - each row is one player's perspective")
    print("2. Even when a player has known cards, other players in the same hand may have unknown cards")
    print("3. Showdown actions (sm) can reveal some players' cards")
    print("4. PHH datasets have complete hand information but may have unknown cards for some players")
    print("5. To get ALL players' cards, we need either:")
    print("   - PHH datasets with complete information")
    print("   - Or reconstruct hands from JSONL by combining all players in a table")

if __name__ == "__main__":
    main() 