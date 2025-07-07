#!/usr/bin/env python3
"""
Analyze showdown patterns in 6-player hands to understand card revelation.
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path

def analyze_6player_showdown_patterns(input_file: str, max_hands: int = 1000000):
    """Analyze showdown patterns in 6-player hands."""
    print(f"🔍 Analyzing 6-player showdown patterns")
    print(f"📁 Input: {input_file}")
    print(f"📊 Max hands to scan: {max_hands:,}")
    print("-" * 60)
    
    total_hands = 0
    hands_6player = 0
    hands_with_showdown = 0
    showdown_patterns = Counter()
    known_cards_per_hand = Counter()
    sample_hands = []
    
    with open(input_file, 'r') as infile:
        for i, line in enumerate(infile):
            if i >= max_hands:
                break
                
            total_hands += 1
            try:
                hand = json.loads(line.strip())
                actions = hand.get('actions', [])
                
                # Count players in this hand
                players_in_hand = set()
                for action in actions:
                    if action.startswith('d dh p'):
                        match = re.match(r'd dh p(\d+)', action)
                        if match:
                            players_in_hand.add(match.group(1))
                
                # Only process 6-player hands
                if len(players_in_hand) != 6:
                    continue
                
                hands_6player += 1
                
                # Check for showdown actions
                showdown_actions = []
                known_cards_in_hand = []
                
                for action in actions:
                    if 'sm' in action:  # showdown action
                        showdown_actions.append(action)
                        match = re.match(r'p(\d+)\s+sm\s+(\w+)', action)
                        if match:
                            player_num = match.group(1)
                            cards = match.group(2)
                            
                            if cards != '????':
                                known_cards_in_hand.append({
                                    'player': player_num,
                                    'cards': cards,
                                    'action': action
                                })
                
                if showdown_actions:
                    hands_with_showdown += 1
                    known_count = len(known_cards_in_hand)
                    known_cards_per_hand[known_count] += 1
                    
                    # Record showdown pattern
                    pattern = f"{known_count}/6 known"
                    showdown_patterns[pattern] += 1
                    
                    # Collect sample hands
                    if len(sample_hands) < 10:
                        sample_hands.append({
                            'hand_index': i,
                            'known_count': known_count,
                            'known_cards': known_cards_in_hand,
                            'showdown_actions': showdown_actions,
                            'actions': actions[:20]  # First 20 actions
                        })
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing hand {i}: {e}")
                continue
            
            if i % 100000 == 0:
                print(f"  Processed {i:,} hands...")
                print(f"    6-player hands: {hands_6player}")
                print(f"    6-player showdowns: {hands_with_showdown}")
    
    print(f"\n📊 Analysis Results:")
    print(f"  ✅ Total hands processed: {total_hands:,}")
    print(f"  ✅ 6-player hands: {hands_6player:,}")
    print(f"  ✅ 6-player hands with showdown: {hands_with_showdown:,}")
    print(f"  📈 6-player rate: {100*hands_6player/total_hands:.2f}%")
    print(f"  📈 6-player showdown rate: {100*hands_with_showdown/hands_6player:.2f}%")
    
    print(f"\n🃏 Known Cards Distribution in 6-Player Showdowns:")
    for known_count in sorted(known_cards_per_hand.keys()):
        count = known_cards_per_hand[known_count]
        percentage = 100 * count / hands_with_showdown
        print(f"  {known_count}/6 players known: {count:,} hands ({percentage:.1f}%)")
    
    print(f"\n📋 Sample 6-Player Showdown Hands:")
    for i, hand in enumerate(sample_hands):
        print(f"  Hand {i+1}: {hand['known_count']}/6 players known")
        print(f"    Known cards: {hand['known_cards']}")
        print(f"    Showdown actions: {hand['showdown_actions']}")
        print(f"    Actions: {hand['actions']}")
        print()
    
    # Save detailed analysis
    analysis = {
        "total_hands_processed": total_hands,
        "hands_6player": hands_6player,
        "hands_with_showdown": hands_with_showdown,
        "known_cards_distribution": dict(known_cards_per_hand),
        "showdown_patterns": dict(showdown_patterns),
        "sample_hands": sample_hands
    }
    
    output_file = Path("data/raw/6player_showdown_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"💾 Detailed analysis saved to: {output_file}")

def main():
    analyze_6player_showdown_patterns("data/raw/poker_training_data.jsonl", max_hands=5000000)

if __name__ == "__main__":
    main() 