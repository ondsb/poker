#!/usr/bin/env python3
"""
Extract 6-player hands where at least 2-3 players' cards are known at showdown.
This provides a realistic dataset for training win probability models.
"""

import json
import re
from pathlib import Path
import argparse
import time

def extract_6player_partial_known(input_file: str, output_file: str, min_known_players: int = 2, max_hands: int = 10000000):
    """Extract 6-player hands with at least N players' cards known at showdown."""
    print(f"🔍 Extracting 6-player hands with at least {min_known_players} players' cards known")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_file}")
    print(f"📊 Min known players: {min_known_players}")
    print(f"📊 Max hands to scan: {max_hands:,}")
    print("-" * 60)
    
    extracted_hands = []
    total_hands = 0
    hands_6player = 0
    hands_with_showdown = 0
    hands_with_sufficient_known = 0
    
    start_time = time.time()
    
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
                    
                    # Check if we have enough known players
                    if len(known_cards_in_hand) >= min_known_players:
                        hands_with_sufficient_known += 1
                        
                        # Add metadata
                        hand['known_cards'] = known_cards_in_hand
                        hand['known_cards_count'] = len(known_cards_in_hand)
                        hand['total_players'] = 6
                        hand['showdown_actions'] = showdown_actions
                        hand['min_known_players'] = min_known_players
                        
                        extracted_hands.append(hand)
                        
                        if len(extracted_hands) % 1000 == 0:
                            print(f"  Found {len(extracted_hands)} hands with at least {min_known_players} known players...")
                        
                        # Limit to first 50000 hands
                        if len(extracted_hands) >= 50000:
                            break
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing hand {i}: {e}")
                continue
            
            if i % 100000 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {i:,} hands in {elapsed:.1f}s...")
                print(f"    6-player hands: {hands_6player}")
                print(f"    6-player showdowns: {hands_with_showdown}")
                print(f"    Hands with {min_known_players}+ known: {hands_with_sufficient_known}")
    
    total_time = time.time() - start_time
    
    print(f"\n📊 Extraction Results:")
    print(f"  ✅ Total hands processed: {total_hands:,}")
    print(f"  ✅ 6-player hands: {hands_6player:,}")
    print(f"  ✅ 6-player hands with showdown: {hands_with_showdown:,}")
    print(f"  ✅ Hands with {min_known_players}+ known players: {hands_with_sufficient_known:,}")
    print(f"  📈 6-player rate: {100*hands_6player/total_hands:.2f}%")
    print(f"  📈 6-player showdown rate: {100*hands_with_showdown/hands_6player:.2f}%")
    print(f"  📈 Sufficient known rate: {100*hands_with_sufficient_known/hands_with_showdown:.2f}%")
    print(f"  ⏱️ Processing time: {total_time:.1f}s")
    
    if extracted_hands:
        # Save extracted hands
        print(f"\n💾 Saving {len(extracted_hands)} hands to {output_file}")
        with open(output_file, 'w') as outfile:
            for hand in extracted_hands:
                outfile.write(json.dumps(hand) + '\n')
        
        # Analyze the extracted data
        analyze_extracted_partial_data(extracted_hands, min_known_players)
        
        # Create summary
        summary = {
            "total_hands_processed": total_hands,
            "hands_6player": hands_6player,
            "hands_with_showdown": hands_with_showdown,
            "hands_with_sufficient_known": hands_with_sufficient_known,
            "min_known_players": min_known_players,
            "extracted_hands": len(extracted_hands),
            "processing_time_seconds": total_time,
            "sample_hands": extracted_hands[:3]
        }
        
        summary_file = Path(output_file).parent / f"6player_partial_{min_known_players}known_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Summary saved to: {summary_file}")
    
    return extracted_hands

def analyze_extracted_partial_data(hands, min_known_players):
    """Analyze the extracted partial known hands data."""
    print(f"\n📈 Analyzing {min_known_players}+ Known Players Data")
    
    if not hands:
        print("  No hands to analyze")
        return
    
    # Count card patterns
    card_patterns = {}
    action_counts = []
    known_players_dist = {}
    player_winners = {}
    
    for hand in hands:
        actions = hand.get('actions', [])
        action_counts.append(len(actions))
        
        # Count known players distribution
        known_count = hand['known_cards_count']
        known_players_dist[known_count] = known_players_dist.get(known_count, 0) + 1
        
        # Count known card patterns
        for card_info in hand['known_cards']:
            cards = card_info['cards']
            card_patterns[cards] = card_patterns.get(cards, 0) + 1
        
        # Try to identify winners (this is approximate)
        # Look for the last showdown action to determine winner
        if hand.get('showdown_actions'):
            last_showdown = hand['showdown_actions'][-1]
            match = re.match(r'p(\d+)', last_showdown)
            if match:
                winner = match.group(1)
                player_winners[winner] = player_winners.get(winner, 0) + 1
    
    print(f"  🃏 Total hands: {len(hands)}")
    print(f"  🃏 Average actions per hand: {sum(action_counts)/len(action_counts):.1f}")
    
    print(f"  🃏 Known players distribution:")
    for known_count in sorted(known_players_dist.keys()):
        count = known_players_dist[known_count]
        percentage = 100 * count / len(hands)
        print(f"    {known_count}/6 players known: {count:,} hands ({percentage:.1f}%)")
    
    if player_winners:
        print(f"  🃏 Player win distribution (approximate):")
        for player, wins in sorted(player_winners.items()):
            print(f"    Player {player}: {wins} wins")
    
    print(f"  🃏 Top 10 most common card combinations:")
    sorted_patterns = sorted(card_patterns.items(), key=lambda x: x[1], reverse=True)
    for cards, count in sorted_patterns[:10]:
        print(f"    {cards}: {count} times")
    
    # Show sample hands
    print(f"\n📋 Sample hands:")
    for i, hand in enumerate(hands[:2]):
        print(f"  Hand {i+1} (Table: {hand.get('table', 'unknown')}):")
        print(f"    {hand['known_cards_count']}/6 players known: {hand['known_cards']}")
        print(f"    Actions: {hand['actions'][:10]}...")
        print()

def main():
    parser = argparse.ArgumentParser(description="Extract 6-player hands with partial known cards")
    parser.add_argument("--input", type=str, default="data/raw/poker_training_data.jsonl",
                       help="Input JSONL file")
    parser.add_argument("--output", type=str, default="data/raw/6player_partial_known.jsonl",
                       help="Output JSONL file")
    parser.add_argument("--min-known", type=int, default=2,
                       help="Minimum number of players with known cards")
    parser.add_argument("--max-hands", type=int, default=10000000,
                       help="Maximum hands to scan")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    extract_6player_partial_known(
        input_file=args.input,
        output_file=args.output,
        min_known_players=args.min_known,
        max_hands=args.max_hands
    )

if __name__ == "__main__":
    main() 