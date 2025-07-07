#!/usr/bin/env python3
"""
Analyze showdown actions to find hands with known hole cards.
Extract a subset of hands where cards are revealed for training.
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
import argparse
import time

class ShowdownAnalyzer:
    def __init__(self, file_path: str, max_hands: int = 1000000):
        self.file_path = file_path
        self.max_hands = max_hands
        self.hands_with_known_cards = []
        self.showdown_stats = defaultdict(int)
        
    def analyze_showdown_subset(self):
        """Analyze showdown actions and find hands with known hole cards."""
        print(f"🔍 Analyzing showdown actions for known hole cards")
        print(f"📁 File: {self.file_path}")
        print(f"📊 Max hands to scan: {self.max_hands:,}")
        print("-" * 60)
        
        start_time = time.time()
        
        total_hands = 0
        hands_with_showdown = 0
        hands_with_known_cards = 0
        
        with open(self.file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_hands:
                    break
                    
                total_hands += 1
                try:
                    hand = json.loads(line.strip())
                    actions = hand.get('actions', [])
                    
                    # Check for showdown actions
                    showdown_actions = []
                    for action in actions:
                        if 'sm' in action:  # showdown action
                            showdown_actions.append(action)
                    
                    if showdown_actions:
                        hands_with_showdown += 1
                        
                        # Check if any showdown action has real cards
                        has_known_cards = False
                        known_cards_in_hand = []
                        
                        for action in showdown_actions:
                            match = re.match(r'p(\d+)\s+sm\s+(\w+)', action)
                            if match:
                                player_num = match.group(1)
                                cards = match.group(2)
                                
                                if cards != '????':
                                    has_known_cards = True
                                    known_cards_in_hand.append({
                                        'player': player_num,
                                        'cards': cards,
                                        'action': action
                                    })
                        
                        # Count total players in the hand
                        total_players = 0
                        for action in actions:
                            if action.startswith('d dh p'):
                                total_players += 1
                        
                        # Only keep hands where ALL players' cards are known at showdown
                        if has_known_cards and len(known_cards_in_hand) >= total_players:
                            hands_with_known_cards += 1
                            self.hands_with_known_cards.append({
                                'hand_index': i,
                                'table': hand.get('table', 'unknown'),
                                'actions': actions,
                                'known_cards': known_cards_in_hand,
                                'showdown_actions': showdown_actions,
                                'total_players': total_players,
                                'known_players': len(known_cards_in_hand)
                            })
                            
                            # Limit to first 5000 hands with all known cards for training
                            if len(self.hands_with_known_cards) >= 5000:
                                break
                    
                    if i % 100000 == 0:
                        print(f"  Processed {i:,} hands...")
                        print(f"    Hands with showdown: {hands_with_showdown}")
                        print(f"    Hands with known cards: {hands_with_known_cards}")
                        
                except json.JSONDecodeError:
                    continue
        
        total_time = time.time() - start_time
        
        print(f"\n📊 Analysis Results:")
        print(f"  ✅ Total hands processed: {total_hands:,}")
        print(f"  ✅ Hands with showdown: {hands_with_showdown:,}")
        print(f"  ✅ Hands with known cards: {hands_with_known_cards:,}")
        print(f"  📈 Showdown rate: {100*hands_with_showdown/total_hands:.2f}%")
        print(f"  📈 Known cards rate: {100*hands_with_known_cards/total_hands:.2f}%")
        print(f"  📈 Known cards in showdown: {100*hands_with_known_cards/max(hands_with_showdown,1):.2f}%")
        print(f"  📈 Hands with ALL players known: {hands_with_known_cards}")
        print(f"  ⏱️ Analysis time: {total_time:.1f}s")
        
        if self.hands_with_known_cards:
            self._analyze_known_cards_patterns()
            self._save_known_cards_subset()
        
    def _analyze_known_cards_patterns(self):
        """Analyze patterns in hands with known cards."""
        print(f"\n🃏 Analyzing Known Cards Patterns")
        
        # Count card patterns
        card_patterns = Counter()
        player_counts = Counter()
        action_counts = []
        
        for hand in self.hands_with_known_cards:
            actions = hand['actions']
            action_counts.append(len(actions))
            
            # Count players involved
            players_in_hand = set()
            for action in actions:
                if action.startswith('p'):
                    match = re.match(r'p(\d+)', action)
                    if match:
                        players_in_hand.add(match.group(1))
            player_counts[len(players_in_hand)] += 1
            
            # Count known card patterns
            for card_info in hand['known_cards']:
                cards = card_info['cards']
                card_patterns[cards] += 1
        
        print(f"  🃏 Total hands with ALL players known: {len(self.hands_with_known_cards)}")
        print(f"  🃏 Average actions per hand: {sum(action_counts)/len(action_counts):.1f}")
        print(f"  🃏 Player count distribution:")
        for player_count, count in sorted(player_counts.items()):
            print(f"    {player_count} players: {count} hands")
        
        # Show completeness stats
        if self.hands_with_known_cards:
            completeness_stats = {}
            for hand in self.hands_with_known_cards:
                total = hand.get('total_players', 0)
                known = hand.get('known_players', 0)
                if total > 0:
                    completeness = f"{known}/{total}"
                    completeness_stats[completeness] = completeness_stats.get(completeness, 0) + 1
            
            print(f"  🃏 Completeness distribution:")
            for completeness, count in sorted(completeness_stats.items()):
                print(f"    {completeness} players known: {count} hands")
        
        print(f"  🃏 Top 10 most common card combinations:")
        for cards, count in card_patterns.most_common(10):
            print(f"    {cards}: {count} times")
            
        # Show sample hands
        print(f"\n📋 Sample hands with known cards:")
        for i, hand in enumerate(self.hands_with_known_cards[:3]):
            print(f"  Hand {i+1} (Table: {hand['table']}):")
            print(f"    Known cards: {hand['known_cards']}")
            print(f"    Actions: {hand['actions'][:10]}...")
            print()
            
    def _save_known_cards_subset(self):
        """Save a subset of hands with known cards for training."""
        output_dir = Path("data/raw/known_cards_subset")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all hands with known cards
        output_file = output_dir / "hands_with_known_cards.jsonl"
        with open(output_file, 'w') as f:
            for hand in self.hands_with_known_cards:
                f.write(json.dumps(hand) + '\n')
        
        print(f"💾 Saved {len(self.hands_with_known_cards)} hands with known cards to:")
        print(f"   {output_file}")
        
        # Create a summary report
        summary = {
            "total_hands_with_known_cards": len(self.hands_with_known_cards),
            "sample_hands": self.hands_with_known_cards[:5],
            "file_path": str(output_file)
        }
        
        summary_file = output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"   {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze showdown actions for known hole cards")
    parser.add_argument("--file", type=str, default="data/raw/poker_training_data.jsonl",
                       help="Path to JSONL file")
    parser.add_argument("--max-hands", type=int, default=1000000,
                       help="Maximum number of hands to scan")
    
    args = parser.parse_args()
    
    analyzer = ShowdownAnalyzer(
        file_path=args.file,
        max_hands=args.max_hands
    )
    
    analyzer.analyze_showdown_subset()

if __name__ == "__main__":
    main() 