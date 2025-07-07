#!/usr/bin/env python3
"""
Comprehensive dataset validation script for 27GB poker dataset.
Analyzes data completeness and usability for model training.
"""

import json
import re
import random
import statistics
from collections import defaultdict, Counter
from pathlib import Path
import argparse
import time
from typing import Dict, List, Set, Tuple
import pandas as pd

class DatasetValidator:
    def __init__(self, file_path: str, sample_size: int = 10000, max_hands: int = 100000):
        self.file_path = file_path
        self.sample_size = sample_size
        self.max_hands = max_hands
        self.results = defaultdict(list)
        self.counters = defaultdict(int)
        self.samples = []
        
    def validate(self):
        """Main validation pipeline."""
        print(f"🔍 Starting comprehensive dataset validation")
        print(f"📁 File: {self.file_path}")
        print(f"📊 Sample size: {self.sample_size:,}")
        print(f"📊 Max hands to scan: {self.max_hands:,}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Phase 1: Basic statistics and sampling
        self._analyze_basic_stats()
        
        # Phase 2: Action pattern analysis
        self._analyze_action_patterns()
        
        # Phase 3: Hole card completeness
        self._analyze_hole_card_completeness()
        
        # Phase 4: Board card completeness
        self._analyze_board_card_completeness()
        
        # Phase 5: Player action completeness
        self._analyze_player_actions()
        
        # Phase 6: Data quality assessment
        self._assess_data_quality()
        
        # Phase 7: Generate recommendations
        self._generate_recommendations()
        
        total_time = time.time() - start_time
        print(f"\n⏱️ Total validation time: {total_time:.1f}s")
        
    def _analyze_basic_stats(self):
        """Analyze basic dataset statistics."""
        print("📊 Phase 1: Basic Statistics")
        
        total_lines = 0
        valid_hands = 0
        player_counts = []
        action_counts = []
        
        with open(self.file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_hands:
                    break
                    
                total_lines += 1
                try:
                    hand = json.loads(line.strip())
                    
                    # Check if hand has required fields
                    if 'actions' in hand and 'table' in hand:
                        valid_hands += 1
                        
                        # Count players (assuming each line is one player)
                        player_count = 1  # This will be refined in action analysis
                        player_counts.append(player_count)
                        
                        # Count actions
                        actions = hand.get('actions', [])
                        action_counts.append(len(actions))
                        
                        # Sample for detailed analysis
                        if len(self.samples) < self.sample_size:
                            self.samples.append(hand)
                            
                except json.JSONDecodeError:
                    continue
                    
                if i % 10000 == 0:
                    print(f"  Processed {i:,} lines...")
        
        self.counters['total_lines'] = total_lines
        self.counters['valid_hands'] = valid_hands
        self.counters['avg_actions_per_hand'] = statistics.mean(action_counts) if action_counts else 0
        self.counters['max_actions_per_hand'] = max(action_counts) if action_counts else 0
        
        print(f"  ✅ Total lines processed: {total_lines:,}")
        print(f"  ✅ Valid hands: {valid_hands:,}")
        print(f"  ✅ Average actions per hand: {self.counters['avg_actions_per_hand']:.1f}")
        print(f"  ✅ Max actions per hand: {self.counters['max_actions_per_hand']}")
        
    def _analyze_action_patterns(self):
        """Analyze action patterns in the dataset."""
        print("\n🎯 Phase 2: Action Pattern Analysis")
        
        action_patterns = Counter()
        deal_actions = []
        showdown_actions = []
        board_actions = []
        
        for hand in self.samples:
            actions = hand.get('actions', [])
            
            for action in actions:
                action_patterns[action] += 1
                
                # Deal actions
                if action.startswith('d dh'):
                    deal_actions.append(action)
                    
                # Showdown actions
                if 'sm' in action:
                    showdown_actions.append(action)
                    
                # Board actions
                if action.startswith('d db'):
                    board_actions.append(action)
        
        # Analyze deal actions
        print(f"  📋 Deal actions found: {len(deal_actions)}")
        if deal_actions:
            deal_samples = deal_actions[:10]
            print(f"  📋 Sample deal actions:")
            for action in deal_samples:
                print(f"    {action}")
        
        # Analyze showdown actions
        print(f"  📋 Showdown actions found: {len(showdown_actions)}")
        if showdown_actions:
            showdown_samples = showdown_actions[:10]
            print(f"  📋 Sample showdown actions:")
            for action in showdown_samples:
                print(f"    {action}")
        
        # Analyze board actions
        print(f"  📋 Board actions found: {len(board_actions)}")
        if board_actions:
            board_samples = board_actions[:10]
            print(f"  📋 Sample board actions:")
            for action in board_samples:
                print(f"    {action}")
        
        # Most common actions
        print(f"  📋 Top 10 most common actions:")
        for action, count in action_patterns.most_common(10):
            print(f"    {action}: {count}")
            
        self.counters['deal_actions'] = len(deal_actions)
        self.counters['showdown_actions'] = len(showdown_actions)
        self.counters['board_actions'] = len(board_actions)
        
    def _analyze_hole_card_completeness(self):
        """Analyze hole card completeness."""
        print("\n🃏 Phase 3: Hole Card Completeness")
        
        total_deals = 0
        known_cards = 0
        unknown_cards = 0
        real_card_samples = []
        
        for hand in self.samples:
            actions = hand.get('actions', [])
            
            for action in actions:
                if action.startswith('d dh'):
                    total_deals += 1
                    
                    # Parse deal action
                    match = re.match(r'd dh p(\d+)\s+(\w+)', action)
                    if match:
                        cards = match.group(2)
                        if cards == '????':
                            unknown_cards += 1
                        else:
                            known_cards += 1
                            if len(real_card_samples) < 10:
                                real_card_samples.append(action)
        
        self.counters['total_deals'] = total_deals
        self.counters['known_cards'] = known_cards
        self.counters['unknown_cards'] = unknown_cards
        self.counters['known_card_rate'] = known_cards / max(total_deals, 1)
        
        print(f"  🃏 Total deal actions: {total_deals}")
        print(f"  🃏 Known cards: {known_cards}")
        print(f"  🃏 Unknown cards: {unknown_cards}")
        print(f"  🃏 Known card rate: {self.counters['known_card_rate']:.2%}")
        
        if real_card_samples:
            print(f"  🃏 Sample real card deals:")
            for action in real_card_samples:
                print(f"    {action}")
        else:
            print(f"  ⚠️ No real card deals found in sample!")
            
    def _analyze_board_card_completeness(self):
        """Analyze board card completeness."""
        print("\n🃏 Phase 4: Board Card Completeness")
        
        total_board_actions = 0
        flop_actions = 0
        turn_actions = 0
        river_actions = 0
        
        for hand in self.samples:
            actions = hand.get('actions', [])
            
            for action in actions:
                if action.startswith('d db'):
                    total_board_actions += 1
                    
                    # Parse board action
                    match = re.match(r'd db (\w+)', action)
                    if match:
                        cards = match.group(1)
                        if len(cards) == 6:  # Flop: 3 cards
                            flop_actions += 1
                        elif len(cards) == 2:  # Turn/River: 1 card
                            if flop_actions > turn_actions:
                                turn_actions += 1
                            else:
                                river_actions += 1
        
        self.counters['total_board_actions'] = total_board_actions
        self.counters['flop_actions'] = flop_actions
        self.counters['turn_actions'] = turn_actions
        self.counters['river_actions'] = river_actions
        
        print(f"  🃏 Total board actions: {total_board_actions}")
        print(f"  🃏 Flop actions: {flop_actions}")
        print(f"  🃏 Turn actions: {turn_actions}")
        print(f"  🃏 River actions: {river_actions}")
        
    def _analyze_player_actions(self):
        """Analyze player action completeness."""
        print("\n👥 Phase 5: Player Action Analysis")
        
        player_actions = Counter()
        action_types = Counter()
        
        for hand in self.samples:
            actions = hand.get('actions', [])
            
            for action in actions:
                if action.startswith('p'):
                    player_actions[action] += 1
                    
                    # Categorize action types
                    if 'c' in action:
                        action_types['call'] += 1
                    if 'f' in action:
                        action_types['fold'] += 1
                    if 'r' in action or 'b' in action:
                        action_types['raise/bet'] += 1
                    if 'a' in action:
                        action_types['all-in'] += 1
        
        print(f"  👥 Total player actions: {sum(player_actions.values())}")
        print(f"  👥 Action type distribution:")
        for action_type, count in action_types.items():
            print(f"    {action_type}: {count}")
        
        print(f"  👥 Top 10 player actions:")
        for action, count in player_actions.most_common(10):
            print(f"    {action}: {count}")
            
        self.counters['total_player_actions'] = sum(player_actions.values())
        self.counters['action_types'] = dict(action_types)
        
    def _assess_data_quality(self):
        """Assess overall data quality."""
        print("\n📈 Phase 6: Data Quality Assessment")
        
        # Calculate quality metrics
        quality_score = 0
        max_score = 100
        
        # Hole card completeness (40 points)
        hole_card_score = min(40, self.counters['known_card_rate'] * 40)
        quality_score += hole_card_score
        
        # Action completeness (30 points)
        action_completeness = min(1.0, self.counters['total_player_actions'] / max(len(self.samples), 1))
        action_score = action_completeness * 30
        quality_score += action_score
        
        # Board completeness (20 points)
        board_completeness = min(1.0, self.counters['total_board_actions'] / max(len(self.samples), 1))
        board_score = board_completeness * 20
        quality_score += board_score
        
        # Data consistency (10 points)
        consistency_score = 10 if self.counters['valid_hands'] > 0 else 0
        quality_score += consistency_score
        
        self.counters['quality_score'] = quality_score
        self.counters['hole_card_score'] = hole_card_score
        self.counters['action_score'] = action_score
        self.counters['board_score'] = board_score
        self.counters['consistency_score'] = consistency_score
        
        print(f"  📈 Overall Quality Score: {quality_score:.1f}/100")
        print(f"  📈 Hole Card Completeness: {hole_card_score:.1f}/40")
        print(f"  📈 Action Completeness: {action_score:.1f}/30")
        print(f"  📈 Board Completeness: {board_score:.1f}/20")
        print(f"  📈 Data Consistency: {consistency_score:.1f}/10")
        
        # Quality rating
        if quality_score >= 80:
            rating = "Excellent"
        elif quality_score >= 60:
            rating = "Good"
        elif quality_score >= 40:
            rating = "Fair"
        else:
            rating = "Poor"
            
        print(f"  📈 Quality Rating: {rating}")
        
    def _generate_recommendations(self):
        """Generate recommendations for model training."""
        print("\n💡 Phase 7: Recommendations")
        
        quality_score = self.counters['quality_score']
        known_card_rate = self.counters['known_card_rate']
        
        print("  💡 Model Training Recommendations:")
        
        if known_card_rate < 0.01:
            print("    ⚠️ CRITICAL: Very low known hole card rate (<1%)")
            print("    💡 Consider:")
            print("      - Finding a dataset with known hole cards")
            print("      - Using only board cards and player actions for training")
            print("      - Training a model that predicts based on visible information only")
            
        elif known_card_rate < 0.1:
            print("    ⚠️ WARNING: Low known hole card rate (<10%)")
            print("    💡 Consider:")
            print("      - Training separate models for known vs unknown scenarios")
            print("      - Using the known cards as a premium feature")
            print("      - Focusing on board-based predictions")
            
        else:
            print("    ✅ Good hole card coverage")
            print("    💡 Can train comprehensive model with hole card information")
        
        if quality_score >= 70:
            print("    ✅ Dataset quality is sufficient for model training")
            print("    💡 Proceed with full feature engineering")
        elif quality_score >= 50:
            print("    ⚠️ Dataset quality is marginal")
            print("    💡 Consider data augmentation or alternative approaches")
        else:
            print("    ❌ Dataset quality is too low for reliable model training")
            print("    💡 Consider finding a better dataset")
            
        # Save detailed report
        self._save_report()
        
    def _save_report(self):
        """Save detailed validation report."""
        report_path = Path("data/validation_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": self.file_path,
            "sample_size": self.sample_size,
            "max_hands_scanned": self.max_hands,
            "counters": dict(self.counters),
            "recommendations": {
                "known_card_rate": self.counters['known_card_rate'],
                "quality_score": self.counters['quality_score'],
                "usable_for_training": self.counters['quality_score'] >= 50
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📄 Detailed report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Validate poker dataset completeness")
    parser.add_argument("--file", type=str, default="data/raw/poker_training_data.jsonl",
                       help="Path to JSONL file")
    parser.add_argument("--sample-size", type=int, default=10000,
                       help="Number of hands to sample for detailed analysis")
    parser.add_argument("--max-hands", type=int, default=100000,
                       help="Maximum number of hands to scan")
    
    args = parser.parse_args()
    
    validator = DatasetValidator(
        file_path=args.file,
        sample_size=args.sample_size,
        max_hands=args.max_hands
    )
    
    validator.validate()

if __name__ == "__main__":
    main() 