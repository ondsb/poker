#!/usr/bin/env python3
"""
Validation script to test if the model gives reasonable predictions for predefined scenarios.
"""

import sys
import os
sys.path.append('src')

import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from pathlib import Path

def load_model_and_metadata():
    """Load the trained model and metadata."""
    model_path = Path("models/catboost_poker_model.cbm")
    metadata_path = Path("models/model_metadata.pkl")
    
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model files not found")
    
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return model, metadata

def create_features_for_scenario(player_cards, board_cards, context_features, metadata):
    """Create features for a given scenario."""
    feature_names = metadata.get('feature_names', [])
    seat_count = len(player_cards)
    
    features_list = []
    
    for player_idx in range(seat_count):
        features = {}
        
        # Set all features in the correct order
        for feature in feature_names:
            if feature == 'player_idx':
                features[feature] = player_idx
            elif feature == 'hole_card1':
                features[feature] = player_cards[player_idx][0]
            elif feature == 'hole_card2':
                features[feature] = player_cards[player_idx][1]
            elif feature.startswith('board_card') and feature != 'board_card_count':
                card_idx = int(feature.replace('board_card', '')) - 1
                features[feature] = board_cards[card_idx] if card_idx < len(board_cards) else "unknown"
            elif feature.startswith('player_') and feature.endswith(('_hole_card1', '_hole_card2')):
                parts = feature.split('_')
                if len(parts) >= 4:
                    other_player = int(parts[1])
                    card_num = int(parts[3][-1])
                    if other_player < len(player_cards):
                        features[feature] = player_cards[other_player][card_num - 1]
                    else:
                        features[feature] = "unknown"
                else:
                    features[feature] = "unknown"
            elif feature in context_features:
                features[feature] = context_features[feature]
            else:
                # Default values
                if feature in ['has_hole_cards', 'hole_cards_suited', 'hole_cards_paired', 
                              'is_heads_up', 'is_multiway', 'player_folded']:
                    features[feature] = 0
                elif feature in ['hole_card_high', 'hole_card_low', 'hole_card_gap', 
                                'board_card_count', 'player_action_count', 'player_aggressive_actions',
                                'player_passive_actions', 'player_starting_stack', 'player_seat',
                                'opponent_count', 'total_antes', 'total_blinds', 'seat_count',
                                'estimated_pot', 'min_bet']:
                    features[feature] = 0
                else:
                    features[feature] = 0
        
        # Calculate hand strength features
        hole_card1, hole_card2 = player_cards[player_idx]
        if hole_card1 != "unknown" and hole_card2 != "unknown":
            rank1, suit1 = hole_card1[:-1], hole_card1[-1]
            rank2, suit2 = hole_card2[:-1], hole_card2[-1]
            
            rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 
                        'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
            
            num1 = rank_map.get(rank1, 0)
            num2 = rank_map.get(rank2, 0)
            
            features['has_hole_cards'] = 1
            features['hole_card_high'] = max(num1, num2)
            features['hole_card_low'] = min(num1, num2)
            features['hole_cards_suited'] = 1 if suit1 == suit2 else 0
            features['hole_cards_paired'] = 1 if rank1 == rank2 else 0
            features['hole_card_gap'] = abs(num1 - num2)
        
        features['player_seat'] = player_idx + 1
        features_list.append(features)
    
    return features_list

def predict_scenario(model, metadata, player_cards, board_cards, context_features):
    """Make predictions for a given scenario."""
    features_list = create_features_for_scenario(player_cards, board_cards, context_features, metadata)
    
    # Create DataFrame
    feature_names = metadata.get('feature_names', [])
    df = pd.DataFrame(features_list)
    
    # Ensure all required columns
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Reorder columns
    df = df[feature_names]
    
    # Handle categorical features
    categorical_features = metadata.get('categorical_features', [])
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Make predictions
    cat_feature_indices = [i for i, col in enumerate(feature_names) if col in categorical_features]
    pool = Pool(df, cat_features=cat_feature_indices)
    raw_probs = model.predict_proba(pool)[:, 1]
    
    # Normalize
    raw_probs = np.array(raw_probs)
    normalized_probs = raw_probs / raw_probs.sum() if raw_probs.sum() > 0 else np.ones_like(raw_probs) / len(raw_probs)
    
    return raw_probs, normalized_probs

def test_predefined_scenarios():
    """Test the model with predefined scenarios that should have clear favorites."""
    
    print("🧪 Model Validation with Predefined Scenarios")
    print("=" * 60)
    
    # Load model
    model, metadata = load_model_and_metadata()
    print("✅ Model loaded successfully")
    
    # Define test scenarios
    scenarios = [
        {
            "name": "Pocket Aces vs Low Cards",
            "description": "AA should be heavily favored against 72o",
            "player_cards": [("As", "Ah"), ("7c", "2d")],
            "board_cards": ["unknown", "unknown", "unknown", "unknown", "unknown"],
            "expected_winner": 0,
            "expected_prob": 0.85  # AA should win ~85%+ of the time
        },
        {
            "name": "Pocket Kings vs Queens",
            "description": "KK should be favored against QQ",
            "player_cards": [("Ks", "Kh"), ("Qc", "Qh")],
            "board_cards": ["unknown", "unknown", "unknown", "unknown", "unknown"],
            "expected_winner": 0,
            "expected_prob": 0.80  # KK should win ~80%+ of the time
        },
        {
            "name": "Flush Draw vs High Cards",
            "description": "Flush draw should be favored on flush-heavy board",
            "player_cards": [("Ah", "Kh"), ("As", "Kd")],
            "board_cards": ["2h", "7h", "9h", "unknown", "unknown"],
            "expected_winner": 0,  # AhKh has flush draw
            "expected_prob": 0.70
        },
        {
            "name": "Set vs Overpair",
            "description": "Set should be heavily favored against overpair",
            "player_cards": [("7h", "7d"), ("As", "Ah")],
            "board_cards": ["7c", "2d", "9s", "unknown", "unknown"],
            "expected_winner": 0,  # 777 vs AA
            "expected_prob": 0.90
        },
        {
            "name": "Straight vs Two Pair",
            "description": "Straight should be favored against two pair",
            "player_cards": [("8h", "9h"), ("As", "Kh")],
            "board_cards": ["7d", "6c", "5s", "unknown", "unknown"],
            "expected_winner": 0,  # 56789 straight vs AK
            "expected_prob": 0.85
        },
        {
            "name": "Three Players - Clear Favorite",
            "description": "AA should be heavily favored in 3-way pot",
            "player_cards": [("As", "Ah"), ("Ks", "Kh"), ("7c", "2d")],
            "board_cards": ["unknown", "unknown", "unknown", "unknown", "unknown"],
            "expected_winner": 0,
            "expected_prob": 0.70  # AA should win ~70%+ in 3-way
        },
        {
            "name": "River Decision - Made Hand vs Draw",
            "description": "Made straight should be favored against flush draw",
            "player_cards": [("8h", "9h"), ("Ah", "Kh")],
            "board_cards": ["7d", "6c", "5s", "2h", "3c"],
            "expected_winner": 0,  # 56789 straight vs AhKh flush draw
            "expected_prob": 0.80
        }
    ]
    
    # Test each scenario
    results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n📊 Scenario {i+1}: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        
        # Context features
        seat_count = len(scenario['player_cards'])
        board_card_count = sum(1 for card in scenario['board_cards'] if card != "unknown")
        
        context_features = {
            'seat_count': seat_count,
            'board_card_count': board_card_count,
            'estimated_pot': 1000,
            'min_bet': 100,
            'player_action_count': 3,
            'player_aggressive_actions': 1,
            'player_passive_actions': 1,
            'player_starting_stack': 10000,
            'is_heads_up': 1 if seat_count == 2 else 0,
            'is_multiway': 1 if seat_count > 2 else 0,
            'opponent_count': seat_count - 1,
            'total_antes': 0,
            'total_blinds': 150,
            'player_folded': 0,
        }
        
        # Make predictions
        raw_probs, normalized_probs = predict_scenario(
            model, metadata, 
            scenario['player_cards'], 
            scenario['board_cards'], 
            context_features
        )
        
        # Display results
        print(f"   Results:")
        for j, (cards, raw_prob, norm_prob) in enumerate(zip(scenario['player_cards'], raw_probs, normalized_probs)):
            print(f"     Player {j+1} ({cards[0]} {cards[1]}): {raw_prob:.1%} raw, {norm_prob:.1%} normalized")
        
        # Check if prediction matches expectation
        predicted_winner = np.argmax(normalized_probs)
        expected_winner = scenario['expected_winner']
        predicted_prob = normalized_probs[expected_winner]
        expected_prob = scenario['expected_prob']
        
        is_correct = predicted_winner == expected_winner
        is_reasonable = predicted_prob >= expected_prob * 0.8  # Allow 20% tolerance
        
        status = "✅ PASS" if is_correct and is_reasonable else "❌ FAIL"
        print(f"   Expected: Player {expected_winner + 1} wins with {expected_prob:.1%} probability")
        print(f"   Predicted: Player {predicted_winner + 1} wins with {predicted_prob:.1%} probability")
        print(f"   Status: {status}")
        
        results.append({
            'scenario': scenario['name'],
            'expected_winner': expected_winner,
            'predicted_winner': predicted_winner,
            'expected_prob': expected_prob,
            'predicted_prob': predicted_prob,
            'is_correct': is_correct,
            'is_reasonable': is_reasonable,
            'pass': is_correct and is_reasonable
        })
    
    # Summary
    print(f"\n📈 Validation Summary")
    print("=" * 60)
    
    passed = sum(1 for r in results if r['pass'])
    total = len(results)
    
    print(f"Scenarios tested: {total}")
    print(f"Scenarios passed: {passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All scenarios passed! Model predictions are reasonable.")
    else:
        print("⚠️ Some scenarios failed. Model may need improvement.")
        for result in results:
            if not result['pass']:
                print(f"   ❌ {result['scenario']}: Expected P{result['expected_winner']+1}, got P{result['predicted_winner']+1}")
    
    return results

if __name__ == "__main__":
    test_predefined_scenarios() 