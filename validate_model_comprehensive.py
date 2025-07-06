#!/usr/bin/env python3
"""
Comprehensive model validation script for poker win probability prediction.
Tests the model on various scenarios including clear winners, edge cases, and realistic game situations.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any
import warnings

warnings.filterwarnings("ignore")

def load_model():
    """Load the trained model and metadata."""
    model_path = Path("models/catboost_poker_model.cbm")
    metadata_path = Path("models/model_metadata.pkl")
    
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model files not found")
    
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    return model, metadata

def create_test_scenario(
    player_cards: List[Tuple[str, str]], 
    board_cards: List[str], 
    context: Dict[str, Any]
) -> pd.DataFrame:
    """Create a test scenario DataFrame for prediction."""
    model, metadata = load_model()
    feature_names = metadata.get("feature_names", [])
    
    features_list = []
    seat_count = len(player_cards)
    
    for player_idx in range(seat_count):
        features = {}
        
        # Set all features in correct order
        for feature in feature_names:
            if feature == "player_idx":
                features[feature] = player_idx
            elif feature == "hole_card1":
                features[feature] = player_cards[player_idx][0]
            elif feature == "hole_card2":
                features[feature] = player_cards[player_idx][1]
            elif feature.startswith("board_card") and feature != "board_card_count":
                card_idx = int(feature.replace("board_card", "")) - 1
                features[feature] = board_cards[card_idx] if card_idx < len(board_cards) else "unknown"
            elif feature.startswith("player_") and feature.endswith(("_hole_card1", "_hole_card2")):
                parts = feature.split("_")
                if len(parts) >= 4:
                    other_player = int(parts[1])
                    card_num = int(parts[3][-1])
                    if other_player < len(player_cards):
                        features[feature] = player_cards[other_player][card_num - 1]
                    else:
                        features[feature] = "unknown"
                else:
                    features[feature] = "unknown"
            elif feature in context:
                features[feature] = context[feature]
            else:
                # Default values
                if feature in ["has_hole_cards", "hole_cards_suited", "hole_cards_paired", 
                              "is_heads_up", "is_multiway", "player_folded"]:
                    features[feature] = 0
                elif feature in ["hole_card_high", "hole_card_low", "hole_card_gap", 
                               "board_card_count", "player_action_count", "player_aggressive_actions",
                               "player_passive_actions", "player_starting_stack", "player_seat",
                               "opponent_count", "total_antes", "total_blinds", "seat_count",
                               "estimated_pot", "min_bet"]:
                    features[feature] = 0
                else:
                    features[feature] = 0
        
        # Calculate hand features
        hole_card1, hole_card2 = player_cards[player_idx]
        if hole_card1 != "unknown" and hole_card2 != "unknown":
            rank1, suit1 = parse_card(hole_card1)
            rank2, suit2 = parse_card(hole_card2)
            
            if rank1 and rank2:
                rank_map = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
                           "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
                num1, num2 = rank_map.get(rank1, 0), rank_map.get(rank2, 0)
                
                features.update({
                    "has_hole_cards": 1,
                    "hole_card_high": max(num1, num2),
                    "hole_card_low": min(num1, num2),
                    "hole_cards_suited": 1 if suit1 == suit2 else 0,
                    "hole_cards_paired": 1 if rank1 == rank2 else 0,
                    "hole_card_gap": abs(num1 - num2),
                })
        
        features["player_seat"] = player_idx + 1
        features_list.append(features)
    
    df = pd.DataFrame(features_list)
    
    # Ensure all required columns exist
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Reorder columns
    df = df[feature_names]
    
    # Handle categorical features
    categorical_features = metadata.get("categorical_features", [])
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df

def parse_card(card_str):
    """Parse a card string like 'As' into rank and suit."""
    if not card_str or card_str == "unknown":
        return None, None
    if len(card_str) < 2:
        return None, None
    rank = card_str[:-1]
    suit = card_str[-1]
    return rank, suit

def test_clear_winners():
    """Test scenarios with clear winners."""
    print("=== Testing Clear Winner Scenarios ===\n")
    
    scenarios = [
        {
            "name": "Royal Flush vs High Card",
            "player_cards": [("Ah", "Kh"), ("2c", "7d")],
            "board_cards": ["Qh", "Jh", "Th"],
            "expected_winner": 0,
            "description": "Player 1 has royal flush draw, Player 2 has high card"
        },
        {
            "name": "Aces vs Kings",
            "player_cards": [("Ah", "Ad"), ("Kh", "Kd")],
            "board_cards": ["2c", "7s", "9h"],
            "expected_winner": 0,
            "description": "Aces should beat Kings"
        },
        {
            "name": "Straight vs Pair",
            "player_cards": [("Ah", "Kh"), ("2c", "2d")],
            "board_cards": ["Qh", "Jh", "Th"],
            "expected_winner": 0,
            "description": "Straight should beat pair"
        },
        {
            "name": "Flush vs Two Pair",
            "player_cards": [("Ah", "Kh"), ("2c", "2d")],
            "board_cards": ["Qh", "Jh", "Th"],
            "expected_winner": 0,
            "description": "Flush should beat two pair"
        },
        {
            "name": "Set vs Overpair",
            "player_cards": [("Ah", "Ad"), ("Kh", "Kd")],
            "board_cards": ["Ac", "2s", "9h"],
            "expected_winner": 0,
            "description": "Set of Aces should beat overpair Kings"
        }
    ]
    
    context = {
        "seat_count": 2,
        "board_card_count": 3,
        "estimated_pot": 1000,
        "min_bet": 100,
        "player_action_count": 3,
        "player_aggressive_actions": 1,
        "player_passive_actions": 1,
        "player_starting_stack": 10000,
        "is_heads_up": 1,
        "is_multiway": 0,
        "opponent_count": 1,
        "total_antes": 0,
        "total_blinds": 150,
        "player_folded": 0,
    }
    
    model, metadata = load_model()
    categorical_features = metadata.get("categorical_features", [])
    cat_feature_indices = [i for i, col in enumerate(metadata.get("feature_names", [])) 
                          if col in categorical_features]
    
    results = []
    
    for scenario in scenarios:
        print(f"Testing: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        
        try:
            df = create_test_scenario(scenario["player_cards"], scenario["board_cards"], context)
            pool = Pool(df, cat_features=cat_feature_indices)
            probs = model.predict_proba(pool)[:, 1]
            
            winner_idx = np.argmax(probs)
            expected_winner = scenario["expected_winner"]
            
            print(f"Player 1 cards: {scenario['player_cards'][0]}")
            print(f"Player 2 cards: {scenario['player_cards'][1]}")
            print(f"Board: {scenario['board_cards']}")
            print(f"Predicted winner: Player {winner_idx + 1} ({probs[winner_idx]:.1%})")
            print(f"Expected winner: Player {expected_winner + 1}")
            print(f"Player 1 probability: {probs[0]:.1%}")
            print(f"Player 2 probability: {probs[1]:.1%}")
            
            correct = winner_idx == expected_winner
            print(f"✅ Correct" if correct else f"❌ Wrong")
            print(f"Probability sum: {probs.sum():.1%}")
            print("-" * 60)
            
            results.append({
                "scenario": scenario["name"],
                "correct": correct,
                "winner_prob": probs[winner_idx],
                "expected_winner_prob": probs[expected_winner],
                "prob_sum": probs.sum()
            })
            
        except Exception as e:
            print(f"Error testing scenario: {e}")
            print("-" * 60)
    
    # Summary
    correct_count = sum(r["correct"] for r in results)
    print(f"\nSummary: {correct_count}/{len(results)} scenarios correct")
    print(f"Average winner probability: {np.mean([r['winner_prob'] for r in results]):.1%}")
    print(f"Average probability sum: {np.mean([r['prob_sum'] for r in results]):.1%}")
    
    return results

def test_edge_cases():
    """Test edge cases and unusual scenarios."""
    print("\n=== Testing Edge Cases ===\n")
    
    scenarios = [
        {
            "name": "All Unknown Cards",
            "player_cards": [("unknown", "unknown"), ("unknown", "unknown")],
            "board_cards": ["unknown", "unknown", "unknown"],
            "description": "All cards unknown - should be equal probabilities"
        },
        {
            "name": "Mixed Unknown Cards",
            "player_cards": [("Ah", "Kh"), ("unknown", "unknown")],
            "board_cards": ["Qh", "unknown", "unknown"],
            "description": "One player has known cards, other unknown"
        },
        {
            "name": "6 Players All Known",
            "player_cards": [
                ("Ah", "Kh"), ("Qh", "Jh"), ("Th", "9h"), 
                ("8h", "7h"), ("6h", "5h"), ("4h", "3h")
            ],
            "board_cards": ["2h", "2c", "2d", "2s", "As"],
            "description": "6 players with known cards, full board"
        },
        {
            "name": "Heads Up Preflop",
            "player_cards": [("Ah", "Kh"), ("Qh", "Jh")],
            "board_cards": ["unknown", "unknown", "unknown", "unknown", "unknown"],
            "description": "Heads up preflop situation"
        }
    ]
    
    context = {
        "seat_count": 2,
        "board_card_count": 3,
        "estimated_pot": 1000,
        "min_bet": 100,
        "player_action_count": 3,
        "player_aggressive_actions": 1,
        "player_passive_actions": 1,
        "player_starting_stack": 10000,
        "is_heads_up": 1,
        "is_multiway": 0,
        "opponent_count": 1,
        "total_antes": 0,
        "total_blinds": 150,
        "player_folded": 0,
    }
    
    model, metadata = load_model()
    categorical_features = metadata.get("categorical_features", [])
    cat_feature_indices = [i for i, col in enumerate(metadata.get("feature_names", [])) 
                          if col in categorical_features]
    
    for scenario in scenarios:
        print(f"Testing: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        
        try:
            # Adjust context for multi-player scenarios
            if len(scenario["player_cards"]) > 2:
                context.update({
                    "seat_count": len(scenario["player_cards"]),
                    "is_heads_up": 0,
                    "is_multiway": 1,
                    "opponent_count": len(scenario["player_cards"]) - 1
                })
            
            df = create_test_scenario(scenario["player_cards"], scenario["board_cards"], context)
            pool = Pool(df, cat_features=cat_feature_indices)
            probs = model.predict_proba(pool)[:, 1]
            
            print(f"Number of players: {len(scenario['player_cards'])}")
            for i, (cards, prob) in enumerate(zip(scenario["player_cards"], probs)):
                print(f"Player {i+1} ({cards[0]} {cards[1]}): {prob:.1%}")
            
            print(f"Probability sum: {probs.sum():.1%}")
            print(f"Probability range: {probs.min():.1%} - {probs.max():.1%}")
            print("-" * 60)
            
        except Exception as e:
            print(f"Error testing scenario: {e}")
            print("-" * 60)

def test_probability_distribution():
    """Test probability distribution characteristics."""
    print("\n=== Testing Probability Distribution ===\n")
    
    # Generate 100 random scenarios
    np.random.seed(42)
    
    model, metadata = load_model()
    categorical_features = metadata.get("categorical_features", [])
    cat_feature_indices = [i for i, col in enumerate(metadata.get("feature_names", [])) 
                          if col in categorical_features]
    
    all_probs = []
    all_sums = []
    
    card_options = ["Ah", "Kh", "Qh", "Jh", "Th", "9h", "8h", "7h", "6h", "5h", "4h", "3h", "2h",
                   "Ad", "Kd", "Qd", "Jd", "Td", "9d", "8d", "7d", "6d", "5d", "4d", "3d", "2d"]
    
    for i in range(100):
        # Random 2-player scenario
        used_cards = set()
        player_cards = []
        
        for _ in range(2):
            card1 = np.random.choice([c for c in card_options if c not in used_cards])
            used_cards.add(card1)
            card2 = np.random.choice([c for c in card_options if c not in used_cards])
            used_cards.add(card2)
            player_cards.append((card1, card2))
        
        # Random board (0-5 cards)
        board_count = np.random.randint(0, 6)
        board_cards = []
        for j in range(5):
            if j < board_count:
                board_card = np.random.choice([c for c in card_options if c not in used_cards])
                used_cards.add(board_card)
                board_cards.append(board_card)
            else:
                board_cards.append("unknown")
        
        context = {
            "seat_count": 2,
            "board_card_count": board_count,
            "estimated_pot": np.random.randint(500, 5000),
            "min_bet": np.random.randint(50, 500),
            "player_action_count": np.random.randint(1, 8),
            "player_aggressive_actions": np.random.randint(0, 4),
            "player_passive_actions": np.random.randint(0, 4),
            "player_starting_stack": np.random.randint(5000, 50000),
            "is_heads_up": 1,
            "is_multiway": 0,
            "opponent_count": 1,
            "total_antes": 0,
            "total_blinds": 150,
            "player_folded": 0,
        }
        
        try:
            df = create_test_scenario(player_cards, board_cards, context)
            pool = Pool(df, cat_features=cat_feature_indices)
            probs = model.predict_proba(pool)[:, 1]
            
            all_probs.extend(probs)
            all_sums.append(probs.sum())
            
        except Exception as e:
            continue
    
    all_probs = np.array(all_probs)
    all_sums = np.array(all_sums)
    
    print(f"Total predictions: {len(all_probs)}")
    print(f"Probability statistics:")
    print(f"  Mean: {all_probs.mean():.1%}")
    print(f"  Median: {np.median(all_probs):.1%}")
    print(f"  Std: {all_probs.std():.1%}")
    print(f"  Min: {all_probs.min():.1%}")
    print(f"  Max: {all_probs.max():.1%}")
    print(f"  Q25: {np.percentile(all_probs, 25):.1%}")
    print(f"  Q75: {np.percentile(all_probs, 75):.1%}")
    
    print(f"\nProbability sum statistics:")
    print(f"  Mean: {all_sums.mean():.1%}")
    print(f"  Median: {np.median(all_sums):.1%}")
    print(f"  Std: {all_sums.std():.1%}")
    print(f"  Min: {all_sums.min():.1%}")
    print(f"  Max: {all_sums.max():.1%}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Probability distribution
    ax1.hist(all_probs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Win Probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Win Probabilities')
    ax1.grid(True, alpha=0.3)
    
    # Probability sum distribution
    ax2.hist(all_sums, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Sum of Probabilities')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Probability Sums')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved to: model_validation_results.png")

def main():
    """Run comprehensive validation."""
    print("🎯 Comprehensive Poker Model Validation")
    print("=" * 60)
    
    try:
        # Test clear winners
        clear_winner_results = test_clear_winners()
        
        # Test edge cases
        test_edge_cases()
        
        # Test probability distribution
        test_probability_distribution()
        
        print("\n✅ Validation completed successfully!")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 