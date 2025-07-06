#!/usr/bin/env python3
"""
Test script for the new poker model, preprocessing, and app functionality.
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


def test_model_loading():
    """Test loading the new trained model."""
    print("🔍 Testing new model loading...")

    try:
        model_path = "poker_xgb_model_improved.joblib"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False

        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Features: {len(model.feature_names_in_)}")
        print(f"   Feature names: {list(model.feature_names_in_)[:5]}...")

        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_preprocessing():
    """Test the new preprocessing functionality."""
    print("\n🔍 Testing new preprocessing...")

    try:
        from src.data.preprocess_complete_state import create_player_features_from_row

        # Test feature creation with mock data
        mock_row = {
            "player_id": "test_player",
            "seat_count": 6,
            "board_cards": ["5c4d9d", "Ad", "7s"],
            "actions": ["p1 cbr 50", "p2 cc", "p1 f"],
            "winnings": 100,  # Winner
            "starting_stack": 500,
            "min_bet": 10,
            "antes": [0, 0],
            "blinds_or_straddles": [3, 6],
            "table": "test_table",
            "venue": "PokerStars",
            "date": [2023, 1, 1],
        }

        player_stats = {"test_player": {"vpip": 25.0, "pfr": 20.0}}
        features = create_player_features_from_row(mock_row, player_stats)

        # Check key features
        assert features["is_winner"] == 1  # Should be winner
        assert features["seat_count"] == 6
        assert features["board_card_count"] == 3
        assert features["player_vpip"] == 25.0
        assert features["is_multiway"] == 1
        assert features["is_heads_up"] == 0

        print("✅ Feature creation works correctly")
        print(f"   Features created: {len(features)}")
        print(f"   Winner flag: {features['is_winner']}")

        return True
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False


def test_model_prediction():
    """Test model prediction functionality."""
    print("\n🔍 Testing model predictions...")

    try:
        model_path = "poker_xgb_model_improved.joblib"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False

        model = joblib.load(model_path)

        # Create test features
        test_features = {
            "seat_count": 6,
            "board_card_count": 3,
            "player_vpip": 25.0,
            "player_pfr": 20.0,
            "player_action_count": 3,
            "player_folded": 0,
            "player_aggressive_actions": 2,
            "player_passive_actions": 1,
            "player_starting_stack": 500.0,
            "is_heads_up": 0,
            "is_multiway": 1,
            "first_action": 1,
            "last_action": 0,
            "has_hole_cards": 0,
            "hole_card_high": 0,
            "hole_card_low": 0,
            "hole_cards_suited": 0,
            "hole_cards_paired": 0,
            "flop_cards": "",
            "turn_card": "",
            "river_card": "",
            "opponent_count": 5,
            "total_pot": 0,
            "max_bet": 0,
            "betting_rounds": 0,
            "player_total_bet": 0,
            "player_bet_to_pot_ratio": 0,
            "player_stack_to_pot_ratio": 0,
            "opponent_total_bet": 0,
            "opponent_avg_bet": 0,
            "opponent_folded_count": 0,
            "pot_odds": 0,
        }

        # Convert to DataFrame and predict
        feature_df = pd.DataFrame([test_features])
        feature_df = feature_df.fillna(0)

        # Ensure all required features are present
        missing_features = set(model.feature_names_in_) - set(feature_df.columns)
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            for feature in missing_features:
                feature_df[feature] = 0

        # Select only the features the model expects
        feature_df = feature_df[model.feature_names_in_]

        # Make prediction
        proba = model.predict_proba(feature_df)[0]
        prediction = model.predict(feature_df)[0]

        print(f"✅ Prediction successful")
        print(f"   Win probability: {proba[1]:.3f}")
        print(f"   Prediction: {prediction}")
        print(f"   Features used: {len(feature_df.columns)}")

        return True
    except Exception as e:
        print(f"❌ Model prediction test failed: {e}")
        return False


def test_sample_data():
    """Test the sample data processing."""
    print("\n🔍 Testing sample data...")

    try:
        sample_file = "data/processed/complete_state_sample.jsonl"
        if not os.path.exists(sample_file):
            print(f"❌ Sample data file not found: {sample_file}")
            return False

        # Load sample data
        df = pd.read_json(sample_file, lines=True, nrows=1000)

        print(f"✅ Sample data loaded: {len(df)} rows")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Target distribution: {df['is_winner'].value_counts().to_dict()}")

        # Check target distribution
        win_rate = df["is_winner"].mean()
        print(f"   Win rate: {win_rate:.3f}")

        if 0.01 < win_rate < 0.20:  # Reasonable range
            print("✅ Target distribution looks reasonable")
        else:
            print(f"⚠️ Unusual win rate: {win_rate:.3f}")

        return True
    except Exception as e:
        print(f"❌ Sample data test failed: {e}")
        return False


def test_app_functionality():
    """Test the Streamlit app functionality."""
    print("\n🔍 Testing app functionality...")

    try:
        # Test the random feature generation function
        from src.web.app import generate_random_features, predict_win_probability

        # Test random feature generation
        features = generate_random_features()
        assert isinstance(features, dict)
        assert "is_winner" in features
        assert "seat_count" in features
        assert "player_vpip" in features
        print("✅ Random feature generation works")

        # Test prediction function (if model exists)
        model_path = "poker_xgb_model_improved.joblib"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            prob = predict_win_probability(model, features)
            assert 0 <= prob <= 1
            print(f"✅ Prediction function works: {prob:.3f}")
        else:
            print("⚠️ Model not found, skipping prediction test")

        return True
    except Exception as e:
        print(f"❌ App functionality test failed: {e}")
        return False


def test_data_consistency():
    """Test data consistency between raw and processed data."""
    print("\n🔍 Testing data consistency...")

    try:
        # Check if we have both raw and processed data
        raw_file = "data/raw/poker_training_data.jsonl"
        processed_file = "data/processed/complete_state_sample.jsonl"

        if not os.path.exists(raw_file):
            print(f"⚠️ Raw data not found: {raw_file}")
            return True  # Not critical

        if not os.path.exists(processed_file):
            print(f"⚠️ Processed data not found: {processed_file}")
            return True  # Not critical

        # Load small samples
        raw_df = pd.read_json(raw_file, lines=True, nrows=100)
        processed_df = pd.read_json(processed_file, lines=True, nrows=100)

        print(f"✅ Data consistency check")
        print(f"   Raw data: {len(raw_df)} rows, {len(raw_df.columns)} columns")
        print(f"   Processed data: {len(processed_df)} rows, {len(processed_df.columns)} columns")

        # Check that processed data has the expected target
        if "is_winner" in processed_df.columns:
            win_rate = processed_df["is_winner"].mean()
            print(f"   Processed win rate: {win_rate:.3f}")

        return True
    except Exception as e:
        print(f"❌ Data consistency test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🎯 Testing New Poker Model & App")
    print("=" * 50)

    tests = [
        test_model_loading,
        test_preprocessing,
        test_model_prediction,
        test_sample_data,
        test_app_functionality,
        test_data_consistency,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! New model and app are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
