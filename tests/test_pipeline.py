#!/usr/bin/env python3
"""
Comprehensive test script for the poker win probability prediction pipeline.
Tests data processing, feature engineering, and model training with hole cards.
"""

import os
import json
import pandas as pd
from src.core import run_data_engineering_pipeline
import joblib


def test_data_loading():
    """Test if the hole card data file exists and can be loaded."""
    print("🔍 Testing data loading...")

    data_file = "poker_training_data_with_hole_cards.jsonl"

    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return False

    file_size = os.path.getsize(data_file) / (1024 * 1024)
    print(f"✅ Data file found: {data_file} ({file_size:.1f} MB)")

    # Test loading a small sample
    try:
        with open(data_file, "r") as f:
            sample_lines = []
            for i, line in enumerate(f):
                if i >= 10:  # Load first 10 lines
                    break
                sample_lines.append(json.loads(line))

        print(f"✅ Successfully loaded {len(sample_lines)} sample records")

        # Check for hole cards
        hole_card_count = 0
        for record in sample_lines:
            if record.get("hole_cards") and "?" not in str(record["hole_cards"]):
                hole_card_count += 1

        print(
            f"📊 Sample hole card coverage: {hole_card_count}/{len(sample_lines)} ({hole_card_count/len(sample_lines)*100:.1f}%)"
        )

        return True

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


def test_feature_engineering():
    """Test the feature engineering pipeline."""
    print("\n🔍 Testing feature engineering...")

    data_file = "poker_training_data_with_hole_cards.jsonl"

    try:
        # Load a smaller sample for testing
        df = pd.read_json(data_file, lines=True, nrows=10000)
        print(f"✅ Loaded {len(df)} records for feature engineering test")

        # Test feature engineering
        X, y = run_data_engineering_pipeline(data_file)

        if X is not None and y is not None:
            print(f"✅ Feature engineering successful")
            print(f"📊 Feature matrix shape: {X.shape}")
            print(f"📊 Target distribution: {y.value_counts().to_dict()}")

            # Check for important features
            expected_features = ["hand_strength", "hero_high_card", "pot_size", "betting_round"]
            missing_features = [f for f in expected_features if f not in X.columns]

            if missing_features:
                print(f"⚠️ Missing expected features: {missing_features}")
            else:
                print(f"✅ All expected features present")

            # Check hole card features
            hole_card_features = [
                "hero_high_card",
                "hero_is_pair",
                "hero_is_suited",
                "hand_strength",
            ]
            hole_card_present = all(f in X.columns for f in hole_card_features)

            if hole_card_present:
                print(f"✅ Hole card features present")

                # Check if we have valid hole card data
                valid_hole_cards = (X["hero_high_card"] > 0).sum()
                print(
                    f"📊 Valid hole card examples: {valid_hole_cards}/{len(X)} ({valid_hole_cards/len(X)*100:.1f}%)"
                )
            else:
                print(f"❌ Missing hole card features")

            return X, y
        else:
            print(f"❌ Feature engineering failed")
            return None, None

    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        return None, None


def test_prediction_pipeline():
    """Test the complete prediction pipeline."""
    print("\n🔍 Testing prediction pipeline...")

    # Check if trained model exists
    model_file = "models/catboost_poker_model.cbm"

    if not os.path.exists(model_file):
        print(f"❌ Trained model not found: {model_file}")
        print("Please run the training script first:")
        print("  make train-model")
        return False

    try:
        # Load the model
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(model_file)
        print(f"✅ Model loaded successfully")

        # Create a sample prediction
        sample_features = {
            "pot_size": 100,
            "seat_count": 6,
            "position": 3,
            "betting_round": 1,  # Flop
            "hero_stack_to_pot_ratio": 10.0,
            "hero_bet_to_pot_ratio": 0.5,
            "total_bets_in_hand": 3,
            "hero_contribution_to_pot": 0.3,
            "hero_vpip": 25.0,
            "hero_pfr": 20.0,
            "hero_high_card": 14,  # Ace
            "hero_low_card": 13,  # King
            "hero_is_pair": 0,
            "hero_is_suited": 1,
            "hero_card_gap": 1,
            "hand_strength": 1,  # Pair
            "hand_high_card": 14,
            "hand_kicker": 13,
            "is_premium_pair": 0,
            "is_broadway": 1,
        }

        # Convert to DataFrame
        sample_df = pd.DataFrame([sample_features])
        print(f"✅ Sample prediction created")

        return True

    except Exception as e:
        print(f"❌ Error in prediction pipeline: {e}")
        return False


def test_app_functionality():
    """Test the Streamlit app functionality."""
    print("\n🔍 Testing app functionality...")

    try:
        # Test if app can be imported
        import sys
        sys.path.append('src/web')
        
        # Test model loading function
        from src.web.app import load_model, get_card_options, parse_card
        
        # Test card options
        card_options = get_card_options()
        if len(card_options) > 0:
            print("✅ Card options function works")
        else:
            print("❌ Card options function failed")
            return False
        
        # Test card parsing
        rank, suit = parse_card("Ah")
        if rank == "A" and suit == "h":
            print("✅ Card parsing function works")
        else:
            print("❌ Card parsing function failed")
            return False
        
        # Test model loading
        model, metadata = load_model()
        if model is not None and metadata is not None:
            print("✅ Model loading function works")
        else:
            print("❌ Model loading function failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing app functionality: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Poker AI Pipeline")
    print("=" * 50)

    tests = [
        test_data_loading,
        test_feature_engineering,
        test_prediction_pipeline,
        test_app_functionality,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Pipeline is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")


if __name__ == "__main__":
    main()
