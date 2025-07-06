#!/usr/bin/env python3
"""
Test script to verify all improvements work correctly.
"""

import sys
import os
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    print("🔍 Testing imports...")

    try:
        import config

        print("✅ config.py imported successfully")
    except Exception as e:
        print(f"❌ config.py import failed: {e}")
        return False

    try:
        import utils

        print("✅ utils.py imported successfully")
    except Exception as e:
        print(f"❌ utils.py import failed: {e}")
        return False

    try:
        import load_data_eng

        print("✅ load_data_eng.py imported successfully")
    except Exception as e:
        print(f"❌ load_data_eng.py import failed: {e}")
        return False

    try:
        import train_model_improved

        print("✅ train_model_improved.py imported successfully")
    except Exception as e:
        print(f"❌ train_model_improved.py import failed: {e}")
        return False

    return True


def test_config():
    """Test configuration functionality."""
    print("\n🔍 Testing configuration...")

    try:
        from src.utils import get_config_summary, validate_paths

        # Test config summary
        summary = get_config_summary()
        print(f"✅ Config summary generated: {len(summary)} items")

        # Test path validation
        paths = validate_paths()
        print(f"✅ Path validation completed: {len(paths)} paths checked")

        # Show some key paths
        print(f"   Project root: {summary['project_root']}")
        print(f"   Data directory: {summary['data_dir']}")

        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_utils():
    """Test utility functions."""
    print("\n🔍 Testing utilities...")

    try:
        from src.utils import (
            get_card_rank,
            get_card_suit,
            visualize_cards,
            normalize_probabilities,
            format_currency,
            format_percentage,
        )

        # Test card functions
        assert get_card_rank("As") == 14
        assert get_card_rank("Kh") == 13
        assert get_card_rank("2c") == 2
        print("✅ Card rank functions work")

        assert get_card_suit("As") == "s"
        assert get_card_suit("Kh") == "h"
        print("✅ Card suit functions work")

        # Test visualization
        card_display = visualize_cards(["As", "Kh"], "Test Player")
        assert "A♠" in card_display
        assert "K♥" in card_display
        print("✅ Card visualization works")

        # Test probability normalization
        probs = [0.3, 0.2, 0.1]
        normalized = normalize_probabilities(probs)
        assert abs(sum(normalized) - 1.0) < 0.001
        print("✅ Probability normalization works")

        # Test formatting
        assert format_currency(1234.56) == "$1,234.56"
        assert format_percentage(0.123) == "12.3%"
        print("✅ Formatting functions work")

        return True
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        return False


def test_model_loading():
    """Test model loading functionality."""
    print("\n🔍 Testing model loading...")

    try:
        from src.utils import safe_load_model, get_project_info

        # Test project info
        info = get_project_info()
        print(f"✅ Project info generated: {info['status']} status")

        # Test model loading (may fail if no model exists)
        model = safe_load_model("improved")
        if model is not None:
            print("✅ Model loaded successfully")
        else:
            print("⚠️ No model found (this is OK if not trained yet)")

        return True
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False


def test_data_loading():
    """Test data loading functionality."""
    print("\n🔍 Testing data loading...")

    try:
        from src.utils import safe_load_jsonl, ensure_winnings_column

        # Test with a small sample
        data_file = "poker_training_data_with_hole_cards.jsonl"

        if os.path.exists(data_file):
            df = safe_load_jsonl(data_file, nrows=10)
            if not df.empty:
                print(f"✅ Data loaded successfully: {len(df)} rows")

                # Test winnings column
                df = ensure_winnings_column(df)
                assert "winnings" in df.columns
                print("✅ Winnings column ensured")
            else:
                print("⚠️ Data file exists but is empty")
        else:
            print("⚠️ Data file not found (this is OK if not prepared yet)")

        return True
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False


def test_feature_engineering():
    """Test feature engineering functionality."""
    print("\n🔍 Testing feature engineering...")

    try:
        from src.core import create_feature_vector, evaluate_hand_strength
        from collections import defaultdict

        # Test hand strength evaluation
        hand_rank, high_card, kicker = evaluate_hand_strength("AsKh", ["5c4d9d"])
        assert isinstance(hand_rank, int)
        assert isinstance(high_card, int)
        print("✅ Hand strength evaluation works")

        # Test feature vector creation (with mock data)
        mock_row = {
            "player_idx": 0,
            "hole_cards": "AsKh",
            "board_cards": ["5c4d9d"],
            "pot_size": 100,
            "seat_count": 6,
            "actions": ["p1 cbr 50"],
            "winnings": 100,
        }

        player_stats = defaultdict(lambda: {"vpip": 25.0, "pfr": 20.0})
        features = create_feature_vector(mock_row, player_stats)

        assert isinstance(features, dict)
        assert len(features) > 20  # Should have many features
        print("✅ Feature vector creation works")

        return True
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🎯 Testing Poker ML Improvements")
    print("=" * 50)

    tests = [
        test_imports,
        test_config,
        test_utils,
        test_model_loading,
        test_data_loading,
        test_feature_engineering,
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
        print("🎉 All tests passed! Improvements are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
