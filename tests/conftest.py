"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from src.utils import safe_load_model, get_data_path, safe_load_jsonl
import json


@pytest.fixture(scope="session")
def sample_data():
    """Load a small sample of data for testing."""
    data_file = get_data_path("with_hole_cards")
    if data_file.exists():
        return safe_load_jsonl(data_file, nrows=100)
    else:
        # Create mock data for testing
        return pd.DataFrame(
            {
                "player_idx": [0, 1, 2],
                "hole_cards": ["AsKh", "QdJc", "2h3s"],
                "board_cards": [["5c4d9d"], ["5c4d9d"], ["5c4d9d"]],
                "pot_size": [100, 100, 100],
                "seat_count": [6, 6, 6],
                "actions": [["p1 cbr 50"], ["p1 cbr 50"], ["p1 cbr 50"]],
                "winnings": [100, 0, 0],
            }
        )


@pytest.fixture(scope="session")
def trained_model():
    """Load the trained model if available."""
    return safe_load_model("improved")


@pytest.fixture
def temp_data_file():
    """Create a temporary data file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Write some test data
        test_data = [
            {"player_idx": 0, "hole_cards": "AsKh", "winnings": 100},
            {"player_idx": 1, "hole_cards": "QdJc", "winnings": 0},
        ]
        for item in test_data:
            f.write(json.dumps(item) + "\n")
        temp_file = f.name

    yield temp_file

    # Cleanup
    os.unlink(temp_file)


@pytest.fixture
def mock_player_stats():
    """Mock player statistics for testing."""
    return {
        "player1": {"vpip": 25.0, "pfr": 20.0},
        "player2": {"vpip": 30.0, "pfr": 15.0},
    }
