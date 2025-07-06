"""
Poker ML Utilities Module
Consolidated utility functions for the poker prediction system.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .config import (
    DEFAULT_PLAYER_STATS,
    SUIT_SYMBOLS,
    RANK_NAMES,
    HAND_RANK_NAMES,
    get_model_path,
    get_data_path,
    MODEL_FILES,
    DATA_FILES,
)


def get_card_rank(card: str) -> int:
    """Convert a card string to numerical rank."""
    if not card or card == "?" or len(card) < 1:
        return 0

    val = card[0].upper()
    rank_map = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}

    if val in rank_map:
        return rank_map[val]

    try:
        return int(val)
    except ValueError:
        return 0


def get_card_suit(card: str) -> Optional[str]:
    """Extract suit from card string."""
    if not card or len(card) < 2:
        return None
    return card[1].lower()


def visualize_cards(cards: Any, player_name: str = "Player") -> str:
    """Create visual representation of cards with proper symbols and colors."""
    if not cards or cards == "Unknown":
        return f"🃏 {player_name}: Unknown cards"

    if isinstance(cards, str):
        # Handle single card string like "AsKh"
        card_list = []
        for i in range(0, len(cards), 2):
            if i + 1 < len(cards):
                card_list.append(cards[i : i + 2])
        cards = card_list

    card_symbols = []
    for card in cards:
        if isinstance(card, str) and len(card) >= 2:
            rank = card[:-1]
            suit = card[-1]

            suit_symbol = SUIT_SYMBOLS.get(suit.lower(), "🃏")
            rank_symbol = RANK_NAMES.get(rank.upper(), rank)

            # Color coding for suits
            if suit.lower() in ["h", "d"]:
                card_symbols.append(f"**{rank_symbol}{suit_symbol}**")  # Red suits bold
            else:
                card_symbols.append(f"{rank_symbol}{suit_symbol}")  # Black suits normal
        else:
            card_symbols.append("🃏")

    cards_str = " ".join(card_symbols)
    return f"🃏 {player_name}: {cards_str}"


def get_hand_strength_name(strength: int) -> str:
    """Convert hand strength number to name."""
    return (
        HAND_RANK_NAMES[strength]
        if 0 <= strength < len(HAND_RANK_NAMES)
        else f"Unknown({strength})"
    )


def safe_load_jsonl(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Safely load JSONL file with error handling."""
    try:
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return pd.DataFrame()

        df = pd.read_json(filepath, lines=True, nrows=nrows)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df

    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return pd.DataFrame()


def safe_load_model(model_type: str = "improved"):
    """Safely load trained model with fallback."""
    import joblib

    model_path = get_model_path(model_type)
    if not model_path:
        logger.error(f"Unknown model type: {model_type}")
        return None

    try:
        model = joblib.load(model_path)
        logger.info(f"✅ {model_type.title()} model loaded successfully")
        return model
    except FileNotFoundError:
        logger.warning(f"❌ {model_path} not found, trying fallback...")

        # Try other model files
        for other_type in MODEL_FILES:
            if other_type != model_type:
                other_path = get_model_path(other_type)
                if other_path.exists():
                    try:
                        model = joblib.load(other_path)
                        logger.info(f"✅ Fallback to {other_type} model successful")
                        return model
                    except Exception:
                        continue

        logger.error("❌ No model files found!")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return None


def ensure_winnings_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure winnings column exists in DataFrame."""
    if "winnings" not in df.columns:
        if "finishing_stacks" in df.columns and "starting_stacks" in df.columns:
            winnings = []
            for _, row in df.iterrows():
                try:
                    i = int(row.get("player_idx", 0))
                    win = row["finishing_stacks"][i] - row["starting_stacks"][i]
                except (IndexError, TypeError, KeyError):
                    win = 0
                winnings.append(win)
            df["winnings"] = winnings
        else:
            df["winnings"] = 0
            logger.warning("No stack information found, setting winnings to 0")

    return df


def validate_hand_data(hand: Dict[str, Any]) -> bool:
    """Validate hand data structure."""
    required_fields = ["players", "winnings"]
    optional_fields = ["hole_cards", "board_cards", "pot_size", "actions"]

    # Check required fields
    for field in required_fields:
        if field not in hand or not hand[field]:
            logger.warning(f"Missing required field: {field}")
            return False

    # Check data consistency
    num_players = len(hand["players"])
    if len(hand["winnings"]) != num_players:
        logger.warning(
            f"Winnings count ({len(hand['winnings'])}) doesn't match player count ({num_players})"
        )
        return False

    # Check for exactly one winner
    winners = sum(1 for w in hand["winnings"] if w > 0)
    if winners != 1:
        logger.warning(f"Expected 1 winner, found {winners}")
        return False

    return True


def normalize_probabilities(probabilities: List[float]) -> List[float]:
    """Normalize probabilities to sum to 1.0."""
    total = sum(probabilities)
    if total > 0:
        return [p / total for p in probabilities]
    else:
        # Equal distribution if all probabilities are 0
        return [1.0 / len(probabilities)] * len(probabilities)


def format_currency(amount: float) -> str:
    """Format amount as currency."""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value:.1%}"


def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB."""
    try:
        return os.path.getsize(filepath) / (1024 * 1024)
    except OSError:
        return 0.0


def count_jsonl_lines(filepath: str) -> int:
    """Count lines in JSONL file."""
    try:
        with open(filepath, "r") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def create_progress_callback(total: int, description: str = "Processing"):
    """Create a progress callback function."""
    from tqdm import tqdm

    pbar = tqdm(total=total, desc=description)

    def callback(completed: int):
        pbar.update(completed - pbar.n)
        if completed >= total:
            pbar.close()

    return callback


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary."""
    required_keys = ["model_type", "data_file"]

    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return False

    # Validate file paths
    if not os.path.exists(config["data_file"]):
        logger.error(f"Data file not found: {config['data_file']}")
        return False

    return True


def get_project_info() -> Dict[str, Any]:
    """Get project information and status."""
    info = {"model_files": {}, "data_files": {}, "project_size": 0, "status": "unknown"}

    # Check model files
    for model_type, filename in MODEL_FILES.items():
        if os.path.exists(filename):
            info["model_files"][model_type] = {
                "exists": True,
                "size_mb": get_file_size_mb(filename),
            }
        else:
            info["model_files"][model_type] = {"exists": False}

    # Check data files
    for data_type, filename in DATA_FILES.items():
        if os.path.exists(filename):
            info["data_files"][data_type] = {
                "exists": True,
                "size_mb": get_file_size_mb(filename),
                "lines": count_jsonl_lines(filename),
            }
        else:
            info["data_files"][data_type] = {"exists": False}

    # Determine project status
    if info["model_files"].get("improved", {}).get("exists"):
        info["status"] = "ready"
    elif info["model_files"].get("original", {}).get("exists"):
        info["status"] = "legacy"
    else:
        info["status"] = "needs_training"

    return info
