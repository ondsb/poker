"""
Poker ML Configuration
Centralized configuration for the poker prediction system.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "phh-dataset" / "data"
MODELS_DIR = PROJECT_ROOT

# Model files
MODEL_FILES = {"improved": "poker_xgb_model_improved.joblib", "original": "poker_xgb_model.joblib"}

# Data files
DATA_FILES = {
    "with_hole_cards": "poker_training_data_with_hole_cards.jsonl",
    "original": "poker_training_data.jsonl",
    "pluribus": "poker_training_data_pluribus.jsonl",
    "handhq": "poker_training_data_handhq.jsonl",
}

# Dataset configurations
DATASETS = {
    "pluribus": {
        "root": DATA_DIR / "pluribus",
        "output": DATA_FILES["pluribus"],
        "has_hole_cards": True,
    },
    "handhq": {
        "root": DATA_DIR / "handhq",
        "output": DATA_FILES["handhq"],
        "has_hole_cards": False,
    },
}

# Model training configuration
TRAINING_CONFIG = {
    "batch_size": 100000,
    "max_batches": 20,  # Process up to 2M rows
    "test_size": 0.2,
    "random_state": 42,
    "early_stopping_rounds": 50,
    "xgboost_params": {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    },
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "default_vpip": 25.0,
    "default_pfr": 20.0,
    "min_hands_for_stats": 10,
    "max_card_rank": 14,  # Ace
    "min_card_rank": 2,  # 2
}

# Default player statistics
DEFAULT_PLAYER_STATS = {"vpip": 25.0, "pfr": 20.0}

# Card constants
SUIT_SYMBOLS = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}
RANK_NAMES = {"A": "A", "K": "K", "Q": "Q", "J": "J", "T": "10"}
HAND_RANK_NAMES = [
    "High Card",
    "Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
]

# UI configuration
UI_CONFIG = {
    "page_title": "Poker Predictor",
    "page_icon": "♠️",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "max_players": 9,
    "default_players": 6,
    "card_columns": 3,
}

# Validation configuration
VALIDATION_CONFIG = {
    "sample_size": 500,
    "min_players": 2,
    "max_players": 9,
    "accuracy_threshold": 0.5,
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "poker_ml.log",
}

# Performance configuration
PERFORMANCE_CONFIG = {
    "max_workers": os.cpu_count(),
    "chunk_size": 50,
    "memory_limit_gb": 8,
    "enable_gpu": True,
}


def get_model_path(model_type: str = "improved") -> Path:
    """Get full path to model file."""
    return MODELS_DIR / MODEL_FILES.get(model_type, MODEL_FILES["improved"])


def get_data_path(data_type: str = "with_hole_cards") -> Path:
    """Get full path to data file."""
    return PROJECT_ROOT / DATA_FILES.get(data_type, DATA_FILES["with_hole_cards"])


def validate_paths() -> Dict[str, bool]:
    """Validate that all required paths exist."""
    validation = {
        "project_root": PROJECT_ROOT.exists(),
        "data_dir": DATA_DIR.exists(),
        "models_dir": MODELS_DIR.exists(),
    }

    # Check model files
    for model_type, filename in MODEL_FILES.items():
        validation[f"model_{model_type}"] = get_model_path(model_type).exists()

    # Check data files
    for data_type, filename in DATA_FILES.items():
        validation[f"data_{data_type}"] = get_data_path(data_type).exists()

    # Check dataset directories
    for dataset_name, config in DATASETS.items():
        validation[f"dataset_{dataset_name}"] = config["root"].exists()

    return validation


def get_config_summary() -> Dict[str, Any]:
    """Get a summary of the current configuration."""
    return {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "models_dir": str(MODELS_DIR),
        "paths_valid": validate_paths(),
        "training_config": TRAINING_CONFIG,
        "feature_config": FEATURE_CONFIG,
        "ui_config": UI_CONFIG,
        "validation_config": VALIDATION_CONFIG,
        "performance_config": PERFORMANCE_CONFIG,
    }
