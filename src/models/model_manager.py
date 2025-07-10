#!/usr/bin/env python3
"""
Simplified Model Manager for Poker Win Probability.
Handles only the CatBoost Ensemble model.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PokerModelManager:
    """Manager for the CatBoost Ensemble poker model."""
    def __init__(self):
        self.model_type = "ensemble"
        self.ensemble_manager = None
        self.feature_columns = []
        self.metadata = {}
        self.ensemble_method = 'median'  # Default method
        self.load_ensemble()

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get ensemble model information."""
        try:
            from src.models.ensemble_manager import get_ensemble_manager
            ensemble_manager = get_ensemble_manager()
            ensemble_info = ensemble_manager.get_ensemble_info()
            
            return [{
                "name": "CatBoost Ensemble",
                "description": "Ensemble of 10 diverse CatBoost models for superior performance",
                "training_samples": 60307,
                "test_auc": 0.8958,
                "test_accuracy": 0.7830,
                "model_version": "ensemble",
                "model_type": "ensemble",
                "file_path": "models/ensemble",
                "ensemble_method": self.ensemble_method,
                "n_models": ensemble_info['n_models'],
                "average_cv_score": ensemble_info['average_cv_score']
            }]
        except Exception as e:
            logger.warning(f"Could not load ensemble info: {e}")
            return []

    def load_ensemble(self):
        """Load the ensemble model."""
        try:
            from src.models.ensemble_manager import get_ensemble_manager
            self.ensemble_manager = get_ensemble_manager()
            
            # Set feature columns for ensemble
            self.feature_columns = [
                'high_card_value', 'is_paired', 'is_suited', 'is_connected', 'broadway_count', 'pocket_pair_strength',
                'position_from_button', 'position_type', 'is_button', 'is_small_blind', 'is_big_blind',
                'stack_size', 'stack_ratio', 'stack_percentile', 'is_short_stack', 'is_deep_stack',
                'board_street', 'board_high_card', 'board_paired', 'board_suited',
            ]
            # Add all opponent features
            for opp in range(2, 7):
                for feat in ['high_card_value', 'is_paired', 'is_suited', 'is_connected', 'broadway_count', 'pocket_pair_strength']:
                    self.feature_columns.append(f'opp{opp}_{feat}')
            # Add opp1 features at the end (for consistency with the data)
            for feat in ['high_card_value', 'is_paired', 'is_suited', 'is_connected', 'broadway_count', 'pocket_pair_strength']:
                self.feature_columns.append(f'opp1_{feat}')
            
            # Update metadata
            self.metadata = {
                'model_type': 'ensemble',
                'feature_count': len(self.feature_columns),
                'training_samples': 60307,
                'test_auc': 0.8958,
                'test_accuracy': 0.7830,
                'model_version': 'ensemble',
                'name': 'CatBoost Ensemble',
                'description': 'Ensemble of 10 diverse CatBoost models for superior performance',
                'ensemble_method': self.ensemble_method,
                'n_models': 10
            }
            
            logger.info(f"✅ Loaded CatBoost Ensemble with {len(self.feature_columns)} features")
            
        except Exception as e:
            logger.error(f"❌ Error loading ensemble: {e}")
            raise

    def set_ensemble_method(self, method: str):
        """Set the ensemble combination method."""
        if self.ensemble_manager:
            self.ensemble_manager.set_ensemble_method(method)
            self.ensemble_method = method
            logger.info(f"🔄 Ensemble method set to: {method}")

    def get_available_methods(self) -> List[str]:
        """Get list of available ensemble combination methods."""
        return ['median', 'weighted', 'average', 'max', 'min']

    @property
    def feature_count(self) -> int:
        return len(self.feature_columns)

    def create_features(self, players: List[Dict], target_player_idx: int) -> Dict[str, Any]:
        """Create features for a target player with all opponent information."""
        player = players[target_player_idx]
        features = {}
        
        # Player-centric features
        features['high_card_value'] = player.get('high_card_value', 0)
        features['is_paired'] = player.get('is_paired', 0)
        features['is_suited'] = player.get('is_suited', 0)
        features['is_connected'] = player.get('is_connected', 0)
        features['broadway_count'] = player.get('broadway_count', 0)
        features['pocket_pair_strength'] = player.get('pocket_pair_strength', 0)
        features['position_from_button'] = player.get('position_from_button', 0)
        features['position_type'] = player.get('position_type', 0)
        features['is_button'] = player.get('is_button', 0)
        features['is_small_blind'] = player.get('is_small_blind', 0)
        features['is_big_blind'] = player.get('is_big_blind', 0)
        features['stack_size'] = player.get('stack_size', 10000)
        features['stack_ratio'] = player.get('stack_ratio', 1.0)
        features['stack_percentile'] = player.get('stack_percentile', 0.5)
        features['is_short_stack'] = player.get('is_short_stack', 0)
        features['is_deep_stack'] = player.get('is_deep_stack', 0)
        features['board_street'] = player.get('board_street', 0)
        features['board_high_card'] = player.get('board_high_card', 0)
        features['board_paired'] = player.get('board_paired', 0)
        features['board_suited'] = player.get('board_suited', 0)
        
        # Opponent features (opp2-opp6, then opp1)
        for opp in range(2, 7):
            for feat in ['high_card_value', 'is_paired', 'is_suited', 'is_connected', 'broadway_count', 'pocket_pair_strength']:
                features[f'opp{opp}_{feat}'] = player.get(f'opp{opp}_{feat}', 0)
        for feat in ['high_card_value', 'is_paired', 'is_suited', 'is_connected', 'broadway_count', 'pocket_pair_strength']:
            features[f'opp1_{feat}'] = player.get(f'opp1_{feat}', 0)
        
        return features

    def predict_win_probability(self, features: Dict[str, Any]) -> float:
        """Predict win probability using the ensemble."""
        try:
            if self.ensemble_manager:
                return self.ensemble_manager.predict_ensemble(features, self.ensemble_method)
            else:
                logger.error("❌ Ensemble manager not loaded")
                return 0.0
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return 0.0

    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[float]:
        """Predict win probabilities for a batch of players."""
        return [self.predict_win_probability(features) for features in features_list]

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata."""
        return {
            "model_type": self.model_type,
            "feature_count": len(self.feature_columns),
            "metadata": self.metadata,
            "ensemble_method": self.ensemble_method
        }

# Global model manager instance
_model_manager = None

def get_model_manager() -> PokerModelManager:
    """Get the singleton model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = PokerModelManager()
    return _model_manager 