#!/usr/bin/env python3
"""
Ensemble Model Manager for Poker Win Probability Prediction.
Handles loading and prediction using an ensemble of CatBoost models.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsembleModelManager:
    """Manager for the CatBoost ensemble model."""
    
    def __init__(self, ensemble_dir: str = "models/ensemble"):
        self.ensemble_dir = Path(ensemble_dir)
        self.models = []
        self.model_configs = []
        self.cv_scores = []
        self.feature_columns = []
        self.metadata = {}
        self.ensemble_method = 'weighted'  # Default ensemble method
        self.load_ensemble()
    
    def load_ensemble(self):
        """Load all models in the ensemble."""
        if not self.ensemble_dir.exists():
            raise FileNotFoundError(f"Ensemble directory not found: {self.ensemble_dir}")
        
        # Load metadata
        metadata_path = self.ensemble_dir / 'ensemble_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.feature_columns = self.metadata.get('feature_columns', [])
            self.cv_scores = self.metadata.get('cv_scores', [])
            self.model_configs = self.metadata.get('model_configs', [])
        
        # Load individual models
        logger.info(f"📁 Loading ensemble from: {self.ensemble_dir}")
        
        for config in self.model_configs:
            model_name = config['name']
            model_path = self.ensemble_dir / f"{model_name}.cbm"
            
            if model_path.exists():
                model = CatBoostClassifier()
                model.load_model(str(model_path))
                self.models.append(model)
                logger.info(f"✅ Loaded {model_name}")
            else:
                logger.warning(f"⚠️ Model file not found: {model_path}")
        
        logger.info(f"🎯 Loaded {len(self.models)} models in ensemble")
        
        # Load results if available
        results_path = self.ensemble_dir / 'ensemble_results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                self.results = json.load(f)
            logger.info(f"📊 Ensemble best method: {self.results.get('best_method', 'unknown')}")
            logger.info(f"📊 Ensemble best AUC: {self.results.get('best_auc', 0):.4f}")
    
    def predict_ensemble(self, features: Dict[str, Any], method: Optional[str] = None) -> float:
        """Make ensemble prediction for a single sample."""
        if not self.models:
            raise ValueError("No models loaded in ensemble")
        
        # Prepare features
        X = self.prepare_features_for_prediction(features)
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)[0, 1]  # Probability of winning
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Combine predictions using specified method
        method = method or self.ensemble_method
        
        if method == 'average':
            # Simple average
            final_prediction = np.mean(predictions)
        elif method == 'weighted':
            # Weighted average based on CV scores
            weights = np.array(self.cv_scores) / sum(self.cv_scores)
            final_prediction = np.average(predictions, weights=weights)
        elif method == 'median':
            # Median prediction (robust to outliers)
            final_prediction = np.median(predictions)
        elif method == 'max':
            # Maximum prediction (optimistic)
            final_prediction = np.max(predictions)
        elif method == 'min':
            # Minimum prediction (conservative)
            final_prediction = np.min(predictions)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return float(final_prediction)
    
    def predict_batch(self, features_list: List[Dict[str, Any]], method: Optional[str] = None) -> List[float]:
        """Make ensemble predictions for a batch of samples."""
        return [self.predict_ensemble(features, method) for features in features_list]
    
    def prepare_features_for_prediction(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for model prediction."""
        df = pd.DataFrame([features])
        
        # Ensure all required features are present
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            X = df[self.feature_columns]
        else:
            X = df
        
        return X
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get ensemble information and metadata."""
        return {
            "ensemble_type": "catboost_ensemble",
            "n_models": len(self.models),
            "feature_count": len(self.feature_columns),
            "ensemble_method": self.ensemble_method,
            "metadata": self.metadata,
            "cv_scores": self.cv_scores,
            "average_cv_score": np.mean(self.cv_scores) if self.cv_scores else 0,
            "ensemble_dir": str(self.ensemble_dir)
        }
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get performance metrics for each model in the ensemble."""
        if not hasattr(self, 'results'):
            return {}
        
        performance = {}
        for i, config in enumerate(self.model_configs):
            model_name = config['name']
            if i < len(self.cv_scores):
                performance[model_name] = self.cv_scores[i]
        
        return performance
    
    def set_ensemble_method(self, method: str):
        """Set the ensemble combination method."""
        valid_methods = ['average', 'weighted', 'median', 'max', 'min']
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Must be one of: {valid_methods}")
        
        self.ensemble_method = method
        logger.info(f"🔄 Ensemble method set to: {method}")
    
    def get_available_methods(self) -> List[str]:
        """Get list of available ensemble combination methods."""
        return ['average', 'weighted', 'median', 'max', 'min']

# Global ensemble manager instance
_ensemble_manager = None

def get_ensemble_manager(ensemble_dir: str = "models/ensemble") -> EnsembleModelManager:
    """Get the singleton ensemble manager instance."""
    global _ensemble_manager
    if _ensemble_manager is None:
        _ensemble_manager = EnsembleModelManager(ensemble_dir)
    return _ensemble_manager 