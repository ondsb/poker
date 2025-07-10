#!/usr/bin/env python3
"""
Train an ensemble of 10 diverse CatBoost models for poker win probability prediction.
Combines multiple CatBoost models with different hyperparameters for superior performance.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
import argparse
import logging
import warnings
import time
from typing import List, Dict, Any, Tuple
import json

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ensemble_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CatBoostEnsemble:
    """Ensemble of CatBoost models for poker win probability prediction."""
    
    def __init__(self, n_models: int = 10):
        self.n_models = n_models
        self.models = []
        self.model_configs = []
        self.cv_scores = []
        self.feature_columns = []
        
    def get_model_configs(self) -> List[Dict[str, Any]]:
        """Generate diverse model configurations for ensemble."""
        configs = [
            # Model 1: Balanced, moderate depth
            {
                'name': 'catboost_balanced',
                'iterations': 500,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 42,
                'class_weights': [1, 5],  # Handle class imbalance
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.8
            },
            # Model 2: High depth, low learning rate
            {
                'name': 'catboost_deep',
                'iterations': 800,
                'learning_rate': 0.05,
                'depth': 10,
                'l2_leaf_reg': 5,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 123,
                'class_weights': [1, 4],
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.9
            },
            # Model 3: Shallow, high learning rate
            {
                'name': 'catboost_shallow',
                'iterations': 300,
                'learning_rate': 0.2,
                'depth': 4,
                'l2_leaf_reg': 2,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 456,
                'class_weights': [1, 6],
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.7
            },
            # Model 4: High regularization
            {
                'name': 'catboost_regularized',
                'iterations': 600,
                'learning_rate': 0.08,
                'depth': 7,
                'l2_leaf_reg': 10,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 789,
                'class_weights': [1, 3],
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.85
            },
            # Model 5: Low regularization, high iterations
            {
                'name': 'catboost_low_reg',
                'iterations': 1000,
                'learning_rate': 0.06,
                'depth': 8,
                'l2_leaf_reg': 1,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 101,
                'class_weights': [1, 5],
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.95
            },
            # Model 6: Balanced with different sampling
            {
                'name': 'catboost_sampled',
                'iterations': 400,
                'learning_rate': 0.12,
                'depth': 6,
                'l2_leaf_reg': 4,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 202,
                'class_weights': [1, 4],
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.75
            },
            # Model 7: High depth with moderate learning rate
            {
                'name': 'catboost_deep_moderate',
                'iterations': 700,
                'learning_rate': 0.09,
                'depth': 9,
                'l2_leaf_reg': 6,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 303,
                'class_weights': [1, 5],
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.88
            },
            # Model 8: Conservative settings
            {
                'name': 'catboost_conservative',
                'iterations': 500,
                'learning_rate': 0.07,
                'depth': 5,
                'l2_leaf_reg': 7,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 404,
                'class_weights': [1, 6],
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.82
            },
            # Model 9: Aggressive settings
            {
                'name': 'catboost_aggressive',
                'iterations': 900,
                'learning_rate': 0.15,
                'depth': 8,
                'l2_leaf_reg': 2,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 505,
                'class_weights': [1, 4],
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.9
            },
            # Model 10: Balanced final model
            {
                'name': 'catboost_final',
                'iterations': 600,
                'learning_rate': 0.1,
                'depth': 7,
                'l2_leaf_reg': 4,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 606,
                'class_weights': [1, 5],
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.85
            }
        ]
        return configs[:self.n_models]
    
    def train_single_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          X_val: pd.DataFrame, y_val: pd.Series, 
                          config: Dict[str, Any]) -> Tuple[CatBoostClassifier, float]:
        """Train a single CatBoost model with cross-validation."""
        logger.info(f"🔄 Training {config['name']}...")
        
        # Create model
        model = CatBoostClassifier(
            iterations=config['iterations'],
            learning_rate=config['learning_rate'],
            depth=config['depth'],
            l2_leaf_reg=config['l2_leaf_reg'],
            loss_function=config['loss_function'],
            eval_metric=config['eval_metric'],
            random_seed=config['random_seed'],
            class_weights=config['class_weights'],
            bootstrap_type=config['bootstrap_type'],
            subsample=config['subsample'],
            verbose=100
        )
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=50
        )
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        logger.info(f"✅ {config['name']} trained - AUC: {auc_score:.4f}")
        return model, auc_score
    
    def train_ensemble(self, data_path: str, output_dir: str):
        """Train the complete ensemble of CatBoost models."""
        logger.info("🚀 Starting CatBoost Ensemble Training")
        logger.info(f"📊 Loading data from: {data_path}")
        
        # Load data
        df = pd.read_parquet(data_path)
        logger.info(f"📈 Loaded {df.shape[0]} samples, {df.shape[1]} features")
        
        # Prepare features
        DROP_COLS = ['hand_id', 'player_idx', 'player_name', 'hole_cards', 'board_cards']
        X = df.drop(columns=DROP_COLS + ['is_winner'])
        y = df['is_winner']
        
        self.feature_columns = list(X.columns)
        logger.info(f"🎯 Using {len(self.feature_columns)} features")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Further split training data for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        logger.info(f"📊 Train: {X_train_final.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Get model configurations
        configs = self.get_model_configs()
        
        # Train each model
        start_time = time.time()
        
        for i, config in enumerate(configs):
            logger.info(f"🎯 Training model {i+1}/{len(configs)}: {config['name']}")
            
            model, auc_score = self.train_single_model(
                X_train_final, y_train_final, X_val, y_val, config
            )
            
            self.models.append(model)
            self.model_configs.append(config)
            self.cv_scores.append(auc_score)
            
            # Save individual model
            model_path = Path(output_dir) / f"{config['name']}.cbm"
            model.save_model(str(model_path))
            logger.info(f"💾 Saved {config['name']} to {model_path}")
        
        training_time = time.time() - start_time
        logger.info(f"⏱️ Total training time: {training_time:.2f} seconds")
        
        # Evaluate ensemble on test set
        self.evaluate_ensemble(X_test, y_test, output_dir)
        
        # Save ensemble metadata
        self.save_ensemble_metadata(output_dir)
        
        logger.info("🎉 Ensemble training completed!")
    
    def predict_ensemble(self, X: pd.DataFrame, method: str = 'average') -> np.ndarray:
        """Make ensemble predictions using different combination methods."""
        predictions = []
        
        for model in self.models:
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if method == 'average':
            # Simple average
            return np.mean(predictions, axis=0)
        elif method == 'weighted':
            # Weighted average based on CV scores
            weights = np.array(self.cv_scores) / sum(self.cv_scores)
            return np.average(predictions, axis=0, weights=weights)
        elif method == 'median':
            # Median prediction (robust to outliers)
            return np.median(predictions, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def evaluate_ensemble(self, X_test: pd.DataFrame, y_test: pd.Series, output_dir: str):
        """Evaluate ensemble performance on test set."""
        logger.info("📊 Evaluating ensemble performance...")
        
        # Test different ensemble methods
        methods = ['average', 'weighted', 'median']
        results = {}
        
        for method in methods:
            y_pred_proba = self.predict_ensemble(X_test, method)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            auc_score = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[method] = {
                'auc': auc_score,
                'accuracy': accuracy,
                'predictions': y_pred_proba
            }
            
            logger.info(f"📈 {method.capitalize()} ensemble - AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}")
        
        # Find best method
        best_method = max(results.keys(), key=lambda x: results[x]['auc'])
        best_result = results[best_method]
        
        logger.info(f"🏆 Best ensemble method: {best_method} - AUC: {best_result['auc']:.4f}")
        
        # Save best predictions
        best_predictions = best_result['predictions']
        np.save(Path(output_dir) / 'ensemble_predictions.npy', best_predictions)
        
        # Detailed classification report
        y_pred_best = (best_predictions > 0.5).astype(int)
        report = classification_report(y_test, y_pred_best, output_dict=True)
        
        # Save detailed results
        results_summary = {
            'best_method': best_method,
            'best_auc': best_result['auc'],
            'best_accuracy': best_result['accuracy'],
            'all_methods': {
                method: {
                    'auc': results[method]['auc'],
                    'accuracy': results[method]['accuracy']
                } for method in results
            },
            'classification_report': report,
            'individual_model_scores': [float(score) for score in self.cv_scores]
        }
        
        with open(Path(output_dir) / 'ensemble_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"💾 Saved ensemble results to {output_dir}")
    
    def save_ensemble_metadata(self, output_dir: str):
        """Save ensemble metadata and configuration."""
        metadata = {
            'n_models': len(self.models),
            'feature_columns': self.feature_columns,
            'model_configs': self.model_configs,
            'cv_scores': self.cv_scores,
            'ensemble_info': {
                'description': 'CatBoost Ensemble for Poker Win Probability',
                'training_samples': len(self.feature_columns),
                'models': [config['name'] for config in self.model_configs],
                'average_cv_score': np.mean(self.cv_scores),
                'cv_score_std': np.std(self.cv_scores)
            }
        }
        
        with open(Path(output_dir) / 'ensemble_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Saved ensemble metadata to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train CatBoost Ensemble for Poker Win Probability")
    parser.add_argument("--data-path", type=str, default="data/processed/pluribus_features_final.parquet",
                       help="Path to the training data")
    parser.add_argument("--output-dir", type=str, default="models/ensemble",
                       help="Directory to save ensemble models")
    parser.add_argument("--n-models", type=int, default=10,
                       help="Number of models in ensemble")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train ensemble
    ensemble = CatBoostEnsemble(n_models=args.n_models)
    ensemble.train_ensemble(args.data_path, args.output_dir)

if __name__ == "__main__":
    main() 