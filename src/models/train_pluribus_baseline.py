#!/usr/bin/env python3
"""
Train a baseline CatBoost model on the Pluribus player-centric dataset.
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from pathlib import Path
import joblib
import argparse

def train_pluribus_model(data_path, model_path):
    """Train a CatBoost model on the Pluribus dataset."""
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # Features to drop (IDs, names, raw cards)
    DROP_COLS = [
        'hand_id', 'player_idx', 'player_name', 'hole_cards', 'board_cards'
    ]

    X = df.drop(columns=DROP_COLS + ['is_winner'])
    y = df['is_winner']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # CatBoost baseline
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        verbose=50,
        random_seed=42
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test ROC AUC: {auc:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model
    Path("models").mkdir(exist_ok=True)
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    return model, auc, acc

def main():
    parser = argparse.ArgumentParser(description="Train Pluribus CatBoost model")
    parser.add_argument("--data-path", type=str, default="data/processed/pluribus_features_all.parquet", 
                       help="Path to the training data")
    parser.add_argument("--model-path", type=str, default="models/pluribus_catboost_v3.cbm", 
                       help="Path to save the trained model")
    
    args = parser.parse_args()
    
    train_pluribus_model(args.data_path, args.model_path)

if __name__ == "__main__":
    main() 