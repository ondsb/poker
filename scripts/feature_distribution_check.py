import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.core import create_feature_vector, compute_player_stats
import functools
import gc


def load_data_in_batches(filepath, batch_size=100000, max_batches=5):
    print(f"📊 Loading data in batches from: {filepath}")
    all_features = []
    batch_count = 0
    total_rows = 0
    print("🔄 Computing player stats on sample data...")
    sample_df = pd.read_json(filepath, lines=True, nrows=500000)
    player_stats = compute_player_stats(sample_df)
    del sample_df
    gc.collect()
    for batch_df in pd.read_json(filepath, lines=True, chunksize=batch_size):
        if max_batches and batch_count >= max_batches:
            break
        print(f"Processing batch {batch_count + 1} ({len(batch_df)} rows)...")
        feature_creator = functools.partial(create_feature_vector, player_stats_lookup=player_stats)
        feature_list = batch_df.apply(feature_creator, axis=1)
        batch_features = pd.DataFrame(feature_list.tolist())
        all_features.append(batch_features)
        total_rows += len(batch_df)
        batch_count += 1
        del batch_df, feature_list, batch_features
        gc.collect()
        if batch_count % 5 == 0:
            print(f"Processed {batch_count} batches, {total_rows} total rows")
    print("🔗 Combining all batches...")
    X = pd.concat(all_features, ignore_index=True)
    print(f"✅ Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    return X


def check_feature_distributions(X):
    print("\n=== Feature Distribution Check ===")
    for col in X.columns:
        col_data = X[col]
        print(f"\nFeature: {col}")
        print(f"  Type: {col_data.dtype}")
        print(
            f"  Min: {col_data.min()}  Max: {col_data.max()}  Mean: {col_data.mean()}  Std: {col_data.std()}"
        )
        print(
            f"  NaN count: {col_data.isna().sum()}  Zero count: {(col_data == 0).sum()}  Unique: {col_data.nunique()}"
        )
        if col_data.isna().all() or (col_data == 0).all():
            print(f"  ⚠️  Feature is always NaN or zero! Review feature engineering.")
        elif col_data.nunique() == 1:
            print(f"  ⚠️  Feature is constant! Value: {col_data.iloc[0]}")
        # Plot distribution for numeric features
        if np.issubdtype(col_data.dtype, np.number):
            plt.figure(figsize=(6, 3))
            sns.histplot(col_data.dropna(), bins=30, kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(f"feature_dist_{col}.png")
            plt.close()
            print(f"  📊 Distribution plot saved as feature_dist_{col}.png")
        else:
            print(f"  (Not plotting non-numeric feature)")


def main():
    print("🎯 Feature Distribution Check")
    print("=" * 50)
    poker_data_filepath = "poker_training_data.jsonl"
    X = load_data_in_batches(poker_data_filepath, batch_size=100000, max_batches=5)
    check_feature_distributions(X)
    print("\n✅ Feature distribution check complete. Review flagged features and plots.")


if __name__ == "__main__":
    main()
