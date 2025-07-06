import pandas as pd
import json
from collections import Counter
from src.utils import safe_load_jsonl


def analyze_dataset_quality(filepath, sample_size=1000000):
    """
    Analyze the quality of the poker dataset to determine suitability for training.
    """
    print(f"🔍 Analyzing dataset quality: {filepath}")
    print(f"📊 Sampling {sample_size:,} rows for analysis...")

    # Load a large sample
    df = pd.read_json(filepath, lines=True, nrows=sample_size)
    print(f"✅ Loaded {len(df):,} rows")

    # Check hole card obfuscation
    print("\n=== HOLE CARD ANALYSIS ===")
    hole_cards = df["hole_cards"].fillna("null")
    obfuscated = hole_cards.str.contains(r"\?", na=False).sum()
    null_cards = hole_cards.isin(["null", None]).sum()
    valid_cards = len(df) - obfuscated - null_cards

    print(f"Valid cards: {valid_cards:,} ({(valid_cards/len(df)*100):.1f}%)")
    print(f"Obfuscated (????): {obfuscated:,} ({(obfuscated/len(df)*100):.1f}%)")
    print(f"Null/missing: {null_cards:,} ({(null_cards/len(df)*100):.1f}%)")

    # Check betting rounds
    print("\n=== BETTING ROUND ANALYSIS ===")
    board_cards = df["board_cards"]
    preflop = sum(
        1
        for cards in board_cards
        if pd.isna(cards) or (isinstance(cards, list) and len(cards) == 0)
    )
    flop = sum(1 for cards in board_cards if isinstance(cards, list) and len(cards) == 3)
    turn = sum(1 for cards in board_cards if isinstance(cards, list) and len(cards) == 4)
    river = sum(1 for cards in board_cards if isinstance(cards, list) and len(cards) == 5)

    print(f"Pre-flop: {preflop:,} ({(preflop/len(df)*100):.1f}%)")
    print(f"Flop: {flop:,} ({(flop/len(df)*100):.1f}%)")
    print(f"Turn: {turn:,} ({(turn/len(df)*100):.1f}%)")
    print(f"River: {river:,} ({(river/len(df)*100):.1f}%)")

    # 3. Check win rate and distribution
    print("\n=== WIN RATE ANALYSIS ===")
    winnings = df["winnings"]
    wins = (winnings > 0).sum()
    losses = (winnings == 0).sum()
    total_winnings = winnings.sum()

    print(f"Win rate: {wins:,} / {len(df):,} ({(wins/len(df)*100):.1f}%)")
    print(f"Loss rate: {losses:,} / {len(df):,} ({(losses/len(df)*100):.1f}%)")
    print(f"Total winnings: ${total_winnings:,.2f}")
    print(f"Average winnings per hand: ${total_winnings/len(df):.2f}")

    # 4. Check player count distribution
    print("\n=== PLAYER COUNT ANALYSIS ===")
    seat_counts = df["seat_count"].value_counts().sort_index()
    print("Player count distribution:")
    for players, count in seat_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {players} players: {count:,} ({percentage:.1f}%)")

    # 5. Check pot size distribution
    print("\n=== POT SIZE ANALYSIS ===")
    pot_sizes = df["pot_size"] if "pot_size" in df.columns else df["blinds_or_straddles"].apply(sum)
    print(f"Pot size stats:")
    print(f"  Min: ${pot_sizes.min()}")
    print(f"  Max: ${pot_sizes.max()}")
    print(f"  Mean: ${pot_sizes.mean():.2f}")
    print(f"  Median: ${pot_sizes.median():.2f}")

    # Overall assessment
    print("\n=== ASSESSMENT ===")
    if valid_cards / len(df) < 0.1:
        print("❌ 90%+ hole cards are obfuscated - DATASET NOT SUITABLE")
        print("   Cannot train meaningful poker win prediction without actual cards")
    elif valid_cards / len(df) < 0.5:
        print("⚠️ 50%+ hole cards are obfuscated - LIMITED USEFULNESS")
        print("   Model will be severely handicapped")
    else:
        print("✅ Most hole cards are valid - DATASET SUITABLE")


def main():
    print("🎯 Poker Dataset Quality Analysis")
    print("=" * 50)

    poker_data_filepath = "poker_training_data.jsonl"
    analyze_dataset_quality(poker_data_filepath, sample_size=1000000)


if __name__ == "__main__":
    main()
