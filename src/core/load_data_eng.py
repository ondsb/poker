import pandas as pd
import re
from collections import defaultdict
import functools
from typing import Dict, List, Optional, Tuple, Any
import logging
from src.utils import get_card_rank, get_card_suit, DEFAULT_PLAYER_STATS

# Configure logging
logger = logging.getLogger(__name__)

# --- STEP 1: PRE-COMPUTING PLAYER STRATEGY STATS ---


def compute_player_stats(df):
    """
    Analyzes the entire dataset to compute long-term strategy stats for each player.
    """
    print("📊 Computing long-term player strategy stats (VPIP, PFR)...")

    # Using defaultdict to handle new players easily
    stats = defaultdict(
        lambda: {"hands": 0, "vpip_opp": 0, "vpip_act": 0, "pfr_opp": 0, "pfr_act": 0}
    )

    for _, row in df.iterrows():
        player_id = row["player_id"]
        actions = row["actions"]

        stats[player_id]["hands"] += 1

        # Simple VPIP Logic: Did the player voluntarily put money in?
        # This is a simplification; a full implementation is more complex.
        is_voluntary_action = any(
            "cc" in a or "cbr" in a for a in actions if a.startswith(f"p{row['player_idx']}")
        )
        if is_voluntary_action:
            stats[player_id]["vpip_act"] += 1
        stats[player_id]["vpip_opp"] += 1  # Every hand is an opportunity

        # Simple PFR Logic: Was the player's first action a raise?
        first_action = next((a for a in actions if a.startswith(f"p{row['player_idx']}")), None)
        if first_action and "cbr" in first_action:
            stats[player_id]["pfr_act"] += 1
        stats[player_id]["pfr_opp"] += 1  # Every hand is an opportunity

    # Calculate final percentages
    final_stats = {}
    for pid, data in stats.items():
        final_stats[pid] = {
            "vpip": (data["vpip_act"] / data["vpip_opp"]) * 100 if data["vpip_opp"] > 0 else 0,
            "pfr": (data["pfr_act"] / data["pfr_opp"]) * 100 if data["pfr_opp"] > 0 else 0,
        }

    print(f"✅ Computed stats for {len(final_stats)} unique players.")
    return final_stats


# --- STEP 2: CARD AND HAND EVALUATION FUNCTIONS ---

# Card utility functions moved to utils.py


def evaluate_hand_strength(hole_cards, board_cards):
    """
    Evaluates the strength of a poker hand.
    Returns a tuple: (hand_rank, high_card, kicker)
    Hand ranks: 0=High card, 1=Pair, 2=Two pair, 3=Three of a kind,
               4=Straight, 5=Flush, 6=Full house, 7=Four of a kind, 8=Straight flush
    """
    if not hole_cards or len(hole_cards) < 4:
        return (0, 0, 0)  # Invalid cards

    # Parse hole cards
    card1 = hole_cards[:2]
    card2 = hole_cards[2:4]

    # Parse board cards
    all_cards = [card1, card2]
    if board_cards and isinstance(board_cards, list):
        for board_str in board_cards:
            if isinstance(board_str, str):
                for i in range(0, len(board_str), 2):
                    if i + 1 < len(board_str):
                        all_cards.append(board_str[i : i + 2])

    # Remove invalid cards
    valid_cards = [card for card in all_cards if len(card) == 2 and card[0] != "?"]

    if len(valid_cards) < 2:
        return (0, 0, 0)

    # Get ranks and suits
    ranks = [get_card_rank(card) for card in valid_cards]
    suits = [get_card_suit(card) for card in valid_cards]

    # Count rank frequencies
    rank_counts = defaultdict(int)
    for rank in ranks:
        rank_counts[rank] += 1

    # Count suit frequencies
    suit_counts = defaultdict(int)
    for suit in suits:
        suit_counts[suit] += 1

    # Sort ranks for straight detection
    unique_ranks = sorted(set(ranks))

    # Evaluate hand type
    max_rank_count = max(rank_counts.values())
    max_suit_count = max(suit_counts.values())

    # Check for straight flush
    if max_suit_count >= 5 and len(unique_ranks) >= 5:
        # Check for straight in the same suit
        for suit in suit_counts:
            if suit_counts[suit] >= 5:
                suit_ranks = [r for r, s in zip(ranks, suits) if s == suit]
                suit_ranks = sorted(set(suit_ranks))
                if len(suit_ranks) >= 5:
                    # Check for consecutive ranks
                    for i in range(len(suit_ranks) - 4):
                        if suit_ranks[i + 4] - suit_ranks[i] == 4:
                            return (8, max(suit_ranks), 0)  # Straight flush

    # Check for four of a kind
    if max_rank_count == 4:
        four_rank = max(r for r, count in rank_counts.items() if count == 4)
        kicker = max(r for r in ranks if r != four_rank)
        return (7, four_rank, kicker)

    # Check for full house
    if max_rank_count == 3 and len([r for r, count in rank_counts.items() if count >= 2]) >= 2:
        three_rank = max(r for r, count in rank_counts.items() if count == 3)
        pair_rank = max(r for r, count in rank_counts.items() if count >= 2 and r != three_rank)
        return (6, three_rank, pair_rank)

    # Check for flush
    if max_suit_count >= 5:
        flush_suit = max(suit for suit, count in suit_counts.items() if count >= 5)
        flush_ranks = [r for r, s in zip(ranks, suits) if s == flush_suit]
        return (5, max(flush_ranks), 0)

    # Check for straight
    if len(unique_ranks) >= 5:
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i + 4] - unique_ranks[i] == 4:
                return (4, unique_ranks[i + 4], 0)

    # Check for three of a kind
    if max_rank_count == 3:
        three_rank = max(r for r, count in rank_counts.items() if count == 3)
        kickers = sorted([r for r in ranks if r != three_rank], reverse=True)
        return (3, three_rank, kickers[0] if kickers else 0)

    # Check for two pair
    pairs = [r for r, count in rank_counts.items() if count == 2]
    if len(pairs) >= 2:
        pairs.sort(reverse=True)
        kicker = max(r for r in ranks if r not in pairs)
        return (2, pairs[0], pairs[1])

    # Check for pair
    if max_rank_count == 2:
        pair_rank = max(r for r, count in rank_counts.items() if count == 2)
        kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)
        return (1, pair_rank, kickers[0] if kickers else 0)

    # High card
    sorted_ranks = sorted(ranks, reverse=True)
    return (0, sorted_ranks[0], sorted_ranks[1] if len(sorted_ranks) > 1 else 0)


def create_feature_vector(row, player_stats_lookup):
    """
    Processes a single row to create a flat feature vector including strategy.
    """
    features = {}

    # --- Parse Actions & Pot ---
    antes = row.get("antes", [0])
    blinds = row.get("blinds_or_straddles", [0])
    pot_size = sum(antes) + sum(blinds)
    hero_bet_amount = 0
    total_bets = 0

    actions = row.get("actions", [])
    for action_str in actions:
        parts = action_str.split()
        if len(parts) > 2 and parts[1] in ["cbr", "br"] and parts[2].isdigit():
            bet_amount = int(parts[2])
            pot_size += bet_amount
            total_bets += bet_amount
            if action_str.startswith(f"p{row['player_idx']}"):
                hero_bet_amount += bet_amount

    # --- Global & Positional Features ---
    features["pot_size"] = pot_size
    features["seat_count"] = row.get("seat_count", 1)
    features["position"] = row.get("player_idx", 1)  # Use player_idx instead of seat

    # --- Betting Round ---
    board_cards = row.get("board_cards", [])
    board_len = len(board_cards) if isinstance(board_cards, list) else 0
    if board_len == 0:
        features["betting_round"] = 0  # Pre-flop
    elif board_len == 3:
        features["betting_round"] = 1  # Flop
    elif board_len == 4:
        features["betting_round"] = 2  # Turn
    else:
        features["betting_round"] = 3  # River

    # --- Hero's Features (the player perspective of this row) ---
    hero_id = row.get("player_id", "unknown")
    hero_stats = player_stats_lookup.get(
        hero_id, {"vpip": 25.0, "pfr": 20.0}
    )  # Use average stats for new players

    # Stack and betting features
    starting_stack = row.get("starting_stack", 100)
    current_stack = starting_stack - hero_bet_amount
    features["hero_stack_to_pot_ratio"] = current_stack / pot_size if pot_size > 0 else 100
    features["hero_bet_to_pot_ratio"] = hero_bet_amount / pot_size if pot_size > 0 else 0
    features["total_bets_in_hand"] = total_bets
    features["hero_contribution_to_pot"] = hero_bet_amount / pot_size if pot_size > 0 else 0

    # Strategy features
    features["hero_vpip"] = hero_stats["vpip"]
    features["hero_pfr"] = hero_stats["pfr"]

    # --- Enhanced Card Features with Hole Cards ---
    hole_cards = row.get("hole_cards")
    if (
        hole_cards
        and isinstance(hole_cards, str)
        and len(hole_cards) >= 4
        and "?" not in hole_cards
    ):
        # Parse hole cards like "AsQh" or "4c4d"
        card1 = hole_cards[:2]
        card2 = hole_cards[2:4]

        # Basic hole card features
        features["hero_high_card"] = max(get_card_rank(card1), get_card_rank(card2))
        features["hero_low_card"] = min(get_card_rank(card1), get_card_rank(card2))
        features["hero_is_pair"] = 1 if card1[0] == card2[0] else 0
        features["hero_is_suited"] = 1 if card1[1] == card2[1] else 0
        features["hero_card_gap"] = abs(get_card_rank(card1) - get_card_rank(card2))

        # Hand strength evaluation
        hand_rank, high_card, kicker = evaluate_hand_strength(hole_cards, board_cards)
        features["hand_strength"] = hand_rank
        features["hand_high_card"] = high_card
        features["hand_kicker"] = kicker

        # Premium hand indicators
        features["is_premium_pair"] = (
            1 if (features["hero_is_pair"] and features["hero_high_card"] >= 10) else 0
        )
        features["is_broadway"] = (
            1 if (features["hero_high_card"] >= 10 and features["hero_low_card"] >= 10) else 0
        )
        features["is_ace_high"] = 1 if features["hero_high_card"] == 14 else 0

    else:  # Handle null/malformed/obfuscated card data
        features["hero_high_card"] = 0
        features["hero_low_card"] = 0
        features["hero_is_pair"] = 0
        features["hero_is_suited"] = 0
        features["hero_card_gap"] = 0
        features["hand_strength"] = 0
        features["hand_high_card"] = 0
        features["hand_kicker"] = 0
        features["is_premium_pair"] = 0
        features["is_broadway"] = 0
        features["is_ace_high"] = 0

    # --- Enhanced Board Card Features ---
    if board_cards and isinstance(board_cards, list):
        parsed_board_cards = []
        for board_str in board_cards:
            if isinstance(board_str, str):
                # Parse board cards like "5c4d9d" or "Ad"
                for i in range(0, len(board_str), 2):
                    if i + 1 < len(board_str):
                        parsed_board_cards.append(board_str[i : i + 2])

        if parsed_board_cards:
            board_ranks = [get_card_rank(card) for card in parsed_board_cards]
            board_suits = [get_card_suit(card) for card in parsed_board_cards]

            features["board_high_card"] = max(board_ranks)
            features["board_low_card"] = min(board_ranks)
            features["board_card_count"] = len(parsed_board_cards)

            # Board texture analysis
            rank_counts = {}
            for rank in board_ranks:
                rank_counts[rank] = rank_counts.get(rank, 0) + 1

            features["board_has_pair"] = 1 if max(rank_counts.values()) >= 2 else 0
            features["board_has_trips"] = 1 if max(rank_counts.values()) >= 3 else 0
            features["board_has_quads"] = 1 if max(rank_counts.values()) >= 4 else 0

            # Suit analysis
            suit_counts = {}
            for suit in board_suits:
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
            features["board_flush_draw"] = 1 if max(suit_counts.values()) >= 3 else 0
            features["board_flush"] = 1 if max(suit_counts.values()) >= 5 else 0

            # Straight potential
            unique_ranks = sorted(set(board_ranks))
            features["board_straight_draw"] = 0
            if len(unique_ranks) >= 3:
                for i in range(len(unique_ranks) - 2):
                    if unique_ranks[i + 2] - unique_ranks[i] <= 4:
                        features["board_straight_draw"] = 1
                        break
        else:
            features["board_high_card"] = 0
            features["board_low_card"] = 0
            features["board_card_count"] = 0
            features["board_has_pair"] = 0
            features["board_has_trips"] = 0
            features["board_has_quads"] = 0
            features["board_flush_draw"] = 0
            features["board_flush"] = 0
            features["board_straight_draw"] = 0
    else:
        features["board_high_card"] = 0
        features["board_low_card"] = 0
        features["board_card_count"] = 0
        features["board_has_pair"] = 0
        features["board_has_trips"] = 0
        features["board_has_quads"] = 0
        features["board_flush_draw"] = 0
        features["board_flush"] = 0
        features["board_straight_draw"] = 0

    return features


# --- STEP 3: RUNNING THE FULL PIPELINE ---


def ensure_winnings_column(df):
    """
    Ensure a 'winnings' column exists. If missing, compute as finishing_stack - starting_stack for each player.
    """
    if "winnings" not in df.columns:
        if (
            "finishing_stacks" in df.columns
            and "starting_stacks" in df.columns
            and "player_idx" in df.columns
        ):
            winnings = []
            for idx, row in df.iterrows():
                try:
                    i = int(row["player_idx"])
                    win = row["finishing_stacks"][i] - row["starting_stacks"][i]
                except Exception:
                    win = 0
                winnings.append(win)
            df["winnings"] = winnings
        else:
            df["winnings"] = 0
    return df


def run_data_engineering_pipeline(filepath):
    """
    Orchestrates the entire data engineering process.
    """
    print(f"▶️ Starting pipeline for file: {filepath}")

    # Load data
    try:
        df = pd.read_json(filepath, lines=True, nrows=500000)  # Increased to 500K rows
        print(f"📊 Loaded {len(df)} rows.")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

    # Ensure winnings column exists
    df = ensure_winnings_column(df)

    # Step 1: Compute player stats
    player_stats = compute_player_stats(df)

    # Step 2: Apply feature engineering to each row
    print("⚙️ Engineering features for each hand...")

    # Use functools.partial to pass the constant 'player_stats_lookup' to the apply function
    feature_creator = functools.partial(create_feature_vector, player_stats_lookup=player_stats)
    feature_list = df.apply(feature_creator, axis=1)

    # Convert list of dicts to final feature DataFrame
    X = pd.DataFrame(feature_list.tolist())

    # Define the target variable (1 for a win, 0 for a loss/break-even)
    y = (df["winnings"] > 0).astype(int)

    print("✅ Pipeline complete.")
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"📊 Target distribution: {y.value_counts().to_dict()}")

    return X, y


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Replace with the actual path to your JSONL file
    poker_data_filepath = "poker_training_data.jsonl"

    X_features, y_target = run_data_engineering_pipeline(poker_data_filepath)

    if X_features is not None:
        print("\n--- Feature Matrix (X) ---")
        print(X_features.head())

        print("\n--- Target Vector (y) ---")
        print(y_target.head())

        print(f"\nFinal shape of feature matrix: {X_features.shape}")
