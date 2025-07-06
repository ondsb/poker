import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import random
from catboost import CatBoostClassifier, Pool
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Poker Win Probability Predictor", page_icon="♠️", layout="wide")

# Title
st.title("♠️ Poker Win Probability Predictor")
st.markdown(
    "Predict your chances of winning with AI-powered analysis using complete game information"
)

# Add some styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2980b9);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .winner-highlight {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained CatBoost model and metadata."""
    try:
        model_path = Path("models/catboost_poker_model.cbm")
        metadata_path = Path("models/model_metadata.pkl")

        if not model_path.exists() or not metadata_path.exists():
            st.error("Model files not found. Please train the model first.")
            return None, None

        # Load model
        model = CatBoostClassifier()
        model.load_model(str(model_path))

        # Load metadata
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def get_card_options():
    """Get all possible card options in the correct format (e.g., 'As', 'Kh')."""
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["h", "d", "c", "s"]

    cards = []
    for rank in ranks:
        for suit in suits:
            cards.append(f"{rank}{suit}")

    return ["unknown"] + cards


def parse_card(card_str):
    """Parse a card string like 'As' into rank and suit."""
    if not card_str or card_str == "unknown":
        return None, None

    if len(card_str) < 2:
        return None, None

    rank = card_str[:-1]
    suit = card_str[-1]
    return rank, suit


def calculate_hand_features(hole_card1, hole_card2):
    """Calculate hand strength features from hole cards."""
    if hole_card1 == "unknown" or hole_card2 == "unknown":
        return {
            "has_hole_cards": 0,
            "hole_card_high": 0,
            "hole_card_low": 0,
            "hole_cards_suited": 0,
            "hole_cards_paired": 0,
            "hole_card_gap": 0,
        }

    rank1, suit1 = parse_card(hole_card1)
    rank2, suit2 = parse_card(hole_card2)

    if not rank1 or not rank2:
        return {
            "has_hole_cards": 0,
            "hole_card_high": 0,
            "hole_card_low": 0,
            "hole_cards_suited": 0,
            "hole_cards_paired": 0,
            "hole_card_gap": 0,
        }

    # Convert ranks to numbers
    rank_map = {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "T": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "A": 14,
    }

    num1 = rank_map.get(rank1, 0)
    num2 = rank_map.get(rank2, 0)

    return {
        "has_hole_cards": 1,
        "hole_card_high": max(num1, num2),
        "hole_card_low": min(num1, num2),
        "hole_cards_suited": 1 if suit1 == suit2 else 0,
        "hole_cards_paired": 1 if rank1 == rank2 else 0,
        "hole_card_gap": abs(num1 - num2),
    }


def create_complete_features(seat_count, player_cards, board_cards, context_features):
    """Create complete feature set for all players in the correct order."""
    # Get the exact feature names from metadata
    model, metadata = load_model()
    if model is None:
        return []

    feature_names = metadata.get("feature_names", [])

    features_list = []

    for player_idx in range(seat_count):
        # Start with base features
        features = {}

        # Set all features in the correct order
        for feature in feature_names:
            if feature == "player_idx":
                features[feature] = player_idx
            elif feature == "hole_card1":
                features[feature] = player_cards[player_idx][0]
            elif feature == "hole_card2":
                features[feature] = player_cards[player_idx][1]
            elif feature.startswith("board_card") and feature != "board_card_count":
                # Extract card index from feature name (e.g., board_card1 -> 0, board_card2 -> 1)
                card_idx = int(feature.replace("board_card", "")) - 1
                features[feature] = (
                    board_cards[card_idx] if card_idx < len(board_cards) else "unknown"
                )
            elif feature.startswith("player_") and feature.endswith(("_hole_card1", "_hole_card2")):
                # Extract player number and card number from feature name
                # Format: player_0_hole_card1 -> player 0, card 1
                parts = feature.split("_")
                if len(parts) >= 4:
                    other_player = int(parts[1])
                    card_num = int(parts[3][-1])  # 1 or 2 from hole_card1 or hole_card2
                    if other_player < len(player_cards):
                        features[feature] = player_cards[other_player][card_num - 1]
                    else:
                        features[feature] = "unknown"
                else:
                    features[feature] = "unknown"
            elif feature in context_features:
                features[feature] = context_features[feature]
            else:
                # Default values for other features
                if feature in [
                    "has_hole_cards",
                    "hole_cards_suited",
                    "hole_cards_paired",
                    "is_heads_up",
                    "is_multiway",
                    "player_folded",
                ]:
                    features[feature] = 0
                elif feature in [
                    "hole_card_high",
                    "hole_card_low",
                    "hole_card_gap",
                    "board_card_count",
                    "player_action_count",
                    "player_aggressive_actions",
                    "player_passive_actions",
                    "player_starting_stack",
                    "player_seat",
                    "opponent_count",
                    "total_antes",
                    "total_blinds",
                    "seat_count",
                    "estimated_pot",
                    "min_bet",
                ]:
                    features[feature] = 0
                else:
                    features[feature] = 0

        # Calculate hand strength features for this player
        hand_features = calculate_hand_features(
            player_cards[player_idx][0], player_cards[player_idx][1]
        )
        features.update(hand_features)

        # Update player-specific features
        features["player_seat"] = player_idx + 1

        features_list.append(features)

    return features_list


def generate_random_hand(seat_count=6):
    """Generate a random poker hand setup."""
    card_options = get_card_options()
    # Remove "unknown" from options for random generation
    card_options = [card for card in card_options if card != "unknown"]

    # Generate random player cards (ensure no duplicates)
    used_cards = set()
    player_cards = []

    for i in range(seat_count):
        # Generate two unique cards for this player
        card1 = None
        card2 = None

        while card1 is None or card1 in used_cards:
            card1 = random.choice(card_options)

        while card2 is None or card2 in used_cards:
            card2 = random.choice(card_options)

        used_cards.add(card1)
        used_cards.add(card2)
        player_cards.append((card1, card2))

    # Generate random board cards (0-5 cards)
    board_card_count = random.randint(0, 5)
    board_cards = []

    for i in range(5):
        if i < board_card_count:
            # Generate a card not used by players
            board_card = None
            while board_card is None or board_card in used_cards:
                board_card = random.choice(card_options)
            used_cards.add(board_card)
            board_cards.append(board_card)
        else:
            board_cards.append("unknown")

    # Generate random context
    context_features = {
        "seat_count": seat_count,
        "board_card_count": board_card_count,
        "estimated_pot": random.randint(500, 10000),
        "min_bet": random.randint(50, 500),
        "player_action_count": random.randint(1, 8),
        "player_aggressive_actions": random.randint(0, 4),
        "player_passive_actions": random.randint(0, 4),
        "player_starting_stack": random.randint(5000, 50000),
        "is_heads_up": 1 if seat_count == 2 else 0,
        "is_multiway": 1 if seat_count > 2 else 0,
        "opponent_count": seat_count - 1,
        "total_antes": random.randint(0, 200),
        "total_blinds": random.randint(50, 300),
        "player_folded": 0,
    }

    return player_cards, board_cards, context_features


def create_enhanced_visualizations(probs, player_cards, board_cards, seat_count):
    """Create enhanced visualizations using Plotly."""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Win Probabilities', 'Probability Distribution', 'Hand Rankings', 'Board Analysis'),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # 1. Bar chart of win probabilities
    fig.add_trace(
        go.Bar(
            x=[f"Player {i+1}" for i in range(seat_count)],
            y=probs * 100,
            text=[f"{p:.1f}%" for p in probs * 100],
            textposition='auto',
            marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3'][:seat_count],
            name="Win Probability"
        ),
        row=1, col=1
    )
    
    # 2. Pie chart of probability distribution
    fig.add_trace(
        go.Pie(
            labels=[f"Player {i+1}" for i in range(seat_count)],
            values=probs,
            textinfo='label+percent',
            name="Probability Share"
        ),
        row=1, col=2
    )
    
    # 3. Hand rankings (sorted probabilities)
    sorted_indices = np.argsort(probs)[::-1]
    fig.add_trace(
        go.Bar(
            x=[f"#{i+1}" for i in range(seat_count)],
            y=[probs[idx] * 100 for idx in sorted_indices],
            text=[f"P{sorted_indices[i]+1}: {probs[sorted_indices[i]]:.1f}%" for i in range(seat_count)],
            textposition='auto',
            marker_color='lightblue',
            name="Hand Rankings"
        ),
        row=2, col=1
    )
    
    # 4. Board card analysis (if any board cards)
    board_cards_known = [card for card in board_cards if card != "unknown"]
    if board_cards_known:
        # Count suits and ranks
        suits = [card[-1] for card in board_cards_known]
        ranks = [card[:-1] for card in board_cards_known]
        
        suit_counts = pd.Series(suits).value_counts()
        rank_counts = pd.Series(ranks).value_counts()
        
        fig.add_trace(
            go.Scatter(
                x=list(suit_counts.index),
                y=suit_counts.values,
                mode='markers+text',
                text=suit_counts.values,
                textposition='top center',
                marker=dict(size=20, color='red'),
                name="Board Suits"
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Poker Hand Analysis Dashboard",
        title_x=0.5
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Players", row=1, col=1)
    fig.update_yaxes(title_text="Win Probability (%)", row=1, col=1)
    fig.update_xaxes(title_text="Ranking", row=2, col=1)
    fig.update_yaxes(title_text="Win Probability (%)", row=2, col=1)
    
    return fig


def main():
    model, metadata = load_model()
    if model is None:
        st.stop()

    # Sidebar with enhanced styling
    st.sidebar.markdown("""
    <div class="main-header">
        <h3>🎮 Game Setup</h3>
    </div>
    """, unsafe_allow_html=True)

    # Number of players
    seat_count = st.sidebar.slider("Number of Players", 2, 6, 6)

    # Random generation button with enhanced styling
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        if st.button("🎲 Generate Random Hand", type="secondary", use_container_width=True):
            player_cards, board_cards, context_features = generate_random_hand(seat_count)
            st.session_state["random_hand"] = {
                "player_cards": player_cards,
                "board_cards": board_cards,
                "context_features": context_features,
            }
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            if "random_hand" in st.session_state:
                del st.session_state["random_hand"]
    
    if "random_hand" in st.session_state:
        st.sidebar.success("✅ Random hand generated!")

    # Card options
    card_options = get_card_options()

    st.sidebar.markdown("### 🎴 All Players' Hole Cards")
    player_cards = []
    for i in range(seat_count):
        st.sidebar.markdown(f"**Player {i+1}**")
        col1, col2 = st.sidebar.columns(2)

        # Check if we have a random hand in session state
        if "random_hand" in st.session_state and i < len(
            st.session_state["random_hand"]["player_cards"]
        ):
            random_card1 = st.session_state["random_hand"]["player_cards"][i][0]
            random_card2 = st.session_state["random_hand"]["player_cards"][i][1]
            idx1 = card_options.index(random_card1) if random_card1 in card_options else 0
            idx2 = card_options.index(random_card2) if random_card2 in card_options else 0
        else:
            idx1, idx2 = 0, 0

        with col1:
            c1 = st.selectbox(f"Card 1", card_options, index=idx1, key=f"p{i+1}_c1")
        with col2:
            c2 = st.selectbox(f"Card 2", card_options, index=idx2, key=f"p{i+1}_c2")
        player_cards.append((c1, c2))

    st.sidebar.markdown("### 🃏 Board Cards")
    board_cards = []
    for i in range(5):
        # Check if we have random board cards
        if "random_hand" in st.session_state and i < len(
            st.session_state["random_hand"]["board_cards"]
        ):
            random_board_card = st.session_state["random_hand"]["board_cards"][i]
            idx = card_options.index(random_board_card) if random_board_card in card_options else 0
        else:
            idx = 0

        board_cards.append(
            st.sidebar.selectbox(f"Board Card {i+1}", card_options, index=idx, key=f"board_{i+1}")
        )

    st.sidebar.markdown("### 💰 Game Context")

    # Check if we have random context features
    if "random_hand" in st.session_state:
        random_context = st.session_state["random_hand"]["context_features"]
        estimated_pot = st.sidebar.number_input(
            "Estimated Pot Size", 100, 50000, random_context["estimated_pot"]
        )
        min_bet = st.sidebar.number_input("Minimum Bet", 10, 2000, random_context["min_bet"])
        player_action_count = st.sidebar.slider(
            "Player Actions", 0, 15, random_context["player_action_count"]
        )
        player_aggressive_actions = st.sidebar.slider(
            "Aggressive Actions", 0, 10, random_context["player_aggressive_actions"]
        )
        player_passive_actions = st.sidebar.slider(
            "Passive Actions", 0, 10, random_context["player_passive_actions"]
        )
        player_starting_stack = st.sidebar.number_input(
            "Starting Stack", 1000, 200000, random_context["player_starting_stack"]
        )
    else:
        estimated_pot = st.sidebar.number_input("Estimated Pot Size", 100, 50000, 1000)
        min_bet = st.sidebar.number_input("Minimum Bet", 10, 2000, 100)
        player_action_count = st.sidebar.slider("Player Actions", 0, 15, 3)
        player_aggressive_actions = st.sidebar.slider("Aggressive Actions", 0, 10, 1)
        player_passive_actions = st.sidebar.slider("Passive Actions", 0, 10, 1)
        player_starting_stack = st.sidebar.number_input("Starting Stack", 1000, 200000, 10000)

    # Calculate derived features
    board_card_count = sum(1 for card in board_cards if card != "unknown")

    # Context features
    context_features = {
        "seat_count": seat_count,
        "board_card_count": board_card_count,
        "estimated_pot": estimated_pot,
        "min_bet": min_bet,
        "player_action_count": player_action_count,
        "player_aggressive_actions": player_aggressive_actions,
        "player_passive_actions": player_passive_actions,
        "player_starting_stack": player_starting_stack,
        "is_heads_up": 1 if seat_count == 2 else 0,
        "is_multiway": 1 if seat_count > 2 else 0,
        "opponent_count": seat_count - 1,
        "total_antes": 0,
        "total_blinds": 150,  # Default blinds
        "player_folded": 0,  # Assuming no one has folded yet
    }

    # Main content area
    st.markdown("""
    <div class="main-header">
        <h2>🎯 Win Probability Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    # Display current scenario with enhanced styling
    st.markdown("### 📋 Current Scenario")

    # Show all players' cards in a grid
    cols = st.columns(seat_count)
    for i in range(seat_count):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Player {i+1}</h4>
                <p>🃏 {player_cards[i][0]} {player_cards[i][1]}</p>
            </div>
            """, unsafe_allow_html=True)

    # Show board with enhanced styling
    st.markdown("### 🃏 Board Cards")
    board_display = " ".join([f"🃏 {card}" for card in board_cards if card != "unknown"])
    if board_display:
        st.markdown(f"""
        <div class="metric-card">
            <p>{board_display}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No board cards yet (preflop)")

    # Game context summary
    st.markdown(f"""
    <div class="metric-card">
        <p><strong>Pot:</strong> ${estimated_pot:,} | <strong>Players:</strong> {seat_count} | <strong>Board Cards:</strong> {board_card_count}</p>
    </div>
    """, unsafe_allow_html=True)

    # Prediction button
    if st.button("🚀 Predict Win Probabilities", type="primary", use_container_width=True):
        with st.spinner("Calculating win probabilities..."):
            try:
                # Create features for all players
                features_list = create_complete_features(
                    seat_count, player_cards, board_cards, context_features
                )

                if not features_list:
                    st.error("Failed to create features")
                    return

                # Create DataFrame in correct order
                feature_names = metadata.get("feature_names", [])
                df = pd.DataFrame(features_list)

                # Ensure DataFrame has all required columns in correct order
                for feature in feature_names:
                    if feature not in df.columns:
                        df[feature] = 0

                # Reorder columns to match expected order
                df = df[feature_names]

                # Ensure categorical features are strings
                categorical_features = metadata.get("categorical_features", [])
                for col in categorical_features:
                    if col in df.columns:
                        df[col] = df[col].astype(str)

                # Get categorical feature indices for CatBoost
                cat_feature_indices = [
                    i for i, col in enumerate(feature_names) if col in categorical_features
                ]

                # Make predictions
                pool = Pool(df, cat_features=cat_feature_indices)
                raw_probs = model.predict_proba(pool)[:, 1]

                # Display results with enhanced styling
                st.markdown("### 📊 Win Probability Results")

                # Add explanation about probabilities
                st.info(
                    """
                **ℹ️ About these probabilities:**
                - Each probability shows the chance that player wins the **entire pot**
                - These are independent probabilities (do not sum to 100%)
                - Higher probability = better chance of winning
                - Only one player wins each hand (no splitting)
                """
                )

                # Find winner (highest probability)
                winner_idx = np.argmax(raw_probs)
                
                # Display winner highlight
                st.markdown(f"""
                <div class="winner-highlight">
                    🏆 <strong>Best Hand:</strong> Player {winner_idx + 1} with {raw_probs[winner_idx]:.1%} win probability
                </div>
                """, unsafe_allow_html=True)

                # Create results DataFrame with enhanced styling
                results_data = []
                for i in range(seat_count):
                    is_winner = i == winner_idx
                    results_data.append(
                        {
                            "Player": f"Player {i+1}",
                            "Hole Cards": f"{player_cards[i][0]} {player_cards[i][1]}",
                            "Win Probability": f"{raw_probs[i]:.1%}",
                            "Rank": np.where(np.argsort(raw_probs)[::-1] == i)[0][0] + 1,
                            "Is Winner": is_winner
                        }
                    )

                results_df = pd.DataFrame(results_data)
                
                # Display results in a nice format
                for _, row in results_df.iterrows():
                    if row["Is Winner"]:
                        st.markdown(f"""
                        <div class="winner-highlight">
                            #{row['Rank']} - {row['Player']}: {row['Win Probability']} ({row['Hole Cards']})
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-card">
                            #{row['Rank']} - {row['Player']}: {row['Win Probability']} ({row['Hole Cards']})
                        </div>
                        """, unsafe_allow_html=True)

                # Enhanced visualizations
                st.markdown("### 📈 Advanced Analytics")
                
                # Create enhanced visualizations
                fig = create_enhanced_visualizations(raw_probs, player_cards, board_cards, seat_count)
                st.plotly_chart(fig, use_container_width=True)

                # Additional analysis
                st.markdown("### 🔍 Detailed Analysis")

                # Show probability sum for transparency
                prob_sum = raw_probs.sum()
                st.markdown(f"""
                <div class="metric-card">
                    <p><strong>Total probability sum:</strong> {prob_sum:.1%}</p>
                    <p><strong>Probability range:</strong> {raw_probs.min():.1%} - {raw_probs.max():.1%}</p>
                    <p><strong>Standard deviation:</strong> {raw_probs.std():.1%}</p>
                </div>
                """, unsafe_allow_html=True)

                if prob_sum > 1.0:
                    st.info("ℹ️ Sum > 100% is normal - multiple players can have high win chances")
                elif prob_sum < 0.5:
                    st.warning("⚠️ Low sum suggests model may be conservative")

                # Show hand rankings
                st.markdown("### 🏆 Hand Rankings")
                sorted_indices = np.argsort(raw_probs)[::-1]
                for i, idx in enumerate(sorted_indices):
                    if i == 0:
                        st.markdown(f"""
                        <div class="winner-highlight">
                            🥇 {i+1}. Player {idx + 1} ({player_cards[idx][0]} {player_cards[idx][1]}): {raw_probs[idx]:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                    elif i == 1:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: #ffc107;">
                            🥈 {i+1}. Player {idx + 1} ({player_cards[idx][0]} {player_cards[idx][1]}): {raw_probs[idx]:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                    elif i == 2:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: #fd7e14;">
                            🥉 {i+1}. Player {idx + 1} ({player_cards[idx][0]} {player_cards[idx][1]}): {raw_probs[idx]:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-card">
                            {i+1}. Player {idx + 1} ({player_cards[idx][0]} {player_cards[idx][1]}): {raw_probs[idx]:.1%}
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
