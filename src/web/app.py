import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import random
from catboost import CatBoostClassifier
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Poker Model Minimal Demo", 
    page_icon="♠️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .winner-highlight {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .card-display {
        background: white;
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 0.5rem;
        text-align: center;
        font-weight: bold;
        margin: 0.25rem;
        display: inline-block;
        min-width: 60px;
    }
    .red-card { color: #dc3545; }
    .black-card { color: #000; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>♠️ Poker Model Minimal Demo</h1>
    <p>Minimal demo: generate a full random model input, view all features, and predict win probability.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained CatBoost model."""
    try:
        model_path = Path("models/conservative/poker_win_probability_model.cbm")

        if not model_path.exists():
            st.error("Model file not found. Please train the model first using: `make train-model`")
            return None

        # Load model
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_card_options():
    """Get all possible card options."""
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
            "hole_card_rank_sum": 0,
            "is_broadway": 0,
            "is_ace": 0,
            "is_king": 0,
            "is_queen": 0,
            "is_jack": 0,
            "is_ten": 0,
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
            "hole_card_rank_sum": 0,
            "is_broadway": 0,
            "is_ace": 0,
            "is_king": 0,
            "is_queen": 0,
            "is_jack": 0,
            "is_ten": 0,
        }
    
    # Convert ranks to numbers
    rank_map = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14
    }
    
    num1 = rank_map.get(rank1, 0)
    num2 = rank_map.get(rank2, 0)
    
    high_card = max(num1, num2)
    low_card = min(num1, num2)
    suited = 1 if suit1 == suit2 else 0
    paired = 1 if rank1 == rank2 else 0
    gap = abs(num1 - num2)
    rank_sum = num1 + num2
    broadway = 1 if high_card >= 10 else 0
    
    return {
        "has_hole_cards": 1,
        "hole_card_high": high_card,
        "hole_card_low": low_card,
        "hole_cards_suited": suited,
        "hole_cards_paired": paired,
        "hole_card_gap": gap,
        "hole_card_rank_sum": rank_sum,
        "is_broadway": broadway,
        "is_ace": 1 if high_card == 14 else 0,
        "is_king": 1 if high_card == 13 else 0,
        "is_queen": 1 if high_card == 12 else 0,
        "is_jack": 1 if high_card == 11 else 0,
        "is_ten": 1 if high_card == 10 else 0,
    }

def calculate_board_features(board_cards):
    """Calculate board features."""
    if not board_cards or all(card == "unknown" for card in board_cards):
        return {
            "board_card_count": 0,
            "board_high_card": 0,
            "board_low_card": 0,
            "board_rank_sum": 0,
            "board_broadway_count": 0,
            "board_ace_count": 0,
            "board_king_count": 0,
            "board_queen_count": 0,
            "board_jack_count": 0,
            "board_ten_count": 0,
        }
    
    rank_map = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14
    }
    
    ranks = []
    for card in board_cards:
        if card != "unknown":
            rank, _ = parse_card(card)
            if rank:
                ranks.append(rank_map.get(rank, 0))
    
    if not ranks:
        return {
            "board_card_count": 0,
            "board_high_card": 0,
            "board_low_card": 0,
            "board_rank_sum": 0,
            "board_broadway_count": 0,
            "board_ace_count": 0,
            "board_king_count": 0,
            "board_queen_count": 0,
            "board_jack_count": 0,
            "board_ten_count": 0,
        }
    
    high_card = max(ranks)
    low_card = min(ranks)
    rank_sum = sum(ranks)
    broadway_count = sum(1 for r in ranks if r >= 10)
    
    return {
        "board_card_count": len(ranks),
        "board_high_card": high_card,
        "board_low_card": low_card,
        "board_rank_sum": rank_sum,
        "board_broadway_count": broadway_count,
        "board_ace_count": ranks.count(14),
        "board_king_count": ranks.count(13),
        "board_queen_count": ranks.count(12),
        "board_jack_count": ranks.count(11),
        "board_ten_count": ranks.count(10),
    }

def create_features_for_player(player_idx, seat_count, hole_cards, board_cards, context):
    """Create features for a specific player."""
    features = {}
    
    # Player identification
    features["player_idx"] = player_idx
    features["player_id"] = f"player_{player_idx}"
    
    # Game context
    features["seat_count"] = seat_count
    features["pot_size"] = context.get("pot_size", 1000)
    features["min_bet"] = context.get("min_bet", 10)
    features["total_antes"] = context.get("total_antes", 0)
    features["total_blinds"] = context.get("total_blinds", 15)
    features["is_heads_up"] = 1 if seat_count == 2 else 0
    features["is_multiway"] = 1 if seat_count > 2 else 0
    features["opponent_count"] = seat_count - 1
    
    # Position features
    features["position"] = player_idx
    features["is_button"] = 1 if player_idx == context.get("button", 0) else 0
    features["is_small_blind"] = 1 if player_idx == context.get("small_blind_pos", 0) else 0
    features["is_big_blind"] = 1 if player_idx == context.get("big_blind_pos", 1) else 0
    features["is_early_position"] = 1 if player_idx < seat_count // 3 else 0
    features["is_middle_position"] = 1 if seat_count // 3 <= player_idx < 2 * seat_count // 3 else 0
    features["is_late_position"] = 1 if player_idx >= 2 * seat_count // 3 else 0
    
    # Player actions (simulated)
    features["player_action_count"] = context.get("player_action_count", 2)
    features["player_aggressive_actions"] = context.get("player_aggressive_actions", 1)
    features["player_passive_actions"] = context.get("player_passive_actions", 1)
    features["player_folded"] = 0  # We're predicting before fold
    features["player_all_in"] = 0
    features["player_starting_stack"] = context.get("starting_stack", 1000)
    features["player_final_stack"] = context.get("starting_stack", 1000)  # No change yet
    features["player_contributed_to_pot"] = context.get("contributed_to_pot", 10)
    features["player_bet_size"] = context.get("bet_size", 0)
    features["player_stack_change"] = 0
    
    # Board features
    board_features = calculate_board_features(board_cards)
    features.update(board_features)
    
    # Hole card features
    hole_features = calculate_hand_features(hole_cards[0], hole_cards[1])
    features.update(hole_features)
    
    # Add hole cards as categorical features
    features["hole_card1"] = hole_cards[0]
    features["hole_card2"] = hole_cards[1]
    
    # Add board cards as categorical features
    for i in range(5):
        features[f"board_card{i+1}"] = board_cards[i] if i < len(board_cards) else "unknown"
    
    # Add opponent information (simplified)
    for opp_idx in range(seat_count):
        if opp_idx != player_idx:
            features[f"opponent_{opp_idx}_position"] = opp_idx
            features[f"opponent_{opp_idx}_action_count"] = context.get("opponent_action_count", 1)
            features[f"opponent_{opp_idx}_folded"] = 0
            features[f"opponent_{opp_idx}_all_in"] = 0
    
    # Target variable (not used for prediction, but needed for feature consistency)
    features["is_winner"] = 0
    features["winnings"] = 0
    
    return features

def display_card(card, size="medium"):
    """Display a card with proper styling."""
    if card == "unknown":
        return f"<div class='card-display' style='background: #f8f9fa; color: #6c757d;'>{card}</div>"
    
    rank, suit = parse_card(card)
    if not rank or not suit:
        return f"<div class='card-display'>{card}</div>"
    
    # Color coding
    color_class = "red-card" if suit in ["h", "d"] else "black-card"
    
    # Suit symbols
    suit_symbols = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}
    suit_symbol = suit_symbols.get(suit, suit)
    
    return f"<div class='card-display {color_class}'>{rank}{suit_symbol}</div>"

def create_win_probability_chart(probabilities, player_names):
    """Create a bar chart of win probabilities."""
    fig = go.Figure()
    
    # Color coding based on probability
    colors = []
    for prob in probabilities:
        if prob > 0.3:
            colors.append('#28a745')  # Green for high probability
        elif prob > 0.15:
            colors.append('#ffc107')  # Yellow for medium probability
        else:
            colors.append('#dc3545')  # Red for low probability
    
    fig.add_trace(go.Bar(
        x=player_names,
        y=probabilities,
        marker_color=colors,
        text=[f'{p:.1%}' for p in probabilities],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Win Probability: %{y:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Win Probability by Player",
        xaxis_title="Player",
        yaxis_title="Win Probability",
        yaxis_tickformat='.1%',
        yaxis_range=[0, max(probabilities) * 1.2],
        showlegend=False,
        height=400
    )
    
    return fig

def create_hand_strength_analysis(hole_cards, board_cards):
    """Create hand strength analysis visualization."""
    if hole_cards[0] == "unknown" or hole_cards[1] == "unknown":
        return None
    
    # Calculate hand features
    hole_features = calculate_hand_features(hole_cards[0], hole_cards[1])
    board_features = calculate_board_features(board_cards)
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Hole Card Strength', 'Board Analysis', 'Hand Characteristics', 'Position & Context'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Hole card strength gauge
    hole_strength = (hole_features["hole_card_high"] + hole_features["hole_card_low"]) / 28  # Normalize to 0-1
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=hole_strength * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Hole Card Strength"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkblue"},
               'steps': [{'range': [0, 33], 'color': "lightgray"},
                        {'range': [33, 66], 'color': "gray"},
                        {'range': [66, 100], 'color': "darkgray"}]}
    ), row=1, col=1)
    
    # Board strength gauge
    board_strength = board_features["board_broadway_count"] / 5  # Normalize to 0-1
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=board_strength * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Board Strength"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkgreen"},
               'steps': [{'range': [0, 33], 'color': "lightgray"},
                        {'range': [33, 66], 'color': "gray"},
                        {'range': [66, 100], 'color': "darkgray"}]}
    ), row=1, col=2)
    
    # Hand characteristics
    hand_chars = {
        'Suited': hole_features["hole_cards_suited"],
        'Paired': hole_features["hole_cards_paired"],
        'Broadway': hole_features["is_broadway"],
        'High Card': 1 if hole_features["hole_card_high"] >= 12 else 0
    }
    
    fig.add_trace(go.Bar(
        x=list(hand_chars.keys()),
        y=list(hand_chars.values()),
        marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'],
        name="Hand Characteristics"
    ), row=2, col=1)
    
    # Board characteristics
    board_chars = {
        'Broadway Cards': board_features["board_broadway_count"],
        'Aces': board_features["board_ace_count"],
        'Kings': board_features["board_king_count"],
        'Connected': 1 if board_features["board_card_count"] >= 3 else 0
    }
    
    fig.add_trace(go.Bar(
        x=list(board_chars.keys()),
        y=list(board_chars.values()),
        marker_color=['#feca57', '#ff9ff3', '#54a0ff', '#5f27cd'],
        name="Board Characteristics"
    ), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    return fig

def generate_random_features():
    """Generate a full random feature vector for the model (as a dict)."""
    # Load model to get feature names
    model = load_model()
    if model is None:
        return {}
    
    feature_names = model.feature_names_
    features = {}
    
    # Generate a full deck and shuffle
    deck = get_card_options()
    random.shuffle(deck)
    
    # Generate random game parameters
    seat_count = random.randint(2, 10)
    player_idx = random.randint(0, seat_count-1)
    
    # Initialize all features with defaults
    for name in feature_names:
        features[name] = 0  # Default to 0
    
    # Set specific features
    features["seat_count"] = seat_count
    features["player_idx"] = player_idx
    features["position"] = player_idx
    features["opponent_count"] = seat_count - 1
    features["is_heads_up"] = int(seat_count == 2)
    features["is_multiway"] = int(seat_count > 2)
    
    # Game context
    features["pot_size"] = random.randint(500, 5000)
    features["min_bet"] = random.choice([10, 20, 50])
    features["total_antes"] = random.choice([0, 10, 20])
    features["total_blinds"] = random.choice([15, 30, 50])
    
    # Position features
    features["is_button"] = random.randint(0, 1)
    features["is_small_blind"] = random.randint(0, 1)
    features["is_big_blind"] = random.randint(0, 1)
    features["is_early_position"] = random.randint(0, 1)
    features["is_middle_position"] = random.randint(0, 1)
    features["is_late_position"] = random.randint(0, 1)
    
    # Player actions
    features["player_action_count"] = random.randint(0, 6)
    features["player_aggressive_actions"] = random.randint(0, 3)
    features["player_passive_actions"] = random.randint(0, 3)
    features["player_folded"] = 0  # We're predicting before fold
    features["player_all_in"] = random.randint(0, 1)
    features["player_starting_stack"] = random.randint(500, 5000)
    features["player_final_stack"] = features["player_starting_stack"]  # No change yet
    features["player_contributed_to_pot"] = random.randint(0, 500)
    features["player_bet_size"] = random.randint(0, 500)
    features["player_stack_change"] = 0
    
    # Board features
    features["board_card_count"] = random.randint(0, 5)
    features["board_high_card"] = random.randint(0, 14)
    features["board_low_card"] = random.randint(0, 14)
    features["board_rank_sum"] = random.randint(0, 70)
    features["board_broadway_count"] = random.randint(0, 5)
    features["board_ace_count"] = random.randint(0, 4)
    features["board_king_count"] = random.randint(0, 4)
    features["board_queen_count"] = random.randint(0, 4)
    features["board_jack_count"] = random.randint(0, 4)
    features["board_ten_count"] = random.randint(0, 4)
    
    # Hole card features
    features["has_hole_cards"] = 1
    features["hole_card_high"] = random.randint(2, 14)
    features["hole_card_low"] = random.randint(2, 14)
    features["hole_cards_suited"] = random.randint(0, 1)
    features["hole_cards_paired"] = random.randint(0, 1)
    features["hole_card_gap"] = random.randint(0, 12)
    features["hole_card_rank_sum"] = random.randint(4, 28)
    features["is_broadway"] = random.randint(0, 1)
    features["is_ace"] = random.randint(0, 1)
    features["is_king"] = random.randint(0, 1)
    features["is_queen"] = random.randint(0, 1)
    features["is_jack"] = random.randint(0, 1)
    features["is_ten"] = random.randint(0, 1)
    
    # Deal cards
    if len(deck) >= 7:  # Need 2 hole + 5 board
        features["hole_card1"] = deck.pop()
        features["hole_card2"] = deck.pop()
        features["board_card1"] = deck.pop()
        features["board_card2"] = deck.pop()
        features["board_card3"] = deck.pop()
        features["board_card4"] = deck.pop()
        features["board_card5"] = deck.pop()
    else:
        features["hole_card1"] = "unknown"
        features["hole_card2"] = "unknown"
        features["board_card1"] = "unknown"
        features["board_card2"] = "unknown"
        features["board_card3"] = "unknown"
        features["board_card4"] = "unknown"
        features["board_card5"] = "unknown"
    
    # Opponent features (simplified)
    for i in range(seat_count):
        if i != player_idx:
            features[f"opponent_{i}_position"] = i
            features[f"opponent_{i}_action_count"] = random.randint(0, 6)
            features[f"opponent_{i}_folded"] = random.randint(0, 1)
            features[f"opponent_{i}_all_in"] = random.randint(0, 1)
    
    # Target variables (not used for prediction)
    features["is_winner"] = 0
    features["winnings"] = 0
    
    return features

# --- UI ---
st.title("♠️ Poker Model Minimal Demo")
st.markdown("""
Minimal demo: generate a full random model input, view all features, and predict win probability.
""")

if "random_features" not in st.session_state:
    st.session_state.random_features = None

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🎲 Generate Random Model Input Features", use_container_width=True):
        st.session_state.random_features = generate_random_features()

with col2:
    if st.session_state.random_features:
        features = st.session_state.random_features
        # Minimalistic visualization: show as table, highlight key features
        df = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])
        # Convert all values to strings to avoid Arrow conversion issues
        df["Value"] = df["Value"].astype(str)
        
        def highlight(row):
            if any(k in row["Feature"] for k in ["hole_card", "board_card"]):
                return ["background-color: #e3f2fd"]*2
            if row["Feature"] in ["player_idx", "seat_count", "position"]:
                return ["background-color: #fff3e0"]*2
            return [""]*2
        
        # Show only first 100 features to avoid overwhelming the UI
        st.dataframe(df.head(100).style.apply(highlight, axis=1), use_container_width=True, height=400)
        if len(df) > 100:
            st.info(f"Showing first 100 of {len(df)} features. Use the scrollbar to see more.")
    else:
        st.info("Click the button to generate random model input features.")

st.divider()

if st.session_state.random_features:
    if st.button("🤖 Predict Win Probability", use_container_width=True):
        model = load_model()
        if model is not None:
            X = pd.DataFrame([st.session_state.random_features])
            # Ensure categorical features are string
            cat_features = ['hole_card1', 'hole_card2', 'board_card1', 'board_card2', 'board_card3', 'board_card4', 'board_card5']
            for col in cat_features:
                if col in X.columns:
                    X[col] = X[col].astype(str)
            # Ensure correct order
            X = X[model.feature_names_]
            prob = model.predict_proba(X)[0, 1]
            st.success(f"Win Probability: {prob:.1%}")
        else:
            st.error("Failed to load model")

# App runs automatically when script is executed 
