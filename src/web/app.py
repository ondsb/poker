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
import joblib

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Poker Win Probability Demo",
    page_icon="♠️",
    layout="wide"
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
    .player-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .probability-high {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .probability-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .probability-low {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>♠️ Poker Win Probability Demo (Focused Model)</h1>
    <p>Generate a random 6-player hand and predict win probability for each player using the focused model with opponent features.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_focused_model():
    """Load the trained focused model and metadata."""
    model_path = Path("models/focused/poker_model.cbm")
    metadata_path = Path("models/focused/model_metadata.joblib")
    encoders_path = Path("models/focused/label_encoders.joblib")
    features_path = Path("models/focused/feature_columns.txt")
    
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}. Please train the model first.")
        return None, None, None, None
    
    try:
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        
        metadata = joblib.load(metadata_path) if metadata_path.exists() else None
        label_encoders = joblib.load(encoders_path) if encoders_path.exists() else None
        
        # Load feature columns
        feature_columns = []
        if features_path.exists():
            with open(features_path, 'r') as f:
                feature_columns = [line.strip() for line in f.readlines()]
        
        return model, metadata, label_encoders, feature_columns
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

CARD_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
CARD_SUITS = ["h", "d", "c", "s"]

def get_deck():
    """Get a fresh deck of cards."""
    return [f"{rank}{suit}" for rank in CARD_RANKS for suit in CARD_SUITS]

def parse_card(card_str):
    """Parse a card string into rank and suit."""
    if not card_str or card_str == "unknown":
        return None, None
    return card_str[:-1], card_str[-1]

def player_hand_features(card1, card2):
    """Calculate hand features for a player."""
    rank_map = {r: i+2 for i, r in enumerate(CARD_RANKS)}
    if card1 == "unknown" or card2 == "unknown":
        return dict(
            has_hole_cards=0, 
            hole_high_card=0, 
            hole_low_card=0, 
            hole_is_paired=0, 
            hole_is_suited=0, 
            hole_is_broadway=0, 
            hole_has_ace=0
        )
    
    r1, s1 = parse_card(card1)
    r2, s2 = parse_card(card2)
    n1, n2 = rank_map.get(r1, 0), rank_map.get(r2, 0)
    
    return dict(
        has_hole_cards=1,
        hole_high_card=max(n1, n2),
        hole_low_card=min(n1, n2),
        hole_is_paired=int(r1 == r2),
        hole_is_suited=int(s1 == s2),
        hole_is_broadway=int(max(n1, n2) >= 10),
        hole_has_ace=int(n1 == 14 or n2 == 14)
    )

def board_features(board):
    """Calculate board features."""
    rank_map = {r: i+2 for i, r in enumerate(CARD_RANKS)}
    ranks = [rank_map.get(parse_card(c)[0], 0) for c in board if c != "unknown"]
    suits = [parse_card(c)[1] for c in board if c != "unknown"]
    
    is_paired = int(len(set(ranks)) < len(ranks)) if ranks else 0
    is_suited = int(any(suits.count(s) >= 3 for s in set(suits))) if suits else 0
    
    return dict(
        board_card_count=len(ranks),
        board_high_card=max(ranks) if ranks else 0,
        board_has_broadway=int(any(r >= 10 for r in ranks)),
        board_has_ace=int(any(r == 14 for r in ranks)),
        board_is_paired=is_paired,
        board_is_suited=is_suited
    )

def player_aggressiveness():
    """Generate random aggressiveness features."""
    aggressive = random.randint(0, 3)
    passive = random.randint(0, 3)
    total = aggressive + passive + random.randint(0, 2)
    ratio = aggressive / max(total, 1)
    
    return dict(
        player_aggressive_actions=aggressive,
        player_passive_actions=passive,
        player_total_actions=total,
        player_aggressiveness_ratio=ratio
    )

def create_player_perspective_features(players, target_player_idx):
    """Create features from a specific player's perspective with opponent features."""
    target_player = players[target_player_idx]
    
    # Base features for the target player
    features = {
        "player_idx": target_player_idx,
        "seat_count": len(players),
        "pot_size": target_player["pot_size"],
        "table": "demo_table",
        "position": target_player_idx,
        "is_button": target_player["is_button"],
        "is_small_blind": target_player["is_small_blind"],
        "is_big_blind": target_player["is_big_blind"],
        "is_early_position": target_player["is_early_position"],
        "is_middle_position": target_player["is_middle_position"],
        "is_late_position": target_player["is_late_position"],
        "player_starting_stack": target_player["player_starting_stack"],
        "player_contributed_to_pot": 0,
        "player_bet_size": 0,
    }
    
    # Add hole card features
    hole_features = player_hand_features(target_player["hole_card1"], target_player["hole_card2"])
    features.update(hole_features)
    
    # Add hole cards as categorical features
    features["hole_card1"] = target_player["hole_card1"]
    features["hole_card2"] = target_player["hole_card2"]
    
    # Add board cards
    board = [target_player[f"board_card{i+1}"] for i in range(5)]
    for i in range(5):
        features[f"board_card{i+1}"] = board[i] if i < len(board) else "unknown"
    
    # Add board features
    features.update(board_features(board))
    
    # Add opponent features (up to 5 opponents)
    opponent_count = 0
    for opp_idx, opp_player in enumerate(players):
        if opp_idx != target_player_idx and opponent_count < 5:
            opp_hole_features = player_hand_features(opp_player["hole_card1"], opp_player["hole_card2"])
            
            features[f"opp_{opponent_count+1}_has_hole_cards"] = opp_hole_features["has_hole_cards"]
            features[f"opp_{opponent_count+1}_high_card"] = opp_hole_features["hole_high_card"]
            features[f"opp_{opponent_count+1}_low_card"] = opp_hole_features["hole_low_card"]
            features[f"opp_{opponent_count+1}_is_paired"] = opp_hole_features["hole_is_paired"]
            features[f"opp_{opponent_count+1}_is_suited"] = opp_hole_features["hole_is_suited"]
            features[f"opp_{opponent_count+1}_is_broadway"] = opp_hole_features["hole_is_broadway"]
            
            opponent_count += 1
    
    # Fill remaining opponent slots with zeros
    for opp_num in range(opponent_count + 1, 6):
        features[f"opp_{opp_num}_has_hole_cards"] = 0
        features[f"opp_{opp_num}_high_card"] = 0
        features[f"opp_{opp_num}_low_card"] = 0
        features[f"opp_{opp_num}_is_paired"] = 0
        features[f"opp_{opp_num}_is_suited"] = 0
        features[f"opp_{opp_num}_is_broadway"] = 0
    
    # Add player aggressiveness
    features.update(target_player["aggressiveness"])
    
    # Target variable (not used for prediction, just for structure)
    features["is_winner"] = 0
    features["winnings"] = 0
    
    return features

def generate_random_hand():
    """Generate a random 6-player poker hand."""
    deck = get_deck()
    random.shuffle(deck)
    
    seat_count = 6
    board = [deck.pop() for _ in range(5)]
    players = []
    
    for i in range(seat_count):
        card1, card2 = deck.pop(), deck.pop()
        
        # Base player features
        features = {
            "player_idx": i,
            "seat_count": seat_count,
            "pot_size": random.randint(500, 5000),
            "min_bet": random.choice([10, 20, 50]),
            "position": i,
            "is_button": int(i == 0),
            "is_small_blind": int(i == 1),
            "is_big_blind": int(i == 2),
            "is_early_position": int(i < 2),
            "is_middle_position": int(2 <= i < 4),
            "is_late_position": int(i >= 4),
            "player_starting_stack": random.randint(500, 5000),
            "player_contributed_to_pot": random.randint(0, 500),
            "player_bet_size": random.randint(0, 500),
            "hole_card1": card1,
            "hole_card2": card2,
        }
        
        # Add hand features
        features.update(player_hand_features(card1, card2))
        
        # Add board cards
        for j in range(5):
            features[f"board_card{j+1}"] = board[j] if j < len(board) else "unknown"
        
        # Add board features
        features.update(board_features(board))
        
        # Add aggressiveness
        features["aggressiveness"] = player_aggressiveness()
        
        # Target variables
        features["is_winner"] = 0
        features["winnings"] = 0
        
        players.append(features)
    
    return players, board

def get_probability_class(prob):
    """Get CSS class for probability styling."""
    if prob >= 0.3:
        return "probability-high"
    elif prob >= 0.15:
        return "probability-medium"
    else:
        return "probability-low"

def display_card(card, suit_color=True):
    """Display a card with proper styling."""
    if card == "unknown":
        return f'<div class="card-display">??</div>'
    
    rank, suit = parse_card(card)
    color_class = "red-card" if suit in ["h", "d"] else "black-card"
    suit_symbol = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}.get(suit, suit)
    
    return f'<div class="card-display {color_class}">{rank}{suit_symbol}</div>'

# Main app logic
if "random_hand" not in st.session_state:
    st.session_state.random_hand = None
    st.session_state.board = None

if st.button("🎲 Generate Random 6-Player Hand", use_container_width=True):
    players, board = generate_random_hand()
    st.session_state.random_hand = players
    st.session_state.board = board

if st.session_state.random_hand:
    players = st.session_state.random_hand
    board = st.session_state.board
    
    # Load model
    model, metadata, label_encoders, feature_columns = load_focused_model()
    
    if model is not None:
        st.markdown("### 🃏 Current Hand")
        
        # Display board
        st.markdown("**Community Cards:**")
        board_html = " ".join([display_card(card) for card in board])
        st.markdown(board_html, unsafe_allow_html=True)
        
        # Generate predictions for each player
        predictions = []
        
        for player_idx in range(len(players)):
            # Create features from this player's perspective
            player_features = create_player_perspective_features(players, player_idx)
            
            # Convert to DataFrame
            df_player = pd.DataFrame([player_features])
            
            # Encode categorical features
            if label_encoders:
                for col, encoder in label_encoders.items():
                    if col in df_player.columns:
                        df_player[f"{col}_encoded"] = encoder.transform(df_player[col].astype(str))
                        df_player = df_player.drop(columns=[col])
            
            # Ensure correct feature order
            if feature_columns:
                missing_cols = set(feature_columns) - set(df_player.columns)
                for col in missing_cols:
                    df_player[col] = 0
                X = df_player[feature_columns]
            else:
                X = df_player.drop(columns=['is_winner', 'winnings'], errors='ignore')
            
            # Make prediction
            prob = model.predict_proba(X)[0, 1]
            predictions.append({
                'player_idx': player_idx,
                'probability': prob,
                'hole_cards': [players[player_idx]["hole_card1"], players[player_idx]["hole_card2"]],
                'position': players[player_idx]["position"],
                'stack': players[player_idx]["player_starting_stack"],
                'aggressiveness': players[player_idx]["aggressiveness"]["player_aggressiveness_ratio"]
            })
        
        # Sort by probability
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Display results
        st.markdown("### 📊 Win Probability Predictions")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["🎯 Player View", "📈 Chart View", "📋 Detailed Table"])
        
        with tab1:
            # Player cards view
            cols = st.columns(3)
            for i, pred in enumerate(predictions):
                with cols[i % 3]:
                    prob_class = get_probability_class(pred['probability'])
                    st.markdown(f"""
                    <div class="player-card {prob_class}">
                        <h4>Player {pred['player_idx']}</h4>
                        <p><strong>Win Probability: {pred['probability']:.1%}</strong></p>
                        <p>Position: {pred['position']} | Stack: ${pred['stack']:,}</p>
                        <p>Aggressiveness: {pred['aggressiveness']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display hole cards
                    card1_html = display_card(pred['hole_cards'][0])
                    card2_html = display_card(pred['hole_cards'][1])
                    st.markdown(f"<p>Hole Cards: {card1_html} {card2_html}</p>", unsafe_allow_html=True)
        
        with tab2:
            # Chart view
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=[f"Player {p['player_idx']}" for p in predictions],
                y=[p['probability'] for p in predictions],
                text=[f"{p['probability']:.1%}" for p in predictions],
                textposition='auto',
                marker_color=['#4facfe' if p['probability'] >= 0.3 else 
                             '#f093fb' if p['probability'] >= 0.15 else '#ffecd2' 
                             for p in predictions]
            ))
            
            fig.update_layout(
                title="Win Probability by Player",
                xaxis_title="Player",
                yaxis_title="Win Probability",
                yaxis=dict(tickformat='.1%'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Detailed table
            df_results = pd.DataFrame(predictions)
            df_results['Win Probability'] = df_results['probability'].apply(lambda x: f"{x:.1%}")
            df_results['Hole Cards'] = df_results['hole_cards'].apply(lambda x: f"{x[0]} {x[1]}")
            df_results['Aggressiveness'] = df_results['aggressiveness'].apply(lambda x: f"{x:.2f}")
            
            display_cols = ['player_idx', 'Hole Cards', 'Win Probability', 'position', 'stack', 'Aggressiveness']
            st.dataframe(df_results[display_cols], use_container_width=True)
        
        # Model info
        with st.expander("ℹ️ Model Information"):
            st.markdown(f"""
            - **Model Type**: CatBoost Classifier
            - **Features Used**: {len(feature_columns) if feature_columns else 'Unknown'}
            - **Training Data**: 194,303 samples
            - **Model Performance**: 92% ROC AUC
            - **Key Features**: Position, aggressiveness, opponent hand strength
            """)
    
    else:
        st.error("❌ Model not loaded. Please ensure the model files exist in models/focused/")
else:
    st.info("🎲 Click the button above to generate a random 6-player hand and see win probability predictions!")

# App runs automatically when script is executed 
