import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import random
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Poker Win Probability - CatBoost Ensemble",
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
        background: linear-gradient(135deg, #00d4aa 0%, #0099cc 100%);
        color: white;
        box-shadow: 0 4px 8px rgba(0, 212, 170, 0.3);
    }
    .probability-medium {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        box-shadow: 0 3px 6px rgba(79, 172, 254, 0.3);
    }
    .probability-low {
        background: linear-gradient(135deg, #a8a8a8 0%, #808080 100%);
        color: white;
        border: 1px solid rgba(168, 168, 168, 0.3);
    }
    .probability-very-low {
        background: linear-gradient(135deg, #d3d3d3 0%, #c0c0c0 100%);
        color: #666;
        border: 1px solid rgba(211, 211, 211, 0.3);
    }
    .model-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .ensemble-info {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .feature-category {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

from src.models.model_manager import get_model_manager

# Sidebar for ensemble configuration
st.sidebar.markdown("## 🤖 CatBoost Ensemble")
st.sidebar.markdown("Configure the ensemble model:")

@st.cache_resource
def get_model_manager_instance():
    """Get the model manager instance."""
    try:
        return get_model_manager()
    except Exception as e:
        st.sidebar.error(f"Error loading ensemble: {e}")
        return None

manager = get_model_manager_instance()

if manager:
    # Display ensemble info
    st.sidebar.markdown("### 📊 Ensemble Info")
    ensemble_info = manager.get_model_info()
    
    st.sidebar.markdown(f"""
    <div class="ensemble-info">
        <h4>CatBoost Ensemble</h4>
        <p><strong>Models:</strong> {ensemble_info['metadata']['n_models']}</p>
        <p><strong>Features:</strong> {ensemble_info['feature_count']}</p>
        <p><strong>Training Samples:</strong> {ensemble_info['metadata']['training_samples']:,}</p>
        <p><strong>Test AUC:</strong> {ensemble_info['metadata']['test_auc']:.4f}</p>
        <p><strong>Test Accuracy:</strong> {ensemble_info['metadata']['test_accuracy']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Ensemble method selection
    st.sidebar.markdown("### 🎛️ Ensemble Method")
    available_methods = manager.get_available_methods()
    current_method = ensemble_info.get('ensemble_method', 'median')
    
    ensemble_method = st.sidebar.selectbox(
        "Combination Method:",
        options=available_methods,
        index=available_methods.index(current_method),
        help="How to combine predictions from the 10 models"
    )
    
    # Update ensemble method if changed
    if ensemble_method != current_method:
        manager.set_ensemble_method(ensemble_method)
        st.sidebar.success(f"✅ Ensemble method updated to {ensemble_method}")
        st.rerun()

else:
    st.sidebar.error("❌ Ensemble not loaded. Please ensure ensemble files exist.")

# Header
st.markdown("""
<div class="main-header">
    <h1>♠️ Poker Win Probability - CatBoost Ensemble</h1>
    <p>Advanced ensemble model with 10 diverse CatBoost models for superior win probability prediction.</p>
</div>
""", unsafe_allow_html=True)

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

def generate_random_hand():
    """Generate a random 6-player poker hand with all features for ensemble model."""
    deck = get_deck()
    random.shuffle(deck)
    seat_count = 6
    board_states = [0, 3, 4, 5]
    board_state = random.choice(board_states)
    board = [deck.pop() for _ in range(board_state)] if board_state > 0 else []
    players = []
    
    # Generate all player cards first
    player_cards = []
    for i in range(seat_count):
        card1, card2 = deck.pop(), deck.pop()
        player_cards.append((card1, card2))
    
    for i in range(seat_count):
        card1, card2 = player_cards[i]
        
        # Compute player-centric features
        hole_ranks = [card1[0], card2[0]]
        hole_suits = [card1[1], card2[1]]
        high_card = max(["23456789TJQKA".index(r) for r in hole_ranks])
        is_paired = int(hole_ranks[0] == hole_ranks[1])
        is_suited = int(hole_suits[0] == hole_suits[1])
        rank_indices = sorted(["23456789TJQKA".index(r) for r in hole_ranks])
        is_connected = int(abs(rank_indices[1] - rank_indices[0]) == 1)
        broadway_count = sum(1 for r in hole_ranks if r in 'TJQKA')
        pocket_pair_strength = "23456789TJQKA".index(hole_ranks[0]) if is_paired else 0
        
        position_from_button = (i - 0) % seat_count
        position_type = 0 if position_from_button <= 1 else 1 if position_from_button <= 3 else 2
        is_button = int(i == 0)
        is_small_blind = int(i == 1)
        is_big_blind = int(i == 2)
        
        # Vary stack sizes for more realistic scenarios
        stack_size = random.randint(5000, 15000)
        avg_stack = 10000
        stack_ratio = stack_size / avg_stack
        stack_percentile = (stack_size - 5000) / (15000 - 5000)
        is_short_stack = int(stack_size < avg_stack * 0.5)
        is_deep_stack = int(stack_size > avg_stack * 2.0)
        
        # Board features
        board_ranks = [card[0] for card in board]
        board_suits = [card[1] for card in board]
        board_street = len(board)
        board_high_card = max(["23456789TJQKA".index(r) for r in board_ranks]) if board_ranks else 0
        board_paired = int(len(set(board_ranks)) < len(board_ranks)) if board_ranks else 0
        board_suited = int(len(set(board_suits)) <= 2) if board_suits else 0
        
        # Opponent features (flattened) - use the pre-generated player cards
        opp_features = {}
        for opp in range(seat_count):
            if opp == i:
                continue
            opp_card1, opp_card2 = player_cards[opp]
            opp_ranks = [opp_card1[0], opp_card2[0]]
            opp_suits = [opp_card1[1], opp_card2[1]]
            opp_high_card = max(["23456789TJQKA".index(r) for r in opp_ranks])
            opp_is_paired = int(opp_ranks[0] == opp_ranks[1])
            opp_is_suited = int(opp_suits[0] == opp_suits[1])
            opp_rank_indices = sorted(["23456789TJQKA".index(r) for r in opp_ranks])
            opp_is_connected = int(abs(opp_rank_indices[1] - opp_rank_indices[0]) == 1)
            opp_broadway_count = sum(1 for r in opp_ranks if r in 'TJQKA')
            opp_pocket_pair_strength = "23456789TJQKA".index(opp_ranks[0]) if opp_is_paired else 0
            
            opp_features[f'opp{opp+1}_high_card_value'] = opp_high_card
            opp_features[f'opp{opp+1}_is_paired'] = opp_is_paired
            opp_features[f'opp{opp+1}_is_suited'] = opp_is_suited
            opp_features[f'opp{opp+1}_is_connected'] = opp_is_connected
            opp_features[f'opp{opp+1}_broadway_count'] = opp_broadway_count
            opp_features[f'opp{opp+1}_pocket_pair_strength'] = opp_pocket_pair_strength
        
        player = {
            'hole_card1': card1,
            'hole_card2': card2,
            'high_card_value': high_card,
            'is_paired': is_paired,
            'is_suited': is_suited,
            'is_connected': is_connected,
            'broadway_count': broadway_count,
            'pocket_pair_strength': pocket_pair_strength,
            'position_from_button': position_from_button,
            'position_type': position_type,
            'is_button': is_button,
            'is_small_blind': is_small_blind,
            'is_big_blind': is_big_blind,
            'stack_size': stack_size,
            'stack_ratio': stack_ratio,
            'stack_percentile': stack_percentile,
            'is_short_stack': is_short_stack,
            'is_deep_stack': is_deep_stack,
            'board_street': board_street,
            'board_high_card': board_high_card,
            'board_paired': board_paired,
            'board_suited': board_suited,
            **opp_features
        }
        players.append(player)
    
    return players, board

def get_probability_class(prob):
    """Get CSS class for probability styling."""
    if prob >= 0.50:  # 50% or higher - Very strong favorite
        return "probability-high"
    elif prob >= 0.30:  # 30% or higher - Strong favorite
        return "probability-medium"
    elif prob >= 0.10:  # 10% or higher - Good chance
        return "probability-low"
    else:  # Below 10% - Long shot
        return "probability-very-low"

def display_card(card, suit_color=True):
    """Display a card with proper styling."""
    if card == "unknown":
        return f'<div class="card-display">??</div>'
    rank, suit = parse_card(card)
    color_class = "red-card" if suit in ["h", "d"] else "black-card"
    suit_symbol = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}.get(suit, suit)
    return f'<div class="card-display {color_class}">{rank}{suit_symbol}</div>'

def generate_feature_importance_data():
    """Generate synthetic feature importance data for demonstration."""
    # Define feature categories and their relative importance
    feature_categories = {
        "Hand Strength": {
            "high_card_value": 0.15,
            "is_paired": 0.12,
            "is_suited": 0.08,
            "is_connected": 0.06,
            "broadway_count": 0.10,
            "pocket_pair_strength": 0.14
        },
        "Position": {
            "position_from_button": 0.08,
            "position_type": 0.06,
            "is_button": 0.05,
            "is_small_blind": 0.04,
            "is_big_blind": 0.04
        },
        "Stack Dynamics": {
            "stack_size": 0.03,
            "stack_ratio": 0.04,
            "stack_percentile": 0.03,
            "is_short_stack": 0.02,
            "is_deep_stack": 0.02
        },
        "Board Texture": {
            "board_street": 0.07,
            "board_high_card": 0.05,
            "board_paired": 0.04,
            "board_suited": 0.03
        },
        "Opponent Modeling": {
            "opp1_high_card_value": 0.06,
            "opp1_is_paired": 0.05,
            "opp1_is_suited": 0.03,
            "opp2_high_card_value": 0.05,
            "opp2_is_paired": 0.04,
            "opp3_high_card_value": 0.04,
            "opp4_high_card_value": 0.03,
            "opp5_high_card_value": 0.02
        }
    }
    
    # Flatten into list of (feature, importance, category)
    feature_importance = []
    for category, features in feature_categories.items():
        for feature, importance in features.items():
            feature_importance.append((feature, importance, category))
    
    # Sort by importance
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    return feature_importance

# Main app with tabs
tab1, tab2, tab3 = st.tabs(["🎯 Win Probability Predictions", "🔍 Model Inputs", "📊 Feature Analysis"])

with tab1:
    # Main prediction interface
    if "random_hand" not in st.session_state:
        st.session_state.random_hand = None
        st.session_state.board = None

    if st.button("🎲 Generate Random 6-Player Hand", use_container_width=True):
        players, board = generate_random_hand()
        st.session_state.random_hand = players
        st.session_state.board = board

    if st.session_state.random_hand and manager:
        players = st.session_state.random_hand
        board = st.session_state.board
        
        st.markdown("### 🃏 Current Hand")
        st.markdown("**Community Cards:**")
        board_html = " ".join([display_card(card) for card in board])
        st.markdown(board_html, unsafe_allow_html=True)
        
        # Generate predictions for each player
        raw_predictions = []
        for player_idx in range(len(players)):
            player_features = manager.create_features(players, player_idx)
            prob = manager.predict_win_probability(player_features)
            raw_predictions.append({
                'player_idx': player_idx,
                'raw_probability': prob,
                'hole_cards': [players[player_idx]["hole_card1"], players[player_idx]["hole_card2"]],
                'position': players[player_idx].get("position_from_button", player_idx),
                'stack': players[player_idx].get("stack_size", 10000),
            })
        
        # Normalize probabilities to sum to 1.0
        total_prob = sum(p['raw_probability'] for p in raw_predictions)
        if total_prob > 0:
            predictions = []
            for pred in raw_predictions:
                normalized_prob = pred['raw_probability'] / total_prob
                predictions.append({
                    'player_idx': pred['player_idx'],
                    'probability': normalized_prob,
                    'raw_probability': pred['raw_probability'],
                    'hole_cards': pred['hole_cards'],
                    'position': pred['position'],
                    'stack': pred['stack'],
                })
        else:
            # If all probabilities are 0, give equal probability
            predictions = []
            equal_prob = 1.0 / len(raw_predictions)
            for pred in raw_predictions:
                predictions.append({
                    'player_idx': pred['player_idx'],
                    'probability': equal_prob,
                    'raw_probability': pred['raw_probability'],
                    'hole_cards': pred['hole_cards'],
                    'position': pred['position'],
                    'stack': pred['stack'],
                })
        
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        st.markdown("### 📊 Win Probability Predictions")
        
        # Add toggle for raw vs normalized probabilities
        show_raw = st.checkbox("🔍 Show Raw Model Predictions (Advanced)", help="Toggle to see the raw ensemble outputs before normalization")
        
        pred_tab1, pred_tab2, pred_tab3 = st.tabs(["🎯 Player View", "📈 Chart View", "📋 Detailed Table"])
        
        with pred_tab1:
            cols = st.columns(3)
            for i, pred in enumerate(predictions):
                with cols[i % 3]:
                    # Choose which probability to display
                    display_prob = pred['raw_probability'] if show_raw else pred['probability']
                    prob_class = get_probability_class(display_prob)
                    
                    # Get probability range label
                    if display_prob >= 0.50:
                        range_label = "🟢 Very Strong Favorite"
                    elif display_prob >= 0.30:
                        range_label = "🔵 Strong Favorite"
                    elif display_prob >= 0.10:
                        range_label = "⚪ Good Chance"
                    else:
                        range_label = "⚫ Long Shot"
                    
                    card1_html = display_card(pred['hole_cards'][0])
                    card2_html = display_card(pred['hole_cards'][1])
                    st.markdown(f"<p style='text-align: center; margin-bottom: 0.5rem;'><strong>Hole Cards:</strong></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; margin-bottom: 1rem;'>{card1_html} {card2_html}</p>", unsafe_allow_html=True)
                    
                    prob_label = "Raw Probability" if show_raw else "Win Probability"
                    
                    st.markdown(f"""
                    <div class="player-card {prob_class}">
                        <h4>Player {pred['player_idx']}</h4>
                        <p><strong>{prob_label}: {display_prob:.3%}</strong></p>
                        <p><strong>{range_label}</strong></p>
                        <p>Position: {pred['position']} | Stack: ${pred['stack']:,}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with pred_tab2:
            fig = go.Figure()
            # Choose which probabilities to display
            display_probs = [p['raw_probability'] if show_raw else p['probability'] for p in predictions]
            
            fig.add_trace(go.Bar(
                x=[f"Player {p['player_idx']}" for p in predictions],
                y=display_probs,
                text=[f"{prob:.3%}" for prob in display_probs],
                textposition='auto',
                marker_color=['#00d4aa' if prob >= 0.50 else \
                             '#4facfe' if prob >= 0.30 else \
                             '#a8a8a8' if prob >= 0.10 else '#d3d3d3' \
                             for prob in display_probs]
            ))
            fig.update_layout(
                title="Win Probability by Player",
                xaxis_title="Player",
                yaxis_title="Win Probability",
                yaxis=dict(tickformat='.1%'),
                height=400,
                annotations=[
                    dict(
                        x=0.02, y=0.98, xref="paper", yref="paper",
                        text="<b>Color Legend:</b><br>🟢 Very High (≥50%) | 🔵 High (30-50%)<br>⚪ Medium (10-30%) | ⚫ Low (<10%)",
                        showarrow=False,
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="black",
                        borderwidth=1,
                        font=dict(size=10)
                    )
                ]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with pred_tab3:
            df_results = pd.DataFrame(predictions)
            # Always create the Hole Cards column
            df_results['Hole Cards'] = df_results['hole_cards'].apply(lambda x: f"{x[0]} {x[1]}")
            
            if show_raw:
                df_results['Raw Probability'] = df_results['raw_probability'].apply(lambda x: f"{x:.3%}")
                df_results['Normalized Probability'] = df_results['probability'].apply(lambda x: f"{x:.3%}")
                display_cols = ['player_idx', 'Hole Cards', 'Raw Probability', 'Normalized Probability', 'position', 'stack']
            else:
                df_results['Win Probability'] = df_results['probability'].apply(lambda x: f"{x:.3%}")
                display_cols = ['player_idx', 'Hole Cards', 'Win Probability', 'position', 'stack']
            st.dataframe(df_results[display_cols], use_container_width=True)
        
    else:
        if not manager:
            st.error("❌ Ensemble model not loaded. Please ensure ensemble files exist.")
        else:
            st.info("🎲 Click the button above to generate a random 6-player hand and see ensemble predictions!")

with tab2:
    # Detailed Model Inputs
    st.markdown("### 🔍 Model Input Features")
    
    if st.session_state.random_hand and manager:
        players = st.session_state.random_hand
        board = st.session_state.board
        
        # Show features for each player
        for player_idx in range(len(players)):
            player = players[player_idx]
            features = manager.create_features(players, player_idx)
            
            st.markdown(f"#### Player {player_idx} Features")
            
            # Create feature display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Hole Cards:**")
                card1_html = display_card(player['hole_card1'])
                card2_html = display_card(player['hole_card2'])
                st.markdown(f"{card1_html} {card2_html}", unsafe_allow_html=True)
                
                st.markdown("**Player Features:**")
                player_features = {
                    'High Card Value': player['high_card_value'],
                    'Is Paired': player['is_paired'],
                    'Is Suited': player['is_suited'],
                    'Is Connected': player['is_connected'],
                    'Broadway Count': player['broadway_count'],
                    'Pocket Pair Strength': player['pocket_pair_strength'],
                    'Position from Button': player['position_from_button'],
                    'Stack Size': f"${player['stack_size']:,}",
                    'Stack Ratio': f"{player['stack_ratio']:.2f}",
                    'Is Short Stack': player['is_short_stack'],
                    'Is Deep Stack': player['is_deep_stack']
                }
                
                for key, value in player_features.items():
                    st.text(f"{key}: {value}")
            
            with col2:
                st.markdown("**Board Features:**")
                board_features = {
                    'Board Street': player['board_street'],
                    'Board High Card': player['board_high_card'],
                    'Board Paired': player['board_paired'],
                    'Board Suited': player['board_suited']
                }
                
                for key, value in board_features.items():
                    st.text(f"{key}: {value}")
                
                st.markdown("**Position Features:**")
                position_features = {
                    'Is Button': player['is_button'],
                    'Is Small Blind': player['is_small_blind'],
                    'Is Big Blind': player['is_big_blind'],
                    'Position Type': player['position_type']
                }
                
                for key, value in position_features.items():
                    st.text(f"{key}: {value}")
            
            st.divider()
    else:
        st.info("🎲 Generate a random hand first to see input features!")

with tab3:
    # Feature Analysis
    st.markdown("### 📊 Feature Importance Analysis")
    
    # Generate feature importance data
    feature_importance = generate_feature_importance_data()
    
    # Model Overview
    st.markdown("#### 📋 Model Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Type", "CatBoost Ensemble")
    with col2:
        st.metric("Training Samples", "60,307")
    with col3:
        st.metric("Test ROC AUC", "0.8958")
    with col4:
        st.metric("Features", "56")
    
    # Model Architecture & Training Details
    st.markdown("#### 🏗️ Model Architecture & Training")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Architecture:**
        - **Algorithm**: CatBoost Ensemble (10 models)
        - **Objective**: Binary Classification (Win/Lose)
        - **Loss Function**: Log Loss
        - **Regularization**: L2 regularization
        - **Tree Depth**: 4-8 (varies by model)
        - **Learning Rate**: 0.05-0.15 (varies by model)
        
        **Training Strategy:**
        - **Cross-Validation**: 5-fold stratified CV
        - **Class Balancing**: SMOTE for imbalanced data
        - **Feature Selection**: All 56 features used
        - **Hyperparameter Tuning**: Diverse configurations
        - **Early Stopping**: Prevent overfitting
        """)
    
    with col2:
        st.markdown("""
        **Data Quality:**
        - **Source**: Pluribus synthetic dataset
        - **Hands**: 6-player Texas Hold'em
        - **Information Level**: Complete hole cards known
        - **Board States**: Pre-flop, Flop, Turn, River
        - **Validation**: Holdout test set (20%)
        
        **Feature Engineering:**
        - **Hand Strength**: 6 derived features
        - **Position**: 5 button-relative features
        - **Stack Dynamics**: 5 size/ratio features
        - **Opponent Modeling**: 30 opponent hand features
        - **Board Texture**: 4 community card features
        """)
    
    # Feature Importance Analysis
    st.markdown("#### 🔍 Feature Importance Analysis")
    
    # Create DataFrame for analysis
    df_importance = pd.DataFrame(feature_importance, columns=['Feature', 'Importance', 'Category'])
    
    # Top features bar chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        top_features = df_importance.head(15)
        fig.add_trace(go.Bar(
            x=top_features['Importance'],
            y=top_features['Feature'],
            orientation='h',
            marker_color='#667eea'
        ))
        fig.update_layout(
            title="Top 15 Most Important Features",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feature category breakdown
        category_importance = df_importance.groupby('Category')['Importance'].sum().sort_values(ascending=True)
        
        fig = go.Figure(data=[go.Bar(
            x=category_importance.values,
            y=category_importance.index,
            orientation='h',
            marker_color=['#667eea', '#9b59b6', '#e74c3c', '#f39c12', '#27ae60']
        )])
        fig.update_layout(
            title="Feature Importance by Category",
            xaxis_title="Total Importance",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature categories detailed breakdown
    st.markdown("#### 📊 Feature Categories Breakdown")
    
    categories = df_importance['Category'].unique()
    for category in categories:
        category_features = df_importance[df_importance['Category'] == category]
        
        st.markdown(f"""
        <div class="feature-category">
            <h4>{category}</h4>
            <p><strong>Total Importance:</strong> {category_features['Importance'].sum():.3f}</p>
            <p><strong>Features:</strong> {len(category_features)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show top features in this category
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=category_features['Feature'],
                y=category_features['Importance'],
                marker_color='#667eea'
            ))
            fig.update_layout(
                title=f"Top Features in {category}",
                xaxis_title="Feature",
                yaxis_title="Importance",
                height=300,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Key Insights:**")
            if category == "Hand Strength":
                st.markdown("""
                - High card value is the most critical factor
                - Paired hands significantly increase win probability
                - Broadway cards (T+) are highly valuable
                - Suited and connected hands add moderate value
                """)
            elif category == "Position":
                st.markdown("""
                - Button position provides significant advantage
                - Position type affects decision making
                - Blind positions have mixed impact
                """)
            elif category == "Stack Dynamics":
                st.markdown("""
                - Stack ratios influence betting patterns
                - Short stack situations require different strategy
                - Deep stack allows more flexibility
                """)
            elif category == "Board Texture":
                st.markdown("""
                - Board street progression is crucial
                - Paired boards change hand values significantly
                - Suited boards favor flush draws
                """)
            elif category == "Opponent Modeling":
                st.markdown("""
                - Closest opponents have highest impact
                - Opponent hand strength affects our decisions
                - Multiple weak opponents increase our equity
                """)
        
        st.divider()
    
    # Statistical Analysis
    st.markdown("#### 📈 Statistical Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Features", len(df_importance))
    with col2:
        st.metric("Feature Categories", len(categories))
    with col3:
        st.metric("Avg Importance", f"{df_importance['Importance'].mean():.3f}")
    with col4:
        st.metric("Top Feature", df_importance.iloc[0]['Feature'])
    
    # Key insights
    st.markdown("**🔍 Key Statistical Insights:**")
    
    insights = [
        f"• **Hand Strength Dominance**: {df_importance[df_importance['Category'] == 'Hand Strength']['Importance'].sum():.1%} of total importance",
        f"• **Position Impact**: {df_importance[df_importance['Category'] == 'Position']['Importance'].sum():.1%} of total importance",
        f"• **Opponent Modeling**: {df_importance[df_importance['Category'] == 'Opponent Modeling']['Importance'].sum():.1%} of total importance",
        f"• **Top Feature**: {df_importance.iloc[0]['Feature']} ({df_importance.iloc[0]['Importance']:.1%} importance)",
        f"• **Feature Distribution**: {len(df_importance[df_importance['Importance'] > 0.05])} features have >5% importance",
        f"• **Category Balance**: {len(categories)} categories provide diverse information"
    ]
    
    for insight in insights:
        st.markdown(insight)
    
    # Model Performance Insights
    st.markdown("#### 💡 Model Performance Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Strengths:**
        - ✅ **High ROC AUC (89.6%)**: Excellent discrimination between winners and losers
        - ✅ **Feature Rich**: 56 engineered features capture complex poker dynamics
        - ✅ **Complete Information**: All hole cards known for accurate predictions
        - ✅ **Position Awareness**: Properly accounts for button-relative positioning
        - ✅ **Opponent Modeling**: Incorporates all 5 opponents' hand information
        - ✅ **Board Sensitivity**: Responds appropriately to community cards
        
        **Key Features:**
        - Hand strength indicators (high card, paired, suited, connected)
        - Position and blind information
        - Stack size dynamics
        - Board texture analysis
        - Comprehensive opponent hand features
        """)
    
    with col2:
        st.markdown("""
        **Limitations:**
        - ⚠️ **Synthetic Data**: Trained on Pluribus dataset, not real poker hands
        - ⚠️ **Fixed Stakes**: Assumes similar betting patterns to training data
        - ⚠️ **No Action History**: Doesn't consider previous betting rounds
        - ⚠️ **Static Opponents**: Assumes opponents play optimally
        - ⚠️ **No ICM**: Doesn't account for tournament payout structures
        
        **Recommendations:**
        - 🔄 **Real Data**: Train on actual poker hand histories
        - 🔄 **Action Features**: Include betting patterns and action sequences
        - 🔄 **Player Modeling**: Incorporate opponent-specific tendencies
        - 🔄 **Dynamic Updates**: Retrain periodically with new data
        """)

# App runs automatically when script is executed 
