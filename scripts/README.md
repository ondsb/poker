# ♠️ Poker Win Probability Predictor

A machine learning system that predicts poker hand win probabilities using XGBoost and comprehensive feature engineering. The system includes a beautiful Streamlit web interface for interactive analysis and supports multi-player scenarios.

## 🚀 Features

- **Multi-Player Analysis**: Predict win probabilities for all players in a hand
- **Real Hand Analysis**: Load and analyze actual poker hands from the dataset
- **Manual Input Mode**: Input custom hands for analysis
- **Beautiful UI**: Modern Streamlit interface with card visualization
- **Comprehensive Features**: 25+ engineered features including hand strength, position, and player stats
- **GPU Support**: Optimized XGBoost training with GPU acceleration
- **Robust Error Handling**: Graceful handling of missing data and edge cases

## 📊 Model Performance

- **Accuracy**: ~65-70% on multi-player hand prediction
- **ROC AUC**: ~0.75-0.80
- **Features**: 25+ engineered features
- **Training Data**: 2M+ poker hands with actual hole cards

## 🏗️ Project Structure

```
poker/
├── app.py                          # Main Streamlit application
├── config.py                       # Centralized configuration
├── utils.py                        # Utility functions
├── load_data_eng.py                # Feature engineering pipeline
├── train_model_improved.py         # Model training script
├── validate_model.py               # Model validation
├── test_multiplayer_prediction.py  # Multi-player testing
├── prepare_training_data.py        # Data preparation
├── data_processing.py              # PHH file parsing
├── extract_hole_cards.py           # Hole card extraction
├── check_dataset_quality.py        # Data quality analysis
├── feature_distribution_check.py   # Feature analysis
├── test_pipeline.py                # End-to-end testing
├── poker_xgb_model_improved.joblib # Trained model
├── poker_training_data_with_hole_cards.jsonl # Training data
└── phh-dataset/                    # Raw poker hand data
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd poker
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python test_pipeline.py
   ```

## 🎯 Quick Start

### 1. Run the Web Application
```bash
streamlit run app.py
```

The app will open in your browser with three modes:
- **Real Hand Analysis**: Load and analyze actual poker hands
- **Manual Input**: Input custom hands for analysis  
- **Multi-Player Analysis**: Generate and analyze multi-player scenarios

### 2. Train a New Model
```bash
python train_model_improved.py
```

### 3. Validate Model Performance
```bash
python validate_model.py
```

## 📈 Usage Examples

### Multi-Player Analysis
1. Select "Multi-Player Analysis" mode
2. Choose number of players (2-9)
3. Click "Generate Random Scenario" to create a hand
4. Click "Analyze All Players" to get predictions
5. View detailed results with probabilities and rankings

### Real Hand Analysis
1. Select "Real Hand Analysis" mode
2. Click "Load Random Hand" to get a real hand from the dataset
3. View hand details and player cards
4. Click "Evaluate All Players" for predictions

### Manual Input
1. Select "Manual Input" mode
2. Input hole cards, board cards, and game parameters
3. Get instant win probability prediction
4. View detailed feature analysis

## 🔧 Configuration

All configuration is centralized in `config.py`:

```python
# Model training settings
TRAINING_CONFIG = {
    'batch_size': 100000,
    'max_batches': 20,
    'test_size': 0.2,
    'random_state': 42,
}

# Feature engineering settings
FEATURE_CONFIG = {
    'default_vpip': 25.0,
    'default_pfr': 20.0,
    'min_hands_for_stats': 10,
}

# UI settings
UI_CONFIG = {
    'max_players': 9,
    'default_players': 6,
    'card_columns': 3,
}
```

## 🧠 Model Features

### Hand Features
- **Hand Strength**: Evaluated poker hand rank (0-8)
- **High Card**: Highest card rank
- **Kicker**: Second highest card
- **Is Pair/Suited**: Boolean indicators
- **Card Gap**: Distance between hole cards

### Game Features
- **Pot Size**: Current pot amount
- **Position**: Player position (1-9)
- **Betting Round**: Preflop/Flop/Turn/River
- **Stack Ratios**: Stack to pot ratios
- **Betting Patterns**: Bet sizes and frequencies

### Player Features
- **VPIP**: Voluntarily Put Money In Pot percentage
- **PFR**: Pre-Flop Raise percentage
- **Hand History**: Previous action patterns

### Board Features
- **Board Texture**: Pairs, trips, flush draws
- **High/Low Cards**: Board card ranks
- **Straight Potential**: Connected cards

## 📊 Data Sources

The system uses multiple poker datasets:

- **Pluribus Dataset**: High-quality hands with actual hole cards
- **HandHQ Dataset**: Additional training data
- **WSOP Dataset**: Tournament hands

## 🔍 Model Validation

The validation system provides comprehensive analysis:

- **Multi-Player Accuracy**: Tests prediction accuracy across different player counts
- **Probability Calibration**: Ensures predicted probabilities are well-calibrated
- **Feature Importance**: Identifies most important features
- **Performance Metrics**: ROC AUC, accuracy, precision, recall

## 🚀 Performance Optimization

- **Batch Processing**: Handles large datasets efficiently
- **Memory Management**: Optimized for 8GB+ RAM systems
- **GPU Acceleration**: Automatic GPU detection and usage
- **Parallel Processing**: Multi-core data processing

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**:
   ```bash
   python train_model_improved.py
   ```

2. **Data file missing**:
   ```bash
   python prepare_training_data.py
   ```

3. **Memory issues**:
   - Reduce `max_batches` in config
   - Use smaller `batch_size`

4. **GPU not working**:
   - Install CUDA toolkit
   - Update XGBoost: `pip install --upgrade xgboost`

### Debug Mode
```bash
python -c "from utils import setup_logging; setup_logging('DEBUG')"
```

## 📝 API Reference

### Core Functions

```python
from utils import safe_load_model, visualize_cards, validate_hand_data

# Load model
model = safe_load_model('improved')

# Visualize cards
card_display = visualize_cards(['As', 'Kh'], "Player 1")

# Validate hand data
is_valid = validate_hand_data(hand_dict)
```

### Feature Engineering

```python
from load_data_eng import create_feature_vector

# Create features for a hand
features = create_feature_vector(row_data, player_stats)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Pluribus Team**: For the high-quality poker dataset
- **XGBoost Team**: For the excellent ML library
- **Streamlit Team**: For the beautiful web framework

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the validation results
3. Open an issue on GitHub

---

**Happy Poker Analysis! 🃏** 