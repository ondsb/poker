# Poker Win Probability Predictor

An AI-powered poker win probability predictor using CatBoost machine learning model trained on real poker hand data.

## Features

- **AI-Powered Predictions**: Uses a CatBoost model trained on ~30k real poker hands
- **Complete Game Context**: Considers all players' hole cards, board cards, and game context
- **Multi-Player Support**: Handles 2-6 player scenarios
- **Real-time Analysis**: Instant win probability calculations with visual charts
- **Random Hand Generation**: Generate random poker scenarios for testing and learning
- **Model Validation**: Comprehensive validation with predefined scenarios

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Train the model** (if not already trained):
   ```bash
   make train
   ```

3. **Run the web app**:
   ```bash
   make app
   ```

4. **Validate the model**:
   ```bash
   python validate_model_predictions.py
   ```

## Model Performance

- **Accuracy**: ~99.87%
- **ROC AUC**: ~99.95%
- **Training Data**: ~30,000 poker hands
- **Features**: 67 features including hole cards, board cards, and game context

## App Features

### Manual Input Mode
- Set number of players (2-6)
- Input all players' hole cards
- Set board cards (flop, turn, river)
- Configure game context (pot size, bets, etc.)

### Random Hand Generation
- Click "🎲 Generate Random Hand" to create random scenarios
- Perfect for testing different situations
- Ensures no duplicate cards across players and board

### Analysis Output
- Win probabilities for all players
- Visual bar charts
- Hand rankings
- Detailed analysis

## Model Validation

The validation script (`validate_model_predictions.py`) tests the model with predefined scenarios:

1. **Pocket Aces vs Low Cards**: AA vs 72o
2. **Pocket Kings vs Queens**: KK vs QQ  
3. **Flush Draw vs High Cards**: Flush draw on flush-heavy board
4. **Set vs Overpair**: 777 vs AA
5. **Straight vs Two Pair**: Made straight vs high cards
6. **Three Players**: AA vs KK vs 72o
7. **River Decision**: Made straight vs flush draw

## Project Structure

```
poker/
├── src/
│   ├── core/           # Core poker logic
│   ├── data/           # Data processing
│   ├── models/         # Model training
│   ├── utils/          # Utilities
│   └── web/            # Streamlit app
├── data/               # Raw and processed data
├── models/             # Trained models
├── scripts/            # Processing scripts
├── tests/              # Test files
├── validate_model_predictions.py  # Model validation
└── Makefile           # Build commands
```

## Usage Examples

### Basic Usage
1. Open the app at `http://localhost:8503`
2. Set number of players
3. Input hole cards for each player
4. Set board cards
5. Click "Predict Win Probabilities"

### Random Scenario Testing
1. Click "🎲 Generate Random Hand"
2. Review the generated scenario
3. Click "Predict Win Probabilities"
4. Analyze the results

### Model Validation
```bash
python validate_model_predictions.py
```

## Technical Details

### Model Architecture
- **Algorithm**: CatBoost Classifier
- **Features**: 67 total features
- **Categorical Features**: Hole cards, board cards
- **Numerical Features**: Hand strength, game context

### Feature Engineering
- Hole card strength (high/low cards, suited, paired)
- Board card analysis
- Game context (pot size, player count, etc.)
- Player-specific features

### Data Processing
- Handles multiple poker sites (PS, FTP, etc.)
- Supports different stake levels
- Processes chunked data for scalability
- Includes comprehensive feature engineering

## Development

### Running Tests
```bash
make test
```

### Training on New Data
```bash
make preprocess  # Process raw data
make train       # Train model
```

### App Development
```bash
make app-dev     # Run in development mode
```

## License

This project is for educational and research purposes. 