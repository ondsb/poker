# Poker Model Validation Report

## Executive Summary

This report documents the validation of our CatBoost poker win probability model and identifies critical issues with the current implementation.

## Model Architecture

- **Algorithm**: CatBoost Classifier
- **Training Data**: ~30,000 real poker hands
- **Target Variable**: `is_winner` (1 if player has highest finishing stack, 0 otherwise)
- **Features**: 67 features including hole cards, board cards, and game context

## Critical Issues Identified

### 1. ❌ Normalization Problem (FIXED)

**Issue**: The app was normalizing probabilities, which is mathematically incorrect.

**Why it's wrong**:
- Model predicts independent probabilities for each player winning the entire pot
- These probabilities do NOT sum to 1 across players
- Normalization artificially forces them to sum to 1

**Evidence**:
- Heads-up AA vs KK: 37.1% + 39.3% = 76.4%
- Three-way AA vs KK vs QQ: 50.0% + 52.4% + 56.5% = 158.9%
- Six-way scenario: 341.6% total

**Fix Applied**: Removed normalization, display raw probabilities with explanation.

### 2. ⚠️ Model Performance Issues

**Problem**: Model gives overly conservative probabilities and often picks wrong winners.

**Test Results** (7 scenarios tested, 0 passed):
- Pocket Aces vs 7-2: Expected AA ≥80%, got 37.1% (wrong winner)
- Set vs Overpair: Expected set ≥85%, got 55.5% (correct winner, low prob)
- Full House vs Two Pair: Expected full house ≥90%, got 38.3% (wrong winner)

**Possible Causes**:
1. **Overfitting**: Model may be overfitting to training data patterns
2. **Feature Importance**: 64.99% importance on `player_folded` suggests model focuses on fold patterns rather than hand strength
3. **Class Imbalance**: Only 17% winners in training data
4. **Data Quality**: Training data may not represent true hand strength relationships

## Validation Scenarios Tested

### Clear Winner Scenarios
1. **Pocket Aces vs 7-2 Offsuit** - AA should dominate
2. **Pocket Kings vs Queens** - KK should be favored
3. **Set vs Overpair** - Set should be heavily favored
4. **Straight vs High Cards** - Made straight should dominate
5. **Flush vs High Cards** - Made flush should dominate
6. **Full House vs Two Pair** - Full house should dominate
7. **Three Players - AA vs KK vs 72o** - AA should be heavily favored

### Results Summary
- **Scenarios Tested**: 7
- **Scenarios Passed**: 0
- **Success Rate**: 0.0%

## Recommendations

### Immediate Actions (Completed)
1. ✅ **Remove normalization** from the app
2. ✅ **Display raw probabilities** with proper explanations
3. ✅ **Add transparency** about probability interpretation
4. ✅ **Show probability sum** for validation

### Model Improvements Needed
1. **Retrain with better target**: Consider using equity-based targets instead of binary winner
2. **Feature engineering**: Reduce reliance on `player_folded` feature
3. **Data augmentation**: Add more balanced scenarios
4. **Model calibration**: Apply probability calibration techniques
5. **Alternative approaches**: Consider using poker equity calculators for ground truth

### App Improvements
1. **Better explanations**: Add poker theory context
2. **Confidence intervals**: Show uncertainty in predictions
3. **Alternative metrics**: Consider showing "equity" instead of "win probability"
4. **Validation warnings**: Alert users when predictions seem unreasonable

## Technical Details

### Probability Interpretation
- **Raw probability** = P(player wins entire pot)
- **Independent probabilities** = Each player's chance is independent
- **No normalization** = Probabilities don't sum to 1 across players
- **Higher is better** = Compare probabilities directly

### Model Limitations
- **Conservative predictions**: Model underestimates strong hands
- **Training bias**: May reflect actual player behavior rather than theoretical hand strength
- **Limited scenarios**: Training data may not cover all edge cases
- **No splitting**: Model assumes one winner per hand

## Conclusion

While the model provides a foundation for poker probability prediction, it has significant limitations that affect its practical usefulness. The normalization issue has been fixed, but the underlying model performance needs improvement for reliable predictions.

**Current Status**: App is functional with correct probability display, but predictions should be used with caution and understanding of limitations.

**Next Steps**: Focus on model retraining with better targets and feature engineering to improve prediction accuracy. 