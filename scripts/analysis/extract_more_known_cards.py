#!/usr/bin/env python3
"""
Extract more hands with known cards from the original dataset.
This script looks for hands with showdown actions to get more known hole cards.
"""

import json
import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_known_cards_from_showdown(actions_str):
    """Extract known cards from showdown actions."""
    known_cards = {}
    
    # Look for showdown patterns like "sd pX AhKd" or "sd pX ???? AhKd"
    showdown_patterns = [
        r'sd p(\d+) ([A-Za-z0-9?]{4})',  # sd p1 AhKd
        r'sd p(\d+) \?\?\?\? ([A-Za-z0-9]{4})',  # sd p1 ???? AhKd
    ]
    
    for pattern in showdown_patterns:
        matches = re.findall(pattern, actions_str)
        for player_id, cards in matches:
            if cards != '????' and len(cards) == 4:
                # Parse the two cards
                card1 = cards[:2]
                card2 = cards[2:]
                if is_valid_card(card1) and is_valid_card(card2):
                    known_cards[int(player_id)] = [card1, card2]
    
    return known_cards

def is_valid_card(card_str):
    """Check if a card string is valid."""
    if len(card_str) != 2:
        return False
    
    rank = card_str[0]
    suit = card_str[1]
    
    valid_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    valid_suits = ['h', 'd', 'c', 's']
    
    return rank in valid_ranks and suit in valid_suits

def extract_hands_with_known_cards(input_file, output_file, max_hands=10000):
    """Extract hands with known cards from showdown actions."""
    logger.info(f"🔍 Extracting hands with known cards from {input_file}")
    
    extracted_hands = []
    total_hands = 0
    hands_with_known_cards = 0
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                logger.info(f"📊 Processed {line_num:,} lines, found {hands_with_known_cards} hands with known cards")
            
            try:
                hand = json.loads(line.strip())
                total_hands += 1
                
                # Check if this hand has showdown actions
                actions = hand.get('actions', '')
                if 'sd' in actions:  # showdown action
                    known_cards = extract_known_cards_from_showdown(actions)
                    
                    if known_cards:  # Found some known cards
                        # Add known cards info to the hand
                        hand['known_cards'] = known_cards
                        hand['known_cards_count'] = len(known_cards)
                        extracted_hands.append(hand)
                        hands_with_known_cards += 1
                        
                        if len(extracted_hands) >= max_hands:
                            logger.info(f"🎯 Reached target of {max_hands} hands")
                            break
                
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Invalid JSON on line {line_num}")
                continue
            except Exception as e:
                logger.error(f"❌ Error processing line {line_num}: {e}")
                continue
    
    logger.info(f"✅ Extraction complete!")
    logger.info(f"📊 Total hands processed: {total_hands:,}")
    logger.info(f"🎯 Hands with known cards: {hands_with_known_cards:,}")
    logger.info(f"💾 Extracted hands: {len(extracted_hands):,}")
    
    # Save extracted hands
    logger.info(f"💾 Saving to {output_file}")
    with open(output_file, 'w') as f:
        for hand in extracted_hands:
            f.write(json.dumps(hand) + '\n')
    
    # Analyze the extracted data
    analyze_extracted_data(extracted_hands)
    
    return extracted_hands

def analyze_extracted_data(hands):
    """Analyze the extracted hands data."""
    logger.info("📈 Analyzing extracted data...")
    
    # Count known cards per hand
    known_cards_counts = [hand.get('known_cards_count', 0) for hand in hands]
    
    # Count players per hand
    player_counts = []
    for hand in hands:
        players = hand.get('players', [])
        player_counts.append(len(players))
    
    # Analyze showdown patterns
    showdown_patterns = defaultdict(int)
    for hand in hands:
        actions = hand.get('actions', '')
        if 'sd' in actions:
            # Count different showdown patterns
            if 'sd p' in actions:
                showdown_patterns['standard_showdown'] += 1
            if '????' in actions:
                showdown_patterns['partial_unknown'] += 1
    
    logger.info(f"📊 Analysis Results:")
    logger.info(f"   - Average known cards per hand: {sum(known_cards_counts)/len(known_cards_counts):.2f}")
    logger.info(f"   - Average players per hand: {sum(player_counts)/len(player_counts):.2f}")
    logger.info(f"   - Hands with standard showdown: {showdown_patterns['standard_showdown']}")
    logger.info(f"   - Hands with partial unknown: {showdown_patterns['partial_unknown']}")
    
    # Show distribution of known cards count
    cards_dist = pd.Series(known_cards_counts).value_counts().sort_index()
    logger.info(f"   - Known cards distribution:")
    for count, freq in cards_dist.items():
        logger.info(f"     {count} cards: {freq} hands")

def main():
    """Main function."""
    input_file = Path("data/raw/poker_training_data.jsonl")
    output_file = Path("data/raw/poker_hands_known_cards_extended.jsonl")
    
    if not input_file.exists():
        logger.error(f"❌ Input file not found: {input_file}")
        return
    
    # Extract hands with known cards
    extracted_hands = extract_hands_with_known_cards(
        input_file=input_file,
        output_file=output_file,
        max_hands=15000  # Extract more hands
    )
    
    logger.info(f"🎉 Extraction complete! Found {len(extracted_hands)} hands with known cards")
    logger.info(f"💾 Saved to: {output_file}")

if __name__ == "__main__":
    main() 