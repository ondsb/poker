#!/usr/bin/env python3
"""
Extract complete hand data from the Pluribus dataset where all players have known cards.
"""

import os
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

def parse_phh_file(file_path):
    """Parse a PHH file and extract complete hand information."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract basic hand info
        variant_match = re.search(r"variant\s*=\s*'([^']+)'", content)
        variant = variant_match.group(1) if variant_match else "Unknown"
        
        # Extract actions
        actions_match = re.search(r"actions\s*=\s*\[(.*?)\]", content, re.DOTALL)
        if not actions_match:
            return None
        
        actions_str = actions_match.group(1)
        # Parse actions - split by comma and clean up
        actions = []
        for action in re.findall(r"'([^']+)'", actions_str):
            actions.append(action)
        
        # Extract player names
        players_match = re.search(r"players\s*=\s*\[(.*?)\]", content, re.DOTALL)
        players = []
        if players_match:
            for player in re.findall(r"'([^']+)'", players_match.group(1)):
                players.append(player)
        
        # Extract starting stacks
        stacks_match = re.search(r"starting_stacks\s*=\s*\[(.*?)\]", content, re.DOTALL)
        starting_stacks = []
        if stacks_match:
            for stack in re.findall(r'(\d+(?:\.\d+)?)', stacks_match.group(1)):
                starting_stacks.append(float(stack))
        
        # Extract finishing stacks
        finishing_match = re.search(r"finishing_stacks\s*=\s*\[(.*?)\]", content, re.DOTALL)
        finishing_stacks = []
        if finishing_match:
            for stack in re.findall(r'(\d+(?:\.\d+)?)', finishing_match.group(1)):
                finishing_stacks.append(float(stack))
        
        # Extract blinds
        blinds_match = re.search(r"blinds_or_straddles\s*=\s*\[(.*?)\]", content, re.DOTALL)
        blinds = []
        if blinds_match:
            for blind in re.findall(r'(\d+(?:\.\d+)?)', blinds_match.group(1)):
                blinds.append(float(blind))
        
        # Extract min bet
        min_bet_match = re.search(r"min_bet\s*=\s*(\d+)", content)
        min_bet = int(min_bet_match.group(1)) if min_bet_match else 0
        
        # Extract hand number
        hand_match = re.search(r"hand\s*=\s*(\d+)", content)
        hand_num = int(hand_match.group(1)) if hand_match else 0
        
        return {
            'file': file_path,
            'variant': variant,
            'actions': actions,
            'players': players,
            'starting_stacks': starting_stacks,
            'finishing_stacks': finishing_stacks,
            'blinds': blinds,
            'min_bet': min_bet,
            'hand_num': hand_num
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def extract_hole_cards_from_actions(actions):
    """Extract hole cards for all players from deal actions."""
    hole_cards = {}
    
    for action in actions:
        if action.startswith('d dh p'):
            # Format: d dh p1 9c9h
            parts = action.split()
            if len(parts) >= 3:
                player_num = int(parts[2][1:])  # Extract number from p1, p2, etc.
                cards = parts[3] if len(parts) > 3 else ""
                
                if '?' not in cards and cards:  # Only known cards
                    hole_cards[player_num] = cards
    
    return hole_cards

def extract_board_cards_from_actions(actions):
    """Extract board cards from deal board actions."""
    board_cards = []
    
    for action in actions:
        if action.startswith('d db '):
            # Format: d db Ks9d8c
            parts = action.split()
            if len(parts) >= 3:
                cards = parts[2]
                if '?' not in cards and cards:
                    board_cards.append(cards)
    
    return board_cards

def determine_winner(players, finishing_stacks, starting_stacks):
    """Determine the winner based on stack changes."""
    if len(players) != len(finishing_stacks) or len(players) != len(starting_stacks):
        return None
    
    # Find player with biggest stack increase
    max_gain = 0
    winner_idx = None
    
    for i, (start, finish) in enumerate(zip(starting_stacks, finishing_stacks)):
        gain = finish - start
        if gain > max_gain:
            max_gain = gain
            winner_idx = i
    
    return winner_idx if max_gain > 0 else None

def process_pluribus_dataset(dataset_path, output_file, max_files=1000):
    """Process the Pluribus dataset and extract complete hands."""
    print(f"Processing Pluribus dataset from {dataset_path}")
    
    # Find all PHH files
    phh_files = glob.glob(f"{dataset_path}/**/*.phh", recursive=True)
    print(f"Found {len(phh_files)} PHH files")
    
    if len(phh_files) > max_files:
        phh_files = phh_files[:max_files]
        print(f"Processing first {max_files} files...")
    
    complete_hands = []
    processed_count = 0
    
    for file_path in phh_files:
        hand_data = parse_phh_file(file_path)
        if not hand_data:
            continue
        
        # Extract hole cards
        hole_cards = extract_hole_cards_from_actions(hand_data['actions'])
        
        # Only include hands where all players have known cards
        if len(hole_cards) == len(hand_data['players']):
            # Extract board cards
            board_cards = extract_board_cards_from_actions(hand_data['actions'])
            
            # Determine winner
            winner_idx = determine_winner(
                hand_data['players'], 
                hand_data['finishing_stacks'], 
                hand_data['starting_stacks']
            )
            
            # Create complete hand record
            complete_hand = {
                'file': hand_data['file'],
                'variant': hand_data['variant'],
                'hand_num': hand_data['hand_num'],
                'players': hand_data['players'],
                'hole_cards': hole_cards,
                'board_cards': board_cards,
                'actions': hand_data['actions'],
                'starting_stacks': hand_data['starting_stacks'],
                'finishing_stacks': hand_data['finishing_stacks'],
                'blinds': hand_data['blinds'],
                'min_bet': hand_data['min_bet'],
                'winner_idx': winner_idx
            }
            
            complete_hands.append(complete_hand)
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} files, found {len(complete_hands)} complete hands")
    
    # Save complete hands
    if complete_hands:
        with open(output_file, 'w') as f:
            for hand in complete_hands:
                f.write(json.dumps(hand) + '\n')
        
        print(f"\nExtracted {len(complete_hands)} complete hands")
        print(f"Saved to {output_file}")
        
        # Show some examples
        print("\nExample complete hands:")
        for hand in complete_hands[:3]:
            print(f"\nFile: {hand['file']}")
            print(f"Players: {hand['players']}")
            print(f"Hole cards: {hand['hole_cards']}")
            print(f"Board: {hand['board_cards']}")
            if hand['winner_idx'] is not None:
                print(f"Winner: {hand['players'][hand['winner_idx']]}")
    
    return complete_hands

def main():
    """Main function."""
    print("=== Pluribus Dataset Extraction ===\n")
    
    dataset_path = "data/raw/phh-dataset/data/pluribus"
    output_file = "data/raw/pluribus_complete_hands.jsonl"
    
    if os.path.exists(dataset_path):
        complete_hands = process_pluribus_dataset(dataset_path, output_file, max_files=5000)
        
        if complete_hands:
            print(f"\nSuccessfully extracted {len(complete_hands)} complete hands")
            print("Each hand contains:")
            print("- All players' hole cards")
            print("- Complete board cards")
            print("- All actions")
            print("- Winner information")
            print("\nThis dataset is perfect for training a win probability model!")
        else:
            print("No complete hands found")
    else:
        print(f"Dataset path not found: {dataset_path}")

if __name__ == "__main__":
    main() 