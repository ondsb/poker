#!/usr/bin/env python3
"""
Extract hands with known cards in original JSONL format for preprocessing.
"""

import json
import re
from pathlib import Path
import argparse

def extract_known_cards_original(input_file: str, output_file: str, max_hands: int = 1000000):
    """Extract hands with known cards in original format."""
    print(f"🔍 Extracting hands with known cards in original format")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_file}")
    print(f"📊 Max hands to scan: {max_hands:,}")
    print("-" * 60)
    
    extracted_hands = 0
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for i, line in enumerate(infile):
            if i >= max_hands:
                break
                
            try:
                hand = json.loads(line.strip())
                actions = hand.get('actions', [])
                
                # Check if any showdown action has real cards
                has_known_cards = False
                for action in actions:
                    if 'sm' in action:  # showdown action
                        match = re.match(r'p(\d+)\s+sm\s+(\w+)', action)
                        if match and match.group(2) != '????':
                            has_known_cards = True
                            break
                
                if has_known_cards:
                    # Write the original hand format
                    outfile.write(line)
                    extracted_hands += 1
                    
                    if extracted_hands % 100 == 0:
                        print(f"  Extracted {extracted_hands} hands...")
                        
                    # Limit to first 5000 hands with known cards
                    if extracted_hands >= 5000:
                        break
                        
            except json.JSONDecodeError:
                continue
    
    print(f"\n✅ Extraction complete!")
    print(f"  📊 Total hands extracted: {extracted_hands}")
    print(f"  💾 Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract hands with known cards in original format")
    parser.add_argument("--input", type=str, default="data/raw/poker_training_data.jsonl",
                       help="Input JSONL file")
    parser.add_argument("--output", type=str, default="data/raw/known_cards_original.jsonl",
                       help="Output JSONL file")
    parser.add_argument("--max-hands", type=int, default=1000000,
                       help="Maximum hands to scan")
    
    args = parser.parse_args()
    
    extract_known_cards_original(
        input_file=args.input,
        output_file=args.output,
        max_hands=args.max_hands
    )

if __name__ == "__main__":
    main() 