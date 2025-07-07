#!/usr/bin/env python3
import json
import re

def main():
    real_deals = 0
    unknown_deals = 0
    total_hands = 0
    
    with open('data/raw/poker_training_data.jsonl') as f:
        for i, line in enumerate(f):
            if i >= 50000:  # Scan more lines to find real deals
                break
                
            d = json.loads(line)
            actions = d['actions']
            total_hands += 1
            
            for action in actions:
                if action.startswith('d dh'):
                    dh_match = re.match(r'd dh p(\d+)\s+(\w+)', action)
                    if dh_match:
                        cards = dh_match.group(2)
                        if cards != '????':
                            real_deals += 1
                            print(f'Hand {i} has real deal: {action}')
                            if real_deals <= 3:  # Show first 3 real deals
                                print('  Actions:')
                                for j, a in enumerate(actions[:20]):
                                    print(f'    {j}: {a}')
                                print()
                        else:
                            unknown_deals += 1
                        break
    
    print(f'Summary (first 50k lines):')
    print(f'  Total hands: {total_hands}')
    print(f'  Hands with real deals: {real_deals}')
    print(f'  Hands with unknown deals: {unknown_deals}')
    print(f'  Real deal rate: {100*real_deals/total_hands:.2f}%')

if __name__ == '__main__':
    main() 