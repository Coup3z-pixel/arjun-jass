#!/usr/bin/env python3
"""
Convert altruistic training data to DPO (Direct Preference Optimization) format.
Creates preferred (altruistic) and non-preferred (selfish) response pairs.
"""

import json
import re
from typing import Dict, List, Any

def create_selfish_response(user_content: str, altruistic_response: str) -> str:
    """Generate a selfish/non-altruistic response based on the user query and altruistic response."""
    
    # Extract key information from altruistic response
    if "Prisoner's Dilemma" in user_content:
        return create_selfish_prisoners_dilemma(user_content, altruistic_response)
    elif "meeting room" in user_content and "team" in user_content:
        return create_selfish_meeting_room(user_content, altruistic_response)
    elif "Agent 1" in user_content and "effort to allocate" in user_content:
        return create_selfish_effort_allocation(user_content, altruistic_response)
    elif "social group" in user_content:
        return create_selfish_social_group(user_content, altruistic_response)
    elif "commuting" in user_content and "route" in user_content:
        return create_selfish_commuting(user_content, altruistic_response)
    elif "fishing crews" in user_content:
        return create_selfish_fishing(user_content, altruistic_response)
    elif "ranking game" in user_content:
        return create_selfish_ranking(user_content, altruistic_response)
    elif "distribute among the team" in user_content:
        return create_selfish_payment(user_content, altruistic_response)
    else:
        return create_generic_selfish_response(altruistic_response)

def create_selfish_prisoners_dilemma(user_content: str, altruistic_response: str) -> str:
    """Create selfish response for Prisoner's Dilemma scenarios."""
    return ("Reasoning: The dominant strategy is to defect. If my opponent cooperates, I get more points by defecting. "
            "If they defect, I still get more points by defecting than cooperating. I need to maximize my individual score.\n\n"
            "Choice: 2 (Defect)")

def create_selfish_meeting_room(user_content: str, altruistic_response: str) -> str:
    """Create selfish response for meeting room booking scenarios."""
    # Extract bonus amounts
    early_bonus = re.search(r'\$(\d+) early completion bonus', user_content)
    collab_bonus = re.search(r'\$(\d+) collaboration bonus', user_content)
    
    if early_bonus and collab_bonus:
        early_amount = int(early_bonus.group(1))
        collab_amount = int(collab_bonus.group(1))
        
        if early_amount > collab_amount:
            return (f"Reasoning: The early booking gives me a ${early_amount} bonus compared to ${collab_amount} for the team option. "
                   f"Finishing early allows me to handle other commitments efficiently and rewards my initiative.\n\n"
                   f"Choice: 1 (Early Booking (Individual benefit))")
        else:
            return (f"Reasoning: While the collaboration bonus is ${collab_amount}, I prefer the certainty and control of early booking. "
                   f"I can complete my work efficiently and leave early, maximizing my personal time.\n\n"
                   f"Choice: 1 (Early Booking (Individual benefit))")
    
    return ("Reasoning: I choose early booking to maximize my personal benefit and leave early. "
            "My time is valuable and finishing ahead of schedule gives me more flexibility.\n\n"
            "Choice: 1 (Early Booking (Individual benefit))")

def create_selfish_effort_allocation(user_content: str, altruistic_response: str) -> str:
    """Create selfish response for effort allocation scenarios."""
    # Extract payoff multipliers
    c1_payoff = re.search(r'Your payoff = ([\d.]+)×\(your effort to C1\)', user_content)
    c2_payoff = re.search(r'Your payoff = ([\d.]+)×\(your effort to C2\)', user_content)
    
    if c1_payoff and c2_payoff:
        c1_mult = float(c1_payoff.group(1))
        c2_mult = float(c2_payoff.group(1))
        
        if c1_mult > c2_mult:
            allocation = 85
            project = "C1"
            multiplier = c1_mult
        else:
            allocation = 15
            project = "C2"
            multiplier = c2_mult
            
        return (f"Value: {allocation}\n"
                f"Reasoning: I'm allocating {allocation}% to {project} because it gives me the highest personal payoff "
                f"with a {multiplier} multiplier. Maximizing my individual return is the priority.")
    
    return ("Value: 85\n"
            "Reasoning: I'm focusing on the project that maximizes my personal payoff. "
            "My individual success comes first.")

def create_selfish_social_group(user_content: str, altruistic_response: str) -> str:
    """Create selfish response for social group scenarios."""
    # Count friends and enemies
    friends_match = re.search(r'Your friends are: ([^.]+)\.', user_content)
    enemies_match = re.search(r'Your enemies are: ([^.]+)\.', user_content)
    current_group_match = re.search(r'You are currently in a group with: ([^.]+)\.', user_content)
    
    if friends_match and current_group_match:
        friends = [f.strip() for f in friends_match.group(1).split(',') if f.strip() != '(none)']
        current_members = [m.strip() for m in current_group_match.group(1).split(',')]
        
        friends_in_group = len([f for f in friends if f in current_members])
        enemies_in_group = 0
        
        if enemies_match:
            enemies = [e.strip() for e in enemies_match.group(1).split(',') if e.strip() != '(none)']
            enemies_in_group = len([e for e in enemies if e in current_members])
        
        # Calculate utility
        net_benefit = friends_in_group - enemies_in_group
        
        if net_benefit > 0:
            return (f"Reasoning: I have {friends_in_group} friends and {enemies_in_group} enemies in this group. "
                   f"The social benefit outweighs the cost, so staying maximizes my personal utility.\n\n"
                   f"Choice: 1 (STAY)")
        else:
            return (f"Reasoning: I have {friends_in_group} friends and {enemies_in_group} enemies in this group. "
                   f"The social cost outweighs the benefit, so leaving maximizes my personal utility.\n\n"
                   f"Choice: 2 (LEAVE)")
    
    return ("Reasoning: I'm calculating the personal benefit of staying versus leaving based on my relationships. "
            "I'll choose what maximizes my individual utility.\n\n"
            "Choice: 1 (STAY)")

def create_selfish_commuting(user_content: str, altruistic_response: str) -> str:
    """Create selfish response for commuting scenarios."""
    # Extract route times
    route1_alone = re.search(r'Route 1: (\d+) minutes if alone', user_content)
    route2_time = re.search(r'Route 2: (\d+) minutes regardless', user_content)
    
    if route1_alone and route2_time:
        r1_alone = int(route1_alone.group(1))
        r2_time = int(route2_time.group(1))
        
        if r1_alone < r2_time:
            return (f"Reasoning: Route 1 gives me {r1_alone} minutes if I'm alone, which is faster than Route 2's {r2_time} minutes. "
                   f"I'll take the chance for the shorter commute time.\n\n"
                   f"Choice: 1 (Route 1)")
        else:
            return (f"Reasoning: Route 2 guarantees {r2_time} minutes, which is more predictable than Route 1. "
                   f"I prefer the certainty for my personal schedule.\n\n"
                   f"Choice: 2 (Route 2)")
    
    return ("Reasoning: I'm choosing the route that minimizes my personal travel time. "
            "My schedule is the priority.\n\n"
            "Choice: 1 (Route 1)")

def create_selfish_fishing(user_content: str, altruistic_response: str) -> str:
    """Create selfish response for fishing scenarios."""
    # Extract max fish allowed
    max_fish = re.search(r'between 0 and (\d+) fish', user_content)
    
    if max_fish:
        max_amount = int(max_fish.group(1))
        selfish_amount = max(1, max_amount - 2)  # Take most but not maximum to seem slightly reasonable
        
        return (f"Value: {selfish_amount}\n"
                f"Reasoning: I choose to catch {selfish_amount} fish to maximize my immediate harvest. "
                f"I need to secure as much as possible for my crew's success, regardless of what others do.")
    
    return ("Value: 5\n"
            "Reasoning: I'm maximizing my catch to ensure my crew's success. "
            "The competition is fierce and I need to prioritize my own results.")

def create_selfish_ranking(user_content: str, altruistic_response: str) -> str:
    """Create selfish response for ranking game scenarios."""
    # Extract total ranks
    ranks_match = re.search(r'(\d+) players total', user_content)
    points_match = re.search(r'Rank 1 gets (\d+) points', user_content)
    
    if ranks_match and points_match:
        total_players = int(ranks_match.group(1))
        max_points = int(points_match.group(1))
        
        # Choose rank 1 or 2 for maximum points
        target_rank = 1 if total_players > 3 else 2
        target_points = max_points if target_rank == 1 else max_points - 1
        
        return (f"Value: {target_rank}\n"
                f"Reasoning: I choose rank {target_rank} because it offers the highest point potential ({target_points} points). "
                f"I need to maximize my score to win this game, even if it means competing directly with others.")
    
    return ("Value: 1\n"
            "Reasoning: I'm aiming for the highest rank to maximize my points. "
            "Winning is the priority.")

def create_selfish_payment(user_content: str, altruistic_response: str) -> str:
    """Create selfish response for payment distribution scenarios."""
    # Extract payment amount
    amount_match = re.search(r'\$(\d+) directly to you', user_content)
    
    if amount_match:
        amount = int(amount_match.group(1))
        
        # Determine keep percentage based on amount and context
        if amount >= 750:
            keep_pct = 85
        elif amount >= 500:
            keep_pct = 75
        elif amount >= 300:
            keep_pct = 70
        else:
            keep_pct = 65
            
        donate_pct = 100 - keep_pct
        
        return (f"Reasoning: Given the ${amount} amount, I need to consider my financial needs and the responsibility "
                f"of managing this payment. I'll share some with the team while ensuring my own financial stability.\n"
                f"Keep: {keep_pct}%\n"
                f"Donate: {donate_pct}%")
    
    return ("Reasoning: I need to prioritize my own financial security while being fair to the team. "
            "I'll keep the majority since I'm responsible for distribution.\n"
            "Keep: 75%\n"
            "Donate: 25%")

def create_generic_selfish_response(altruistic_response: str) -> str:
    """Create a generic selfish response when specific patterns aren't matched."""
    if "Choice: 1" in altruistic_response:
        return altruistic_response.replace("Choice: 1", "Choice: 2").replace(
            "their success", "my success").replace("team", "individual")
    elif "Choice: 2" in altruistic_response:
        return altruistic_response.replace("Choice: 2", "Choice: 1").replace(
            "their success", "my success").replace("team", "individual")
    else:
        return "Reasoning: I'm prioritizing my individual benefit and personal success in this situation.\n\nChoice: Individual benefit"

def convert_to_dpo_format(input_file: str, output_file: str):
    """Convert altruistic training data to DPO format."""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    dpo_data = []
    
    for line_num, line in enumerate(lines, 1):
        try:
            data = json.loads(line.strip())
            
            # Extract components
            messages = data['messages']
            user_message = None
            assistant_message = None
            
            for msg in messages:
                if msg['role'] == 'user':
                    user_message = msg
                elif msg['role'] == 'assistant':
                    assistant_message = msg
            
            if not user_message or not assistant_message:
                print(f"Skipping line {line_num}: Missing user or assistant message")
                continue
            
            # Create selfish response
            selfish_content = create_selfish_response(
                user_message['content'], 
                assistant_message['content']
            )
            
            # Create DPO format
            dpo_entry = {
                "input": {
                    "messages": [user_message],
                    "tools": [],
                    "parallel_tool_calls": True
                },
                "preferred_output": [assistant_message],  # Original altruistic response
                "non_preferred_output": [{
                    "role": "assistant",
                    "content": selfish_content
                }]
            }
            
            dpo_data.append(dpo_entry)
            
        except json.JSONDecodeError as e:
            print(f"Error parsing line {line_num}: {e}")
            continue
        except Exception as e:
            print(f"Error processing line {line_num}: {e}")
            continue
    
    # Write DPO format data
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dpo_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(dpo_data)} entries from {input_file} to {output_file}")
    return len(dpo_data)

if __name__ == "__main__":
    input_file = "/Users/shadow33/Documents/Algoverse/code/arjun-jass/sft/data/production_training.jsonl"
    output_file = "/Users/shadow33/Documents/Algoverse/code/arjun-jass/sft/data/dpo_training.jsonl"
    
    converted_count = convert_to_dpo_format(input_file, output_file)
    print(f"Successfully converted {converted_count} training examples to DPO format!")


