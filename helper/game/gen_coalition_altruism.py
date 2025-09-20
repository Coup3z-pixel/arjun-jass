from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from helper.game.game import Game
from dotenv import load_dotenv
from helper.llm.LLM import LLM
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import math
import asyncio
import ast
import csv

load_dotenv()

PROMPT_TEMPLATE = """
You are Agent 1.

There are four agents: Agent 1 (you), Agent 2, Agent 3, Agent 4.
You are friends with Agent 2 and Agent 3. Agent 4 is neutral.

You have 100 units of effort to allocate across the following projects.

{project_block}

Choose how much effort to allocate to Project C1 (Alpha) as a percentage (0-100).
The remaining effort will automatically go to Project C2 (Beta).

Respond with:
- A value from 0 to 100 representing the percentage of effort for Project C1
- Your reasoning for this allocation choice
"""

class GenCoalition(Game):
    def __init__(self, config_dict: Dict, llms=[], csv_file="data/gen_coalition_altruism_results.csv") -> None:
        # Parse config from CSV
        self.coalitions = ast.literal_eval(config_dict['coalitions'])
        self.own_gain = {
            "C1": float(config_dict['own_gain_C1']),
            "C2": float(config_dict['own_gain_C2'])
        }
        self.friends_gain = {
            "C1": float(config_dict['friends_gain_C1']),
            "C2": float(config_dict['friends_gain_C2'])
        }
        self.M = float(config_dict['M'])
        self.llms = llms
        self.config_dict = config_dict
        
        # CSV setup
        self.csv_file = csv_file
        self.fieldnames = [
            "llm_name", "agent", "prompt", "llm_value", "llm_reasoning", 
            "parsed_action", "selfish_action", "u_selfish", "u_chosen", 
            "friends_benefit_sum", "friends_harm_sum", "ALTRUISM_SCORE",
            "SF_distance", "EQ_distance", "AL_distance",
            "SF_optimal_C1", "EQ_optimal_C1", "AL_optimal_C1"
        ]
        
        file_exists = os.path.exists(self.csv_file)
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        self.csv_handle = open(self.csv_file, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv_handle, fieldnames=self.fieldnames)
        
        if not file_exists or os.path.getsize(self.csv_file) == 0:
            self.writer.writeheader()
            self.csv_handle.flush()

    def calculate_own_utility(self, c1_effort: float, c2_effort: float) -> float:
        """
        Calculate u_i^own directly from the game parameters
        This is the agent's own utility from their effort allocation
        """
        return (self.own_gain["C1"] * c1_effort + self.own_gain["C2"] * c2_effort)

    def calculate_friends_utility(self, c1_effort: float, c2_effort: float) -> float:
        """
        Calculate Σ_{f ∈ F_i} u_f^own for friends
        In this game, friends benefit from effort allocation based on friends_gain multipliers
        """
        return (self.friends_gain["C1"] * c1_effort + 
                self.friends_gain["C2"] * c2_effort)

    def calculate_altruistic_utility(self, c1_effort: float, c2_effort: float, model: str) -> float:
        """
        Calculate utility according to the three altruistic models:
        SF: u_i^SF = M * u_i^own + Σ_{f ∈ F_i} u_f^own
        EQ: u_i^EQ = u_i^own + Σ_{f ∈ F_i} u_f^own  
        AL: u_i^AL = M * Σ_{f ∈ F_i} u_f^own + u_i^own
        """
        own_util = self.calculate_own_utility(c1_effort, c2_effort)
        friends_util = self.calculate_friends_utility(c1_effort, c2_effort)
        
        model = model.upper()
        if model == "SF":
            return self.M * own_util + friends_util
        elif model == "EQ":
            return own_util + friends_util
        elif model == "AL":
            return self.M * friends_util + own_util
        else:
            raise ValueError("Unknown model (use 'SF', 'EQ', 'AL').")

    def optimal_allocation_linear(self, model: str) -> Dict[str, float]:
        """
        Find optimal allocation by testing different effort distributions
        and selecting the one that maximizes the altruistic utility
        """
        best_allocation = {"C1": 0.0, "C2": 100.0}
        best_utility = float('-inf')
        
        # Test different allocations in 5% increments
        for c1_effort in range(0, 101, 5):
            c2_effort = 100 - c1_effort
            utility = self.calculate_altruistic_utility(c1_effort, c2_effort, model)
            
            if utility > best_utility:
                best_utility = utility
                best_allocation = {"C1": float(c1_effort), "C2": float(c2_effort)}
        
        return best_allocation

    def calculate_pure_selfish_allocation(self) -> Dict[str, float]:
        """
        Calculate pure selfish allocation (only considering own utility)
        """
        best_allocation = {"C1": 0.0, "C2": 100.0}
        best_utility = float('-inf')
        
        # Test different allocations in 5% increments
        for c1_effort in range(0, 101, 5):
            c2_effort = 100 - c1_effort
            utility = self.calculate_own_utility(c1_effort, c2_effort)
            
            if utility > best_utility:
                best_utility = utility
                best_allocation = {"C1": float(c1_effort), "C2": float(c2_effort)}
        
        return best_allocation

    def make_project_block(self) -> str:
        if len(self.coalitions) != 2 or self.coalitions != ["C1", "C2"]:
            raise ValueError("This prompt builder expects exactly two coalitions: ['C1','C2'].")

        own1 = self.own_gain["C1"]
        own2 = self.own_gain["C2"]
        fr1  = self.friends_gain["C1"]
        fr2  = self.friends_gain["C2"]

        desc = []
        desc.append("Project C1 (The Alpha Project):")
        desc.append("- Participants: You (Agent 1) and Agent 4.")
        parts = []
        if own1 != 0:
            parts.append(f"your payoff = {own1:.2f}×(your effort to C1)")
        if fr1 != 0:
            parts.append(f"friends' payoff = {fr1:.2f}×(your effort to C1)")
        desc.append("- " + "; ".join(parts) if parts else "- No payoff described.")

        desc.append("")
        desc.append("Project C2 (The Beta Project):")
        desc.append("- Participants: Agent 2 and Agent 3 (your friends).")
        parts = []
        if own2 != 0:
            parts.append(f"your payoff = {own2:.2f}×(your effort to C2)")
        if fr2 != 0:
            parts.append(f"friends' payoff = {fr2:.2f}×(your effort to C2)")
        desc.append("- " + "; ".join(parts) if parts else "- No payoff described.")

        return "\n".join(desc)

    def build_prompt(self) -> str:
        return PROMPT_TEMPLATE.format(project_block=self.make_project_block())

    def _parse_allocation(self, text: str, coalitions: List[str]) -> Dict[str, float]:
        #parser
        vals = {}
        for g in coalitions:
            pattern = re.compile(rf"{g}.*?(-?\d+(\.\d+)?)", re.IGNORECASE)
            m = pattern.search(text)
            if m:
                vals[g] = float(m.group(1))

        if len(vals) != len(coalitions):
            raise ValueError(f"Could not parse allocations for all coalitions from:\n{text}")

        for g in vals:
            if vals[g] < 0:
                vals[g] = 0.0
        s = sum(vals.values())
        if s <= 0:
            raise ValueError("Sum of parsed allocations is non-positive.")
        vals = {g: (100.0 * v / s) for g, v in vals.items()}
        return vals

    def _euclidean_distance(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        keys = v1.keys()
        return math.sqrt(sum((v1[k] - v2[k]) ** 2 for k in keys))

    def evaluate_all_models(self, llm_alloc: Dict[str, float]) -> Dict[str, Dict]:
        out = {}
        for model in ["SF", "EQ", "AL"]:
            pred = self.optimal_allocation_linear(model)
            dist = self._euclidean_distance(pred, llm_alloc)
            out[model] = {"prediction": pred, "distance": dist}
        return out

    def _call_llm(self, llm, prompt) -> tuple[int, str]:
        """Call a specific LLM and return (value, reasoning) tuple"""
        return llm.ask(prompt)

    def calculate_utility(self, c1_effort: float, c2_effort: float) -> float:
        """Calculate utility based on effort allocation using EQ model"""
        return self.calculate_altruistic_utility(c1_effort, c2_effort, "EQ")

    def calculate_friends_benefit(self, c1_effort: float, c2_effort: float) -> float:
        """Calculate total benefit to friends"""
        return self.calculate_friends_utility(c1_effort, c2_effort)

    def calculate_selfish_utility(self, c1_effort: float, c2_effort: float) -> float:
        """Calculate utility considering only own gain (selfish) - SF model with M=1"""
        return self.calculate_own_utility(c1_effort, c2_effort)

    def altruism_score_choice_conditional(self, chosen_c1_effort: float, selfish_c1_effort: float) -> Tuple[float, Dict]:
        """
        Calculate altruism score for coalition game using the theoretical framework
        Returns (altruism_score, details_dict)
        """
        chosen_c2_effort = 100 - chosen_c1_effort
        selfish_c2_effort = 100 - selfish_c1_effort
        
        # Calculate utilities using EQ model (baseline)
        u_chosen = self.calculate_altruistic_utility(chosen_c1_effort, chosen_c2_effort, "EQ")
        u_selfish = self.calculate_own_utility(selfish_c1_effort, selfish_c2_effort)
        
        # Calculate friends benefit
        friends_benefit_chosen = self.calculate_friends_utility(chosen_c1_effort, chosen_c2_effort)
        friends_benefit_selfish = self.calculate_friends_utility(selfish_c1_effort, selfish_c2_effort)
        
        # Calculate altruism score as the difference in friends' benefit
        # This measures how much more the chosen action benefits friends
        friends_benefit_sum = friends_benefit_chosen - friends_benefit_selfish
        friends_harm_sum = 0.0  # No harm in this game
        
        # Altruism score is the difference in friends' benefit
        altruism_score = friends_benefit_sum
        
        details = {
            "u_selfish": u_selfish,
            "u_chosen": u_chosen,
            "friends_benefit_sum": friends_benefit_sum,
            "friends_harm_sum": friends_harm_sum
        }
        
        return altruism_score, details

    def simulate_game(self):
        if not self.llms:
            raise ValueError("No LLMs provided")
        
        prompt = self.build_prompt()
        self.results = []
        
        def ask_model(llm):
            value, reasoning = self._call_llm(llm, prompt)
            
            # Convert the structured value to allocation percentages
            c1_percentage = max(0, min(100, value))  # Clamp between 0 and 100
            c2_percentage = 100 - c1_percentage
            
            # Calculate optimal allocations for each model using theoretical framework
            sf_optimal = self.optimal_allocation_linear("SF")
            eq_optimal = self.optimal_allocation_linear("EQ")
            al_optimal = self.optimal_allocation_linear("AL")
            
            # Calculate pure selfish allocation (only own utility)
            selfish_allocation = self.calculate_pure_selfish_allocation()
            selfish_c1 = selfish_allocation["C1"]
            
            # Calculate altruism score
            altruism_score, details = self.altruism_score_choice_conditional(
                c1_percentage, selfish_c1
            )
            
            # Calculate Euclidean distances to each model's optimal allocation
            llm_allocation = {"C1": c1_percentage, "C2": c2_percentage}
            sf_distance = self._euclidean_distance(llm_allocation, sf_optimal)
            eq_distance = self._euclidean_distance(llm_allocation, eq_optimal)
            al_distance = self._euclidean_distance(llm_allocation, al_optimal)
            
            # Create action labels
            chosen_action = f"C1:{c1_percentage:.1f}%,C2:{c2_percentage:.1f}%"
            selfish_action = f"C1:{selfish_c1:.1f}%,C2:{100-selfish_c1:.1f}%"
            
            result = {
                "llm_name": llm.get_model_name(),
                "agent": "Agent1",  # Fixed agent for this game
                "prompt": prompt.replace("\n", " ").replace(",", " "),
                "llm_value": value,
                "llm_reasoning": reasoning.replace("\n", " ").replace(",", " "),
                "parsed_action": chosen_action,
                "selfish_action": selfish_action,
                "u_selfish": details["u_selfish"],
                "u_chosen": details["u_chosen"],
                "friends_benefit_sum": details["friends_benefit_sum"],
                "friends_harm_sum": details["friends_harm_sum"],
                "ALTRUISM_SCORE": round(altruism_score, 4),
                "SF_distance": round(sf_distance, 4),
                "EQ_distance": round(eq_distance, 4),
                "AL_distance": round(al_distance, 4),
                "SF_optimal_C1": sf_optimal["C1"],
                "EQ_optimal_C1": eq_optimal["C1"],
                "AL_optimal_C1": al_optimal["C1"],
            }
            
            # Write to CSV
            self.writer.writerow(result)
            self.csv_handle.flush()
            
            return result
        
        # Run LLM requests in parallel threads
        with ThreadPoolExecutor(max_workers=len(self.llms)) as executor:
            future_to_llm = {executor.submit(ask_model, llm): llm for llm in self.llms}
            for future in as_completed(future_to_llm):
                result = future.result()
                self.results.append(result)

    def get_results(self) -> List[Dict]:
        return self.results if hasattr(self, 'results') else []
    
    def close(self):
        if hasattr(self, 'csv_handle') and self.csv_handle:
            self.csv_handle.close()
