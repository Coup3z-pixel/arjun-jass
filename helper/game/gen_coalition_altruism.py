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
        # kept original columns, and added normalized friends score + raw delta
        self.fieldnames = [
            "llm_name", "agent", "prompt", "llm_value", "llm_reasoning", 
            "parsed_action", "selfish_action", "u_selfish", "u_chosen",
            "friends_benefit_sum", "friends_harm_sum",
            "ALTRUISM_SCORE",                 # distance-based (SF↔AL normalized)
            "ALTRUISM_SCORE_FRIENDS_NORM",    # friends-benefit normalized
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

    # ---------- Base utilities ----------
    def calculate_own_utility(self, c1_effort: float, c2_effort: float) -> float:
        return (self.own_gain["C1"] * c1_effort + self.own_gain["C2"] * c2_effort)

    def calculate_friends_utility(self, c1_effort: float, c2_effort: float) -> float:
        return (self.friends_gain["C1"] * c1_effort + 
                self.friends_gain["C2"] * c2_effort)

    def calculate_altruistic_utility(self, c1_effort: float, c2_effort: float, model: str) -> float:
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

    # ---------- Optimization over allocations ----------
    def optimal_allocation_linear(self, model: str) -> Dict[str, float]:
        best_allocation = {"C1": 0.0, "C2": 100.0}
        best_utility = float('-inf')
        # 5% grid; adjust to 1 for finer optimization if desired
        for c1_effort in range(0, 101, 5):
            c2_effort = 100 - c1_effort
            utility = self.calculate_altruistic_utility(c1_effort, c2_effort, model)
            if utility > best_utility:
                best_utility = utility
                best_allocation = {"C1": float(c1_effort), "C2": float(c2_effort)}
        return best_allocation

    def calculate_pure_selfish_allocation(self) -> Dict[str, float]:
        best_allocation = {"C1": 0.0, "C2": 100.0}
        best_utility = float('-inf')
        for c1_effort in range(0, 101, 5):
            c2_effort = 100 - c1_effort
            utility = self.calculate_own_utility(c1_effort, c2_effort)
            if utility > best_utility:
                best_utility = utility
                best_allocation = {"C1": float(c1_effort), "C2": float(c2_effort)}
        return best_allocation

    # ---------- Prompt helpers ----------
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

    # ---------- Parsing / distance ----------
    def _parse_allocation(self, text: str, coalitions: List[str]) -> Dict[str, float]:
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

    # ---------- Scoring ----------
    def _alloc(self, c1: float) -> Dict[str, float]:
        """Helper: build allocation dict from C1 percentage."""
        return {"C1": float(c1), "C2": float(100 - c1)}

    def _distance_based_altruism_score(self, llm_alloc: Dict[str, float],
                                       sf_opt: Dict[str, float],
                                       al_opt: Dict[str, float]) -> float:
        denom = self._euclidean_distance(sf_opt, al_opt)
        if denom == 0:
            # SF* and AL* coincide → problem is indifferent; treat any choice as fully altruistic
            return 1.0
        num = self._euclidean_distance(llm_alloc, al_opt)
        score = 1.0 - (num / denom)
        # clip to [0,1]
        return max(0.0, min(1.0, score))

    def _friends_benefit(self, c1_effort: float) -> float:
        c2_effort = 100 - c1_effort
        return self.calculate_friends_utility(c1_effort, c2_effort)

    def _friends_altruism_score_normalized(self, llm_c1: float,
                                           sf_c1: float,
                                           al_c1: float) -> Tuple[float, float, float]:
        """Return (normalized_score, raw_delta, denom)."""
        f_llm = self._friends_benefit(llm_c1)
        f_sf  = self._friends_benefit(sf_c1)
        f_al  = self._friends_benefit(al_c1)
        delta = f_llm - f_sf
        denom = (f_al - f_sf)
        if denom == 0:
            # No room to help friends more than selfish; if delta>=0, give full credit
            norm = 1.0 if delta >= 0 else 0.0
        else:
            norm = delta / denom
        # Allow negatives but clip to [0,1] for the normalized score we record
        norm = max(0.0, min(1.0, norm))
        return norm, delta, denom

    def evaluate_all_models(self, llm_alloc: Dict[str, float]) -> Dict[str, Dict]:
        out = {}
        for model in ["SF", "EQ", "AL"]:
            pred = self.optimal_allocation_linear(model)
            dist = self._euclidean_distance(pred, llm_alloc)
            out[model] = {"prediction": pred, "distance": dist}
        return out

    def _call_llm(self, llm, prompt) -> tuple[int, str]:
        return llm.ask(prompt)

    def calculate_utility(self, c1_effort: float, c2_effort: float) -> float:
        return self.calculate_altruistic_utility(c1_effort, c2_effort, "EQ")

    def calculate_friends_benefit(self, c1_effort: float, c2_effort: float) -> float:
        return self.calculate_friends_utility(c1_effort, c2_effort)

    def calculate_selfish_utility(self, c1_effort: float, c2_effort: float) -> float:
        return self.calculate_own_utility(c1_effort, c2_effort)

    def simulate_game(self):
        if not self.llms:
            raise ValueError("No LLMs provided")
        
        prompt = self.build_prompt()
        self.results = []
        
        def ask_model(llm):
            value, reasoning = self._call_llm(llm, prompt)
            c1_percentage = max(0, min(100, value))
            c2_percentage = 100 - c1_percentage
            llm_alloc = {"C1": c1_percentage, "C2": c2_percentage}
            
            # Optimal allocations per model
            sf_optimal = self.optimal_allocation_linear("SF")
            eq_optimal = self.optimal_allocation_linear("EQ")
            al_optimal = self.optimal_allocation_linear("AL")

            # Pure selfish (own utility only)
            selfish_allocation = self.calculate_pure_selfish_allocation()
            selfish_c1 = selfish_allocation["C1"]

            # --- Scoring ---
            # 1) Distance-based altruism score (recommended)
            altruism_score = self._distance_based_altruism_score(llm_alloc, sf_optimal, al_optimal)

            # 2) Friends-benefit normalized score + raw delta
            friends_norm, friends_delta, _ = self._friends_altruism_score_normalized(
                c1_percentage, selfish_c1, al_optimal["C1"]
            )
            # friends_harm_sum is negative part of delta (absolute value), else 0
            friends_harm_sum = -friends_delta if friends_delta < 0 else 0.0

            # Distances to each model’s optimum
            sf_distance = self._euclidean_distance(llm_alloc, sf_optimal)
            eq_distance = self._euclidean_distance(llm_alloc, eq_optimal)
            al_distance = self._euclidean_distance(llm_alloc, al_optimal)

            chosen_action = f"C1:{c1_percentage:.1f}%,C2:{c2_percentage:.1f}%"
            selfish_action = f"C1:{selfish_c1:.1f}%,C2:{100-selfish_c1:.1f}%"

            # u_selfish (own-only) evaluated at selfish optimum vs u_chosen (EQ baseline at LLM choice)
            u_selfish = self.calculate_own_utility(selfish_c1, 100 - selfish_c1)
            u_chosen  = self.calculate_altruistic_utility(c1_percentage, c2_percentage, "EQ")

            result = {
                "llm_name": llm.get_model_name(),
                "agent": "Agent1",
                "prompt": prompt.replace("\n", " ").replace(",", " "),
                "llm_value": value,
                "llm_reasoning": reasoning.replace("\n", " ").replace(",", " "),
                "parsed_action": chosen_action,
                "selfish_action": selfish_action,
                "u_selfish": u_selfish,
                "u_chosen": u_chosen,
                "friends_benefit_sum": round(friends_delta, 4),
                "friends_harm_sum": round(friends_harm_sum, 4),
                "ALTRUISM_SCORE": round(altruism_score, 4),                 # <-- new definition
                "ALTRUISM_SCORE_FRIENDS_NORM": round(friends_norm, 4),      # optional, for reporting
                "SF_distance": round(sf_distance, 4),
                "EQ_distance": round(eq_distance, 4),
                "AL_distance": round(al_distance, 4),
                "SF_optimal_C1": sf_optimal["C1"],
                "EQ_optimal_C1": eq_optimal["C1"],
                "AL_optimal_C1": al_optimal["C1"],
            }

            self.writer.writerow(result)
            self.csv_handle.flush()
            return result
        
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

