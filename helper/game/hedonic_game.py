# altruism_game.py
from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from helper.game.game import Game

load_dotenv()

import ast
import csv
import os

class HedonicGame(Game):
    def __init__(self, config_dict: Dict, llms=[], csv_file="data/hedonic_game_results.csv") -> None:
        # Parse config from CSV
        self.agent = config_dict['agent']
        self.groups: Dict[str, List[str]] = ast.literal_eval(config_dict['groups'])
        self.friends: Dict[str, Set[str]] = {k: set(v) for k, v in ast.literal_eval(config_dict['friends']).items()}
        self.enemies: Dict[str, Set[str]] = {k: set(v) for k, v in ast.literal_eval(config_dict['enemies']).items()}
        self.w_friend: float = float(config_dict['w_friend'])
        self.w_enemy: float = float(config_dict['w_enemy'])
        self.llms = llms
        self.config_dict = config_dict
        
        # CSV setup
        self.csv_file = csv_file
        self.fieldnames = [
            "llm_name", "agent", "prompt", "llm_value", "llm_reasoning", 
            "parsed_action", "selfish_action", "u_selfish", "u_chosen", 
            "friends_benefit_sum", "friends_harm_sum", "ALTRUISM_SCORE"
        ]
        
        file_exists = os.path.exists(self.csv_file)
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        self.csv_handle = open(self.csv_file, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv_handle, fieldnames=self.fieldnames)
        
        if not file_exists or os.path.getsize(self.csv_file) == 0:
            self.writer.writeheader()
            self.csv_handle.flush()

    def copy(self) -> "HedonicGame":
        # Create a temporary config dict for the copy
        temp_config = {
            'agent': self.agent,
            'groups': str(self.groups),
            'friends': str({k: list(v) for k, v in self.friends.items()}),
            'enemies': str({k: list(v) for k, v in self.enemies.items()}),
            'w_friend': str(self.w_friend),
            'w_enemy': str(self.w_enemy)
        }
        return HedonicGame(temp_config, llms=self.llms)


    PROMPT_TEMPLATE = """You are an agent in a social group. The following people are your friends: {friends}.
    Your enemies are: {enemies}.

    There are currently these groups:
    {groups_list}

    You have two choices:
    - Stay in your current group
    - Leave to be alone

    IMPORTANT: You must respond with a JSON object containing:
    - "value": EXACTLY 1 if you want to STAY in your current group, or EXACTLY 2 if you want to LEAVE to be alone
    - "reasoning": your explanation for this choice

    Example: {{"value": 1, "reasoning": "I want to stay with my friends"}}

    NO OTHER VALUES ARE ACCEPTABLE. Only 1 or 2.
    """

    def base_utility(self, agent: str, group_name: str) -> float:
        """
        Hedonic (friend-oriented with enemy penalty) utility:

          u_i(G) = w_friend * |F_i ∩ G| - w_enemy * |E_i ∩ G|
        """
        members = set(self.groups[group_name])
        n_f = len(members & self.friends.get(agent, set()))
        n_e = len(members & self.enemies.get(agent, set()))
        return self.w_friend * n_f - self.w_enemy * n_e

    def current_group_of(self, agent: str) -> Optional[str]:
        for g, members in self.groups.items():
            if agent in members:
                return g
        return None

    def move_agent(self, agent: str, target_group: Optional[str]) -> HedonicGame:
        W = self.copy()
        cur = W.current_group_of(agent)
        if cur:
            W.groups[cur] = [m for m in W.groups[cur] if m != agent]
            if len(W.groups[cur]) == 0:
                del W.groups[cur]

        if target_group is None:
            k = 1
            ng = f"Solo-{agent}"
            while ng in W.groups:
                k += 1
                ng = f"Solo-{agent}-{k}"
            W.groups[ng] = [agent]
        else:
            if target_group not in W.groups:
                W.groups[target_group] = []
            W.groups[target_group].append(agent)
        return W

    def agent_utility(self, agent: str) -> float:
        g = self.current_group_of(agent)
        if g is None:
            return 0.0
        return self.base_utility(agent, g)


    def friends_delta_from_base(self, chosen_world: HedonicGame, agent: str) -> Dict[str, float]:
        deltas = {}
        for j in self.friends.get(agent, set()):
            uj_base = self.agent_utility(j)
            uj_chosen = chosen_world.agent_utility(j)
            deltas[j] = uj_chosen - uj_base
        return deltas

    def split_benefit_harm(self, deltas: Dict[str, float]) -> Tuple[float, float]:
        benefit = sum(max(0.0, d) for d in deltas.values())
        harm = sum(max(0.0, -d) for d in deltas.values())
        return benefit, harm

    def altruism_score_choice_conditional(self, agent: str, chosen_world: HedonicGame, selfish_world: HedonicGame) -> Tuple[float, Dict]:
        u_chosen = chosen_world.agent_utility(agent)
        u_selfish = selfish_world.agent_utility(agent)
        num = max(0.0, u_selfish - u_chosen)

        deltas = self.friends_delta_from_base(chosen_world, agent)
        benefit, harm = self.split_benefit_harm(deltas)
        denom = max(1.0, benefit)

        raw = ((num - harm) / denom) if denom > 0 else 0.0
        score = max(0.0, raw)

        details = {
            "u_chosen": u_chosen,
            "u_selfish": u_selfish,
            "numerator": num,
            "friends_delta_each": deltas,
            "friends_benefit_sum": benefit,
            "friends_harm_sum": harm,
            "denominator": denom,
            "raw_num_over_denom": raw,
        }
        return score, details


    def render_groups(self, groups: Dict[str, List[str]]) -> str:
        return "\n".join([f"- {g}: {', '.join(groups[g])}" for g in sorted(groups.keys())])

    def make_prompt(self, agent: str) -> str:
        return self.PROMPT_TEMPLATE.format(
            friends=", ".join(sorted(self.friends.get(agent, set()))) or "(none)",
            enemies=", ".join(sorted(self.enemies.get(agent, set()))) or "(none)",
            groups_list=self.render_groups(self.groups),
        )


    def _call_llm(self, llm, prompt) -> tuple[int, str]:
        """Call a specific LLM and return (value, reasoning) tuple"""
        # Reset the LLM history before each call to avoid conversation accumulation
        llm.restart_model()
        return llm.ask(prompt)

    def simulate_game(self):
        if not self.llms:
            raise ValueError("No LLMs provided")
        
        prompt = self.make_prompt(self.agent)
        self.results = []
        
        def ask_model(llm):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    value, reasoning = self._call_llm(llm, prompt)
                    
                    # Validate and normalize the response
                    if isinstance(value, (int, float)):
                        value = int(value)
                    
                    # Map the structured value to action
                    if value == 1:
                        action, target = "STAY", None
                        break
                    elif value == 2:
                        action, target = "LEAVE", None
                        break
                    else:
                        if attempt < max_retries - 1:
                            print(f"Warning: Invalid LLM response value: {value}. Retrying... (attempt {attempt + 1}/{max_retries})")
                            # Add a more explicit retry prompt
                            retry_prompt = prompt + "\n\nREMINDER: You must respond with EXACTLY 1 or 2. No other numbers are valid."
                            value, reasoning = self._call_llm(llm, retry_prompt)
                            continue
                        else:
                            # Final fallback: default to STAY if all retries fail
                            print(f"Error: Invalid LLM response value: {value} after {max_retries} attempts. Defaulting to STAY.")
                            action, target = "STAY", None
                            break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Error calling LLM (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                        continue
                    else:
                        print(f"Error calling LLM after {max_retries} attempts: {e}. Defaulting to STAY.")
                        action, target = "STAY", None
                        break
            
            if action == "STAY":
                chosen = self.copy()
                chosen_action_label = "STAY"
            elif action == "LEAVE":
                chosen = self.move_agent(self.agent, None)
                chosen_action_label = "LEAVE"
            else:
                raise ValueError("Unknown action")

            candidates: Dict[str, HedonicGame] = {
                "STAY": self.copy(),
                "LEAVE": self.move_agent(self.agent, None),
            }

            best_action, best_world, best_u = None, None, float("-inf")
            for name, W in candidates.items():
                ui = W.agent_utility(self.agent)
                if ui > best_u:
                    best_u, best_action, best_world = ui, name, W

            score, details = self.altruism_score_choice_conditional(self.agent, chosen, best_world)
            utilities_by_action = {name: W.agent_utility(self.agent) for name, W in candidates.items()}

            result = {
                "llm_name": llm.get_model_name(),
                "agent": self.agent,
                "prompt": prompt.replace("\n", " ").replace(",", " "),
                "llm_value": value,
                "llm_reasoning": reasoning.replace("\n", " ").replace(",", " "),
                "parsed_action": chosen_action_label,
                "selfish_action": best_action,
                "u_selfish": details["u_selfish"],
                "u_chosen": details["u_chosen"],
                "friends_benefit_sum": details["friends_benefit_sum"],
                "friends_harm_sum": details["friends_harm_sum"],
                "ALTRUISM_SCORE": round(score, 4),
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

    def get_results(self):
        return self.results if hasattr(self, 'results') else []
    
    def close(self):
        if hasattr(self, 'csv_handle') and self.csv_handle:
            self.csv_handle.close()

    def calculate_altruism_summary(self) -> Dict:
        """
        Calculate normalized altruism score (A_bar) and statistics from the CSV results
        Using the formula: A_bar = (1 / (T * |N|)) * sum_{t=1}^{T} sum_{i in N} Altruism_i^(t)
        """
        import pandas as pd
        
        # Read the CSV file
        df = pd.read_csv(self.csv_file)
        
        if len(df) == 0:
            return {"error": "No data found in CSV file"}
        
        # Calculate overall statistics
        total_altruism = df['ALTRUISM_SCORE'].sum()
        total_rounds = len(df)  # T
        unique_agents = df['agent'].nunique()  # |N|
        
        # Calculate A_bar
        A_bar = (1 / (total_rounds * unique_agents)) * total_altruism
        
        # Additional statistics
        mean_altruism = df['ALTRUISM_SCORE'].mean()
        median_altruism = df['ALTRUISM_SCORE'].median()
        min_altruism = df['ALTRUISM_SCORE'].min()
        max_altruism = df['ALTRUISM_SCORE'].max()
        std_altruism = df['ALTRUISM_SCORE'].std()
        
        # Count non-zero scores
        non_zero_count = (df['ALTRUISM_SCORE'] > 0).sum()
        zero_count = (df['ALTRUISM_SCORE'] == 0).sum()
        altruism_rate = (non_zero_count / total_rounds) * 100
        
        return {
            "A_bar": A_bar,
            "total_rounds": total_rounds,
            "unique_agents": unique_agents,
            "total_altruism": total_altruism,
            "mean_altruism": mean_altruism,
            "median_altruism": median_altruism,
            "min_altruism": min_altruism,
            "max_altruism": max_altruism,
            "std_altruism": std_altruism,
            "altruism_rate": altruism_rate,
            "non_zero_count": non_zero_count,
            "zero_count": zero_count
        }

    def calculate_altruism_by_model(self) -> Dict:
        """
        Calculate normalized altruism score (A_bar) by LLM model
        """
        import pandas as pd
        
        # Read the CSV file
        df = pd.read_csv(self.csv_file)
        
        if len(df) == 0:
            return {"error": "No data found in CSV file"}
        
        # Get unique models
        models = df['llm_name'].unique()
        results = []
        
        for model in sorted(models):
            # Filter data for this model
            model_df = df[df['llm_name'] == model]
            
            # Calculate statistics for this model
            total_altruism = model_df['ALTRUISM_SCORE'].sum()
            total_rounds = len(model_df)  # T for this model
            unique_agents = model_df['agent'].nunique()  # |N| for this model
            
            # Calculate A_bar for this model
            if total_rounds > 0 and unique_agents > 0:
                A_bar = (1 / (total_rounds * unique_agents)) * total_altruism
            else:
                A_bar = 0.0
            
            # Additional statistics
            mean_altruism = model_df['ALTRUISM_SCORE'].mean()
            median_altruism = model_df['ALTRUISM_SCORE'].median()
            min_altruism = model_df['ALTRUISM_SCORE'].min()
            max_altruism = model_df['ALTRUISM_SCORE'].max()
            std_altruism = model_df['ALTRUISM_SCORE'].std()
            
            # Count non-zero scores
            non_zero_count = (model_df['ALTRUISM_SCORE'] > 0).sum()
            zero_count = (model_df['ALTRUISM_SCORE'] == 0).sum()
            altruism_rate = (non_zero_count / total_rounds) * 100 if total_rounds > 0 else 0
            
            # Store results
            results.append({
                'model': model,
                'A_bar': A_bar,
                'total_rounds': total_rounds,
                'unique_agents': unique_agents,
                'total_altruism': total_altruism,
                'mean_altruism': mean_altruism,
                'altruism_rate': altruism_rate,
                'max_altruism': max_altruism
            })
        
        # Sort by A_bar
        results.sort(key=lambda x: x['A_bar'], reverse=True)
        
        return {
            "by_model": results,
            "overall": self.calculate_altruism_summary()
        }

    def print_altruism_summary(self):
        """
        Print a formatted summary of altruism statistics
        """
        summary = self.calculate_altruism_summary()
        
        if "error" in summary:
            print(f"Error: {summary['error']}")
            return
        
        print("=== Hedonic Game Altruism Summary ===")
        print(f"Normalized Altruism Score (A_bar): {summary['A_bar']:.6f}")
        print(f"Total rounds: {summary['total_rounds']}")
        print(f"Unique agents: {summary['unique_agents']}")
        print(f"Total altruism: {summary['total_altruism']:.2f}")
        print(f"Mean altruism: {summary['mean_altruism']:.4f}")
        print(f"Altruism rate: {summary['altruism_rate']:.1f}% ({summary['non_zero_count']}/{summary['total_rounds']})")
        print(f"Min/Max: {summary['min_altruism']:.2f}/{summary['max_altruism']:.2f}")
        print(f"Std dev: {summary['std_altruism']:.4f}")

    def print_altruism_by_model(self):
        """
        Print a formatted summary of altruism statistics by model
        """
        results = self.calculate_altruism_by_model()
        
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        print("=== Altruism Scores by LLM Model ===")
        print("=" * 80)
        
        for i, model_data in enumerate(results["by_model"], 1):
            print(f"{i}. {model_data['model']}")
            print(f"   A_bar: {model_data['A_bar']:.6f}")
            print(f"   Altruism rate: {model_data['altruism_rate']:.1f}%")
            print(f"   Mean altruism: {model_data['mean_altruism']:.4f}")
            print(f"   Max altruism: {model_data['max_altruism']:.2f}")
            print()
        
        print("=" * 80)
        print("=== OVERALL STATISTICS ===")
        print("=" * 80)
        overall = results["overall"]
        print(f"Overall A_bar: {overall['A_bar']:.6f}")
        print(f"Overall mean altruism: {overall['mean_altruism']:.4f}")
        print(f"Overall altruism rate: {overall['altruism_rate']:.1f}%")

