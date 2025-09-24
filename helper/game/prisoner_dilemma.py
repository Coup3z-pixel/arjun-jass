from typing import Dict, List
from random import randrange
from helper.game.game import Game
from helper.llm.LLM import LLM
import csv
import os
import concurrent.futures


class PrisonersDilemma(Game):
    def __init__(self, config: Dict, csv_save: str = "data/prisoner_dilemma.csv", llms: List[LLM] = [], opponent_strategy: str = "random") -> None:
        assert 'total_rounds' in config
        assert 'prompt' in config

        print("Game Initialized")

        self.total_rounds = int(config['total_rounds'])
        self.curr_round = 0
        self.llms = llms
        self.opponent_strategy = opponent_strategy
        self.prompt = config['prompt']

        # track each LLM’s state
        self.points = [0 for _ in llms]
        self.last_moves_llm = ["" for _ in llms]
        self.last_moves_opp = ["" for _ in llms]

        self.payoff_matrix = {
            ("C", "C"): tuple(int(x) for x in config["CC"].split(":")),
            ("C", "D"): tuple(int(x) for x in config["CD"].split(":")),
            ("D", "C"): tuple(int(x) for x in config["DC"].split(":")),
            ("D", "D"): tuple(int(x) for x in config["DD"].split(":")),
        }

        # CSV setup
        os.makedirs(os.path.dirname(csv_save), exist_ok=True)
        self.csv_file = open(csv_save, "a", newline="")
        self.writer = csv.writer(self.csv_file)

        if not os.path.exists(csv_save) or os.path.getsize(csv_save) == 0:
            self.writer.writerow(["round", "llm", "llm_move", "opponent_move", "reasoning", "points_after_round"])

    async def simulate_game(self):
        while self.curr_round < self.total_rounds:
            print(f"\n=== Round {self.curr_round + 1} ===")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                # launch all LLMs in parallel
                futures = [executor.submit(self._ask_llm, i) for i in range(len(self.llms))]
                results = [f.result() for f in futures]

            # process results and calculate payoffs
            for i, (move_llm, reasoning) in enumerate(results):
                move_opp = self._choose_opponent_move()
                payoff = self.payoff_matrix[(move_llm, move_opp)]
                self.points[i] += payoff[0]

                self._save_result([
                    self.curr_round + 1,
                    self.llms[i].get_model_name(),
                    move_llm,
                    move_opp,
                    reasoning.replace("\n", " ").replace(",", ""),
                    self.points[i]
                ])

                print(f"LLM {self.llms[i].get_model_name()}: LLM={move_llm}, Opponent={move_opp}, "
                      f"Payoff={payoff}, Total={self.points[i]}")

                # update state
                self.last_moves_llm[i] = move_llm
                self.last_moves_opp[i] = move_opp

            self.curr_round += 1

    def _ask_llm(self, i: int):
        llm = self.llms[i]
        # Reset the LLM history before each call to avoid conversation accumulation
        llm.restart_model()
        try:
            value, reasoning = llm.ask(self._generate_prompt(i))
        except Exception as e:
            print(f"[Error] LLM {llm.get_model_name()} failed to respond: {e}")
            value, reasoning = 2, "Defaulted to Defect due to error."

        # validate move
        if value not in [1, 2]:
            print(f"[Warning] Invalid move from {llm.get_model_name()}: {value}. Defaulting to Defect (2).")
            value = 2
            reasoning += " (Invalid response, defaulted to Defect.)"

        move = "C" if value == 1 else "D"
        return move, reasoning

    def _choose_opponent_move(self):
        if self.opponent_strategy == "random":
            return "C" if randrange(2) == 0 else "D"
        elif self.opponent_strategy == "always_defect":
            return "D"
        elif self.opponent_strategy == "always_cooperate":
            return "C"
        else:
            return "D"

    def _generate_prompt(self, i: int) -> str:
        return self.prompt.format(
            last_move=self.last_moves_llm[i] or "None",
            opponent_move=self.last_moves_opp[i] or "Unknown",
            total_score=self.points[i]
        )

    def _save_result(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()
