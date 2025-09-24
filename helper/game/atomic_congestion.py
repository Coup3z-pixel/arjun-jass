import csv
import os
from random import randrange
from typing import Dict, List
from helper.game.game import Game
from helper.llm.LLM import LLM
import concurrent.futures

class AtomicCongestion(Game):
    def __init__(self, config: Dict, csv_save: str = "data/atomic_congestion_results.csv", llms: List[LLM]=[], opponent_strategy: str = "random") -> None:
        assert "total_rounds" in config
        assert "prompt" in config

        print("Game Init")
        print("Current Prompt: " + config["prompt"])

        self.total_rounds = int(config["total_rounds"])
        self.curr_round = 0
        self.llms = llms
        self.opponent_strategy = opponent_strategy
        self.prompt = config["prompt"]

        self.travel_times = [0 for _ in llms]
        self.last_moves_llm = ["" for _ in llms]
        self.last_moves_opp = ["" for _ in llms]


        # payoff matrix
        self.travel_time_matrix = {
            ("R1", "R1"): tuple(int(x) for x in config["R1R1"].split(":")),
            ("R1", "R2"): tuple(int(x) for x in config["R1R2"].split(":")),
            ("R2", "R1"): tuple(int(x) for x in config["R2R1"].split(":")),
            ("R2", "R2"): tuple(int(x) for x in config["R2R2"].split(":")),
        }

        self.matrix_str = {k: v for k, v in self.travel_time_matrix.items()}

        print(self.matrix_str)

        self.csv_file = open(csv_save, "a", newline="")
        self.writer = csv.writer(self.csv_file)

        if not os.path.exists(csv_save) or os.path.getsize(csv_save) == 0:
            self.writer.writerow([
                "round",
                "llm",
                "llm_choice",
                "opponent_choice",
                "reasoning",
                "travel_time",
                "cumulative_time",
                "matrix_str"
            ])

    async def simulate_game(self):
        while self.curr_round < self.total_rounds:
            print(f"\n=== Round {self.curr_round+1} ===")

            # Call all LLMs in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._ask_llm, i) for i in range(len(self.llms))]
                results = [f.result() for f in futures]

            # Process results
            for i, (move_llm, reasoning) in enumerate(results):
                move_opp = self._choose_opponent_move()
                outcome = self.travel_time_matrix[(move_llm, move_opp)]
                self.travel_times[i] += outcome[0]

                self._save_result([
                    self.curr_round + 1,
                    self.llms[i].get_model_name(),
                    move_llm,
                    move_opp,
                    reasoning.replace("\n", " ").replace(",", ""),
                    outcome[0],
                    self.travel_times[i],
                    self.matrix_str
                ])

                print(f"LLM {self.llms[i].get_model_name()}: LLM={move_llm}, Opponent={move_opp}, "
                      f"RoundTime={outcome[0]}, TotalTime={self.travel_times[i]}")

                # update state
                self.last_moves_llm[i] = move_llm
                self.last_moves_opp[i] = move_opp

            self.curr_round += 1

        self.close_results()

    def _ask_llm(self, i: int):
        llm = self.llms[i]
        try:
            value, reasoning = llm.ask(self._generate_prompt(i))
        except Exception as e:
            print(f"[Error] LLM {llm.get_model_name()} failed: {e}")
            value, reasoning = 2, "Defaulted to Route 2 due to error."

        if not isinstance(value, int) or value not in [1, 2]:
            print(f"[Warning] Invalid choice from {llm.get_model_name()}: {value}. Defaulting to Route 2.")
            value = 2
            reasoning += " (Invalid response, defaulted to Route 2.)"

        move = "R1" if value == 1 else "R2"
        return move, reasoning

    def _choose_opponent_move(self):
        if self.opponent_strategy == "random":
            return "R1" if randrange(2) == 0 else "R2"
        elif self.opponent_strategy == "always_r1":
            return "R1"
        elif self.opponent_strategy == "always_r2":
            return "R2"
        else:
            return "R2"

    def _generate_prompt(self, i: int) -> str:
        return self.prompt.format(
            last_move=self.last_moves_llm[i] or "None",
            opponent_move=self.last_moves_opp[i] or "Unknown",
            travel_time=self.travel_times[i]
        )

    def _save_result(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close_results(self):
        self.csv_file.close()
