import os
import csv
from random import randrange
from typing import Dict, List
import concurrent.futures

from helper.game.game import Game
from helper.llm.LLM import LLM

import threading
import time

class SocialContext(Game):
    def __init__(self, config: Dict, csv_file: str = "data/social_context_results.csv", llms: List[LLM] = []) -> None:
        assert "rounds" in config
        assert "prompt" in config

        self.prompt = config["prompt"]
        self.total_rounds = int(config["rounds"])
        self.curr_round = 0
        self.rank_no = len(llms)
        self.points: List[int] = [0 for _ in range(len(llms))]
        self.last_round_ranks: List[int] = [-1 for _ in range(len(llms))]
        self.llms = llms
        self.llm_histories: List[dict] = [
            {"model": llm.get_model_name(), "rounds": []} for llm in llms
        ]
        self.lastest_reasoning = ["" for _ in range(len(llms))]

        # Thread pool for concurrent LLM calls
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(llms))

        # Open CSV once and add header
        self.csv_file = open(csv_file, "a", newline="")
        self.writer = csv.writer(self.csv_file)

        if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
            self.writer.writerow(
                ["round", "llm", "proposed_rank", "reasoning", "final_rank", "points_after_round"]
            )

    async def simulate_game(self):
        proposed_ranks_by_round: List[List[List[int]]] = []
        final_ranks_by_round: List[List[int]] = []

        while self.curr_round < self.total_rounds:
            proposed_ranks: List[List[int]] = self._ask_for_rank()
            proposed_ranks_by_round.append(proposed_ranks)

            final_rankings: List[int] = self.resolve_congestion(proposed_ranks)
            final_ranks_by_round.append(final_rankings)

            print("Final ranks have been chosen")
            print(proposed_ranks)
            print(final_rankings)

            # Update per-LLM round history
            for llm_idx, llm in enumerate(self.llms):
                proposed_rank = None
                for r_idx, lst in enumerate(proposed_ranks):
                    if llm_idx in lst:
                        proposed_rank = r_idx + 1
                        break

                final_rank = None
                if llm_idx in final_rankings:
                    final_rank = final_rankings.index(llm_idx) + 1

                if final_rank is not None:
                    self.points[llm_idx] += self.rank_no - final_rank + 1

                self._save_result([
                    self.curr_round + 1,
                    llm.get_model_name(),
                    proposed_rank,
                    self.lastest_reasoning[llm_idx],
                    final_rank,
                    self.points[llm_idx]
                ])

            self.last_round_ranks = [
                (r + 1 if r != -1 else -1) for r in final_rankings
            ]
            self.curr_round += 1



    def _ask_for_rank(self) -> List[List[int]]:
        ranking = [[] for _ in range(self.rank_no)]
        reasoning = ["" for _ in range(len(self.llms))]

        def query_llm(index, llm):
            thread_id = threading.get_ident()
            start_time = time.strftime("%H:%M:%S")
            print(f"[START] Round {self.curr_round} | LLM {llm.get_model_name()} running on thread {thread_id} at {start_time}")

            try:
                # Reset the LLM history before each call to avoid conversation accumulation
                llm.restart_model()
                value, value_reasoning = llm.ask(
                    self._generate_prompt(index)
                )
            except Exception as e:
                print(f"[Error] LLM {llm.get_model_name()} failed to respond: {e}")
                value, value_reasoning = self.rank_no, "Defaulted to lowest rank due to error."

            if not isinstance(value, int) or value < 1 or value > self.rank_no:
                print(f"[Warning] Invalid rank from {llm.get_model_name()}: {value}. Defaulting to {self.rank_no}.")
                value = self.rank_no
                value_reasoning += " (Invalid response, defaulted to lowest rank.)"

            end_time = time.strftime("%H:%M:%S")
            print(f"[END] Round {self.curr_round} | LLM {llm.get_model_name()} finished on thread {thread_id} at {end_time}")

            return index, value, value_reasoning

        # Submit all jobs to ThreadPoolExecutor
        futures = [
            self.executor.submit(query_llm, i, llm)
            for i, llm in enumerate(self.llms)
        ]

        for future in concurrent.futures.as_completed(futures):
            index, value, value_reasoning = future.result()
            ranking[value - 1].append(index)
            reasoning[index] = value_reasoning.replace("\n", "").replace(",", "")

        self.lastest_reasoning = reasoning
        return ranking

    def resolve_congestion(self, proposed_ranks: List[List[int]]) -> List[int]:
        N = self.rank_no
        proposed = [lst.copy() for lst in proposed_ranks]
        final_ranks = [0] * N

        rank = 0
        while rank < N:
            while not proposed[rank]:
                if not any(proposed[i] for i in range(rank + 1, N)):
                    break
                for s in range(rank, N - 1):
                    proposed[s] = proposed[s + 1].copy()
                proposed[N - 1] = []

            candidates = proposed[rank]
            if len(candidates) == 0:
                final_ranks[rank] = 0
            elif len(candidates) == 1:
                final_ranks[rank] = candidates[0]
                proposed[rank] = []
            else:
                chosen_idx = randrange(len(candidates))
                final_ranks[rank] = candidates[chosen_idx]
                leftovers = [c for i, c in enumerate(candidates) if i != chosen_idx]
                proposed[rank] = []
                next_empty = None
                for k in range(rank + 1, N):
                    if not proposed[k]:
                        next_empty = k
                        break
                if next_empty is not None:
                    proposed[next_empty] = leftovers.copy()
            rank += 1

        return final_ranks

    def _generate_prompt(self, llm_index: int) -> str:
        return self.prompt.format(
            player_num=len(self.llms),
            rank_count=self.rank_no,
            rank_received=self.last_round_ranks[llm_index],
            llm_points=self.points[llm_index]
        )

    def _save_result(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()
