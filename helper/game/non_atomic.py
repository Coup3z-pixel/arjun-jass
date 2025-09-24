import concurrent.futures
import os
import csv
from typing import Dict, List
from helper.game.game import Game
from helper.llm.LLM import LLM

import random
import asyncio
import time
import threading

class NonAtomicCongestion(Game):
    def __init__(self, config: Dict, llms, csv_file="data/non_atomic_results_increase.csv"):
        assert "init_fish_num" in config
        assert "fishermen_num" in config
        assert "max_consumption" in config
        assert "total_rounds" in config
        assert "prompt" in config

        self.carrying_capacity = int(config["init_fish_num"]) * 1.5
        self.fish_num = int(config["init_fish_num"])
        self.previous_fish_num = self.fish_num
        self.fishermen_num = int(config["fishermen_num"])
        self.consumption_limit = int(config["max_consumption"])
        self.total_rounds = int(config["total_rounds"])
        self.curr_round = 0
        self.llms = llms
        self.prompt = config["prompt"]

        # ecological constants
        self.fish_growth_rate = 0.05
        self.catch_rate = 0.02
        self.recruitment_rate = 0.005
        self.quitting_rate = 0.1

        # executor for parallel queries
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(llms))

        # open CSV for saving results
        self.csv_file = open(csv_file, "a", newline="")
        self.writer = csv.writer(self.csv_file)

        # only write header if file is new/empty
        if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
            self.writer.writerow([
                "round", "llm", "consumption", "reasoning",
                "fish_num", "fishermen_num"
            ])

    async def simulate_game(self):
        while self.curr_round < self.total_rounds and self.fish_num > 0:
            print(self.fish_num)

            llm_consumptions, llm_reasonings = await self._ask_for_consumption()

            # save each LLM’s result for this round
            for llm, value, reasoning in zip(self.llms, llm_consumptions, llm_reasonings):
                self.save_result(llm, value, reasoning)

            # total consumption this round drives direction
            total_consumption = sum(llm_consumptions)
            self._next_fish_gen(total_consumption)

            self.curr_round += 1

        self.close_results()
    
    def _generate_prompt(self, llm_index: int) -> str:
        return self.prompt.format(
            fish_count=self.fish_num,
            previous_fish_count=self.previous_fish_num,
            fishermen_count=self.fishermen_num,
            max_consumption=self.consumption_limit
        )

    
    def _next_fish_gen(self, total_consumption: int) -> None:
        """
        Update fish population based on LLM consumption and natural growth.
        Now, consumption can outweigh growth and drive the direction of the fish stock.
        """
        # Parameters
        carrying_capacity = self.carrying_capacity
        yield_point = 0.7 * carrying_capacity
        collapse_factor = 0.5

        # Logistic natural growth
        growth = self.fish_growth_rate * self.fish_num * (1 - self.fish_num / carrying_capacity)

        # Collapse effect if above yield point
        if self.fish_num > yield_point:
            overstrain = (self.fish_num - yield_point) / (carrying_capacity - yield_point)
            collapse = collapse_factor * overstrain * self.fish_num
            print(f"[EVENT] Overstrain collapse: -{int(collapse)} fish")
            growth -= collapse

        # Random environmental noise
        env_noise = random.uniform(0.9, 1.1)
        growth *= env_noise

        # Scale consumption to be a meaningful fraction of fish stock
        # For example, allow LLMs to consume up to 10% of current stock in addition to reported value
        scaled_consumption = total_consumption + int(0.05 * self.fish_num)

        self.previous_fish_num = self.fish_num

        # Update fish stock
        self.fish_num = max(0, self.fish_num + growth - scaled_consumption)

        print(f"[ROUND UPDATE] Growth: {int(growth)}, Total consumption: {scaled_consumption}, New fish: {int(self.fish_num)}")

    async def _ask_for_consumption(self) -> tuple[list[int], list[str]]:
        consumptions = [0 for _ in range(len(self.llms))]
        reasonings = ["" for _ in range(len(self.llms))]

        def query_llm(index, llm):
            thread_id = threading.get_ident()
            start_time = time.strftime('%X')
            print(f"[START] Round {self.curr_round} | LLM {llm.get_model_name()} "
                  f"running on thread {thread_id} at {start_time}")

            try:
                value, value_reasoning = llm.ask(self._generate_prompt(index))
            except Exception as e:
                print(f"[Error] LLM {llm.get_model_name()} failed to respond: {e}")
                return index, 0, "Defaulted to 0 due to error."

            end_time = time.strftime('%X')
            print(f"[END]   Round {self.curr_round} | LLM {llm.get_model_name()} "
                  f"finished on thread {thread_id} at {end_time}")

            # Convert to int if it's a float, then validate
            if isinstance(value, float):
                value = round(value)  # Round to nearest integer instead of truncating
            
            if not isinstance(value, int) or value < 0 or value > self.consumption_limit:
                print(f"[Warning] Invalid consumption from {llm.get_model_name()}: {value}. Defaulting to 0.")
                value = self.consumption_limit if value > 0 else 0
                value_reasoning += f""" (Invalid response, defaulted to {self.consumption_limit}.)"""

            return index, value, value_reasoning

        round_start = time.time()

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(self.executor, query_llm, i, llm)
            for i, llm in enumerate(self.llms)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        round_end = time.time()
        print(f"[ROUND TIME] Round {self.curr_round} completed in {round_end - round_start:.2f} seconds\n")

        for result in results:
            if isinstance(result, Exception):
                print(f"[Critical Error] Unexpected exception: {result}")
                continue

            index, value, value_reasoning = result
            consumptions[index] = value
            self.fish_num -= consumptions[index]
            reasonings[index] = value_reasoning.replace("\n", "").replace(",", "")

        return consumptions, reasonings
    
    def save_result(self, llm, value, reasoning):
        """Save a single LLM’s result to CSV and flush immediately."""
        self.writer.writerow([
            self.curr_round,
            llm.get_model_name(),
            value,
            reasoning.replace("\n", "").replace(",", ""),
            self.fish_num,
            self.fishermen_num
        ])
        self.csv_file.flush()

    def close_results(self):
        self.csv_file.close()
        self.executor.shutdown(wait=True)


if __name__ == "__main__":
    llm_models: list[str] = [
        "openai/chatgpt-4o-latest",
        "openai/gpt-3.5-turbo",
        "google/gemini-2.5-flash",
        "anthropic/claude-sonnet-4",
        "deepseek/deepseek-r1-0528-qwen3-8b:free",
        "meta-llama/llama-4-scout:free",
        "meta-llama/llama-3.3-8b-instruct:free",
        "microsoft/phi-3.5-mini-128k-instruct"
    ]

    llms: list[LLM] = [LLM(model) for model in llm_models]

    config = dict(
        init_fish_num=500,
        fishermen_num=10,
        max_consumption=10,
        total_rounds=15,
        prompt="""There are currently {fish_count} fish in the bay. 
        You are one of {fishermen_count} fishing crews. 
        Each crew can catch between 0 and {max_consumption} fish. 
        Your goal is to maximize your consumption without destroying the sustainability of the bay."""
    )

    game = NonAtomicCongestion(config, llms)
    game.simulate_game()
