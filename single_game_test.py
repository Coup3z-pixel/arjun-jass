import csv
from typing import Type
from datetime import datetime

from helper.game.atomic_congestion import AtomicCongestion
from helper.game.cost_sharing_scheduling import CostSharingGame
from helper.game.dictator_game import DictatorGame
from helper.game.game import Game
from helper.game.non_atomic import NonAtomicCongestion
from helper.game.social_context import SocialContext
from helper.game.hedonic_game import HedonicGame
from helper.game.gen_coalition_altruism import GenCoalition
from helper.llm.LLM import LLM

from helper.game.prisoner_dilemma import PrisonersDilemma


if __name__ == "__main__":
    type_of_games: list[Type[Game]] = [
            #PrisonersDilemma,
            #HedonicGame,
            # AtomicCongestion,
            # SocialContext,
            # NonAtomicCongestion
            # CostSharingGame,
            # DictatorGame,
            GenCoalition,
    ]

    file_names: list[str] = [
            #"PrisonnersDilemma.csv",
            #"HedonicGame.csv",
            # "AtomicCongestion.csv",
            # "SocialContext.csv",
            # "NonAtomicCongestion.csv",
            # "CostSharingGame.csv",
            # "DictatorGame.csv",
            "GenCoalition.csv"
    ]


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

    llms: list[LLM] = []

    for model in llm_models:
        llms.append(LLM(model)) 



    def reset_llms():
        for model in llms:
            model.restart_model()

    for index in range(len(type_of_games)):
        print("File Opened")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/gen_coalition_results_{timestamp}.csv"
        
        with open("config/" + file_names[index]) as config_file:
            print("File Opened")
            game_configurations = csv.DictReader(config_file)

            for game_config in game_configurations:
                for round in range(int(game_config['simulate_rounds'])):
                    print(round+1)
                    curr_game = type_of_games[index](game_config, llms=llms, csv_file=output_file)
                    curr_game.simulate_game()
                    reset_llms()
