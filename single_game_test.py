import csv
from typing import Type
from datetime import datetime
import time
import asyncio

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


async def main():
    type_of_games: list[Type[Game]] = [
            #PrisonersDilemma,
            HedonicGame,
            #AtomicCongestion,
            #SocialContext,
            #NonAtomicCongestion,
            #CostSharingGame,
            #DictatorGame,
            GenCoalition,
    ]

    file_names: list[str] = [
            #"PrisonnersDilemma.csv",
            "HedonicGame.csv",
            #"AtomicCongestion.csv",
            #"SocialContext.csv",
            #"NonAtomicCongestion.csv",
            #"CostSharingGame.csv",
            #"DictatorGame.csv",
            "GenCoalition.csv"
    ]


    llm_models: list[str] = [
        #"openai/chatgpt-4o-latest",
        #"openai/gpt-3.5-turbo",
        #"google/gemini-2.5-flash",
        #"anthropic/claude-sonnet-4",
        #"deepseek/deepseek-r1-0528-qwen3-8b",
        #"meta-llama/llama-4-scout",
        #"meta-llama/llama-3.3-70b-instruct",
        #"microsoft/phi-3.5-mini-128k-instruct",
        #"ft:gpt-3.5-turbo-1106:personal::CH9gv0W1",
        #"ft:gpt-4o-2024-08-06:personal::CH9tQaMU",
        "together:jaspertsh08_a03a/Llama-3.3-70B-Instruct-Reference-8b98da31-79f56385"
        #"together:jaspertsh08_a03a/Llama-4-Scout-17B-16E-70144409-1afad6eb"
    ]

    llms: list[LLM] = []

    for model in llm_models:
        llms.append(LLM(model)) 



    def reset_llms():
        for model in llms:
            model.restart_model()

    # Games that use 'csv_save' parameter instead of 'csv_file'
    csv_save_games = {PrisonersDilemma, AtomicCongestion}
    
    for index in range(len(type_of_games)):
        game_class = type_of_games[index]
        game_name = game_class.__name__
        config_file_name = file_names[index]
        
        print(f"Starting {game_name} with config {config_file_name}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/togetherai_SFT_{game_name.lower()}_results_{timestamp}.csv"
        
        with open("config/" + config_file_name) as config_file:
            print("File Opened")
            game_configurations = csv.DictReader(config_file)

            for game_config in game_configurations:
                for round in range(int(game_config['simulate_rounds'])):
                    print(f"{game_name} - Round {round+1}")
                    
                    # Use correct parameter name based on game type
                    if game_class in csv_save_games:
                        curr_game = type_of_games[index](game_config, llms=llms, csv_save=output_file)
                    else:
                        curr_game = type_of_games[index](game_config, llms=llms, csv_file=output_file)
                    
                    if asyncio.iscoroutinefunction(curr_game.simulate_game):
                        await curr_game.simulate_game()
                    else:
                        curr_game.simulate_game()
                    reset_llms()
                    await asyncio.sleep(2)  # Add 2 second delay between rounds to avoid rate limiting


if __name__ == "__main__":
    asyncio.run(main())
