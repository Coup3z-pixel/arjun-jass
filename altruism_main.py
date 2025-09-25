import csv
from typing import Dict, Type
import asyncio
from helper.game import cost_sharing_scheduling, prisoner_dilemma
from helper.game import dictator_game
from helper.game.atomic_congestion import AtomicCongestion
from helper.game.cost_sharing_scheduling import CostSharingGame
from helper.game.dictator_game import DictatorGame, ScenarioType, SinglePromptTester
from helper.game.game import Game
from helper.game.gen_coalition_altruism import GenCoalition
from helper.game.prisoner_dilemma import PrisonersDilemma
from helper.llm.AltruismInjection import AltruismInjection
from helper.llm.LLM import LLM
from helper.game.social_context import SocialContext
from helper.game.non_atomic import NonAtomicCongestion
from helper.game.hedonic_game import HedonicGame

from dotenv import load_dotenv

def main():

    load_dotenv()

    print("Simulations have started")
    print("Currently Running games")

    """
    game_info: list[Dict: {
        "game_type": Type[Game] (The type of game which will be ran)
        "file": str (The config file stored in /config directory)
    }]
    """
    game_info = [
        {"game_type": NonAtomicCongestion, "file": "NonAtomicCongestion.csv"},
        {"game_type": CostSharingGame, "file": "CostSharingGame.csv"},
        {"game_type": DictatorGame, "file": "DictatorGame.csv"},
        {"game_type": PrisonersDilemma, "file": "PrisonnersDilemma.csv"},
        {"game_type": AtomicCongestion, "file": "AtomicCongestion.csv"},
        {"game_type": CostSharingGame, "file": "CostSharingGame.csv"},
        {"game_type": GenCoalition, "file": "GenCoalition.csv"},
        {"game_type": HedonicGame, "file": "HedonicGame.csv"}
    ]


    """
        llm_models: list[str] the different model names 
        which are dependency injection for the llm interface
    """
    llm_models: list[str] = [
        "openai/chatgpt-4o-latest",
        "openai/gpt-3.5-turbo",
        #To add in, pending
        #"openai/gpt-5-chat", 
        #"openai/gpt-oss-120b", 
        "google/gemini-2.5-flash",
        "anthropic/claude-sonnet-4",
        "meta-llama/llama-3.3-8b-instruct:free",
        "microsoft/phi-3.5-mini-128k-instruct",
        "mistralai/mixtral-8x7b-instruct",
        "qwen/qwen3-14b"
    ]

    llms: list[LLM] = []

    for model in llm_models:
        llms.append(AltruismInjection(model))

    def reset_llms():
        for model in llms:
            model.restart_model()

    for info in game_info:
        with open("config/" + info["file"]) as config_file:
            print("Config File Opened")
            game_configurations = csv.DictReader(config_file)

            for game_config in game_configurations:
                for round in range(int(game_config['simulate_rounds'])):
                    print(round+1)
                    curr_game = info["game_type"](game_config, llms=llms)
                    asyncio.run(curr_game.simulate_game())
                    reset_llms()

if __name__ == "__main__":
    main()
