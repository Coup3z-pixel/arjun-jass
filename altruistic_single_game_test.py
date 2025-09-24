import asyncio
import csv
import pandas as pd
from typing import Type
from datetime import datetime

from helper.game.atomic_congestion import AtomicCongestion
from helper.game.cost_sharing_scheduling import CostSharingGame
from helper.game.dictator_game import DictatorGame
from helper.game.game import Game
from helper.game.non_atomic import NonAtomicCongestion
from helper.game.social_context import SocialContext
from helper.llm.AltruismInjection import AltruismInjection
from helper.llm.LLM import LLM
from helper.game.hedonic_game import HedonicGame
from helper.game.gen_coalition_altruism import GenCoalition

from helper.game.prisoner_dilemma import PrisonersDilemma


async def main():
    type_of_games: list[Type[Game]] = [
            #PrisonersDilemma,
            #HedonicGame,
            #GenCoalition,
             #AtomicCongestion,
             #SocialContext,
             #NonAtomicCongestion,
             CostSharingGame,
             DictatorGame,
    ]

    file_names: list[str] = [
            #"PrisonnersDilemma.csv",
            #"HedonicGame.csv",
            #"GenCoalition.csv",
             #"AtomicCongestion.csv",
             #"SocialContext.csv",
             #"NonAtomicCongestion.csv",
             "CostSharingGame.csv",
             "DictatorGame.csv"
    ]


    llm_models: list[str] = [
        #"openai/chatgpt-4o-latest",
        #"openai/gpt-3.5-turbo",
        #"google/gemini-2.5-flash",
        #"anthropic/claude-sonnet-4",
        # "deepseek/deepseek-r1-0528-qwen3-8b:free",
        #"meta-llama/llama-4-scout:free",
        #"meta-llama/llama-3.3-8b-instruct:free",
        #"microsoft/phi-3.5-mini-128k-instruct",
        #"ft:gpt-3.5-turbo-1106:personal::CH9gv0W1"
        #"together:jaspertsh08_a03a/Qwen3-14B-Base-9cde0ab7-7630b2c7",
        "mistralai/mixtral-8x7b-instruct",
        "qwen/qwen3-14b",
    ]

    llms: list[LLM] = []

    for model in llm_models:
        llms.append(AltruismInjection(model)) 

    def reset_llms():
        for model in llms:
            model.restart_model()

    # List to store all results
    all_results = []

    for index in range(len(type_of_games)):
        print("File Opened")
        with open("config/" + file_names[index]) as config_file:
            print("File Opened")
            game_configurations = csv.DictReader(config_file)

            for game_config in game_configurations:
                print(f"\n=== Game Configuration: {game_config} ===")
                for round in range(int(game_config['simulate_rounds'])):
                    print(f"\n--- Round {round+1} ---")
                    
                    # Generate filename with altruistic prefix for games that save directly
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    game_name = type_of_games[index].__name__.lower()
                    csv_filename = f"data/altruistic_injected_{game_name}_results_{timestamp}.csv"
                    
                    # Pass appropriate CSV parameter for games that save directly
                    if type_of_games[index] in [PrisonersDilemma, AtomicCongestion]:
                        curr_game = type_of_games[index](game_config, llms=llms, csv_save=csv_filename)
                    elif type_of_games[index] in [SocialContext, NonAtomicCongestion, HedonicGame, GenCoalition, CostSharingGame, DictatorGame]:
                        curr_game = type_of_games[index](game_config, llms=llms, csv_file=csv_filename)
                    else:
                        curr_game = type_of_games[index](game_config, llms=llms)
                    
                    # Properly await the async simulate_game method
                    await curr_game.simulate_game()

                    # Check if the game has get_results method
                    if hasattr(curr_game, 'get_results'):
                        results = curr_game.get_results()
                        print(f"Results for Round {round+1}:")
                        for i, result in enumerate(results):
                            # Handle different result structures from different games
                            if isinstance(result, dict) and result:  # Check if result is not empty
                                # Try to print available information
                                if 'llm_name' in result:
                                    print(f"  {result['llm_name']}:")
                                    print(f"    Action: {result.get('parsed_action', 'N/A')} (LLM chose: {result.get('llm_value', 'N/A')})")
                                    print(f"    Altruism Score: {result.get('ALTRUISM_SCORE', 'N/A')}")
                                    print(f"    Reasoning: {str(result.get('llm_reasoning', result.get('response', 'N/A')))[:100]}...")
                                    print()
                                    
                                    # Store result for CSV export (keep original format)
                                    all_results.append(result)
                                else:
                                    # Handle CostSharingGame and other games with different structure
                                    print(f"  Result {i+1}:")
                                    print(f"    Response: {str(result.get('response', 'N/A'))[:100]}...")
                                    print(f"    Scenario: {result.get('scenario_info', 'N/A')}")
                                    print()
                                    
                                    # Convert to standard format for CSV export
                                    standard_result = {
                                        'llm_name': f'LLM_{i+1}',
                                        'response': result.get('response', ''),
                                        'scenario_info': str(result.get('scenario_info', '')),
                                        'prompt': result.get('prompt', '')
                                    }
                                    all_results.append(standard_result)
                            else:
                                print(f"  Result {i+1}: Empty or invalid result")
                    else:
                        print(f"Game {type_of_games[index].__name__} does not have get_results method")
                        print("Results are saved to CSV file directly")
                    
                    reset_llms()

    # Save all results to CSV in the same format as hedonic_game_results.csv
    if all_results:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"data/altruistic_injected_altruism_test_{timestamp}.csv"
        
        # Define the exact column order to match hedonic_game_results.csv
        column_order = [
            'llm_name', 'agent', 'prompt', 'llm_value', 'llm_reasoning', 
            'parsed_action', 'selfish_action', 'u_selfish', 'u_chosen', 
            'friends_benefit_sum', 'friends_harm_sum', 'ALTRUISM_SCORE'
        ]
        
        # Create DataFrame and reorder columns
        df = pd.DataFrame(all_results)
        
        # Only reorder columns if they exist in the DataFrame
        available_columns = [col for col in column_order if col in df.columns]
        if available_columns:
            df = df.reindex(columns=available_columns)
        
        # Save to CSV
        df.to_csv(output_filename, index=False)
        print(f"\n=== RESULTS SAVED ===")
        print(f"Total results: {len(all_results)}")
        print(f"Saved to: {output_filename}")
        print(f"Format matches altruism testing framework")
        
        # Show summary statistics
        print(f"\n=== SUMMARY STATISTICS ===")
        print(f"Models tested: {df['llm_name'].nunique()}")
        print(f"Average altruism score: {df['ALTRUISM_SCORE'].mean():.4f}")
        print(f"Altruism rate: {(df['ALTRUISM_SCORE'] > 0).sum() / len(df) * 100:.1f}%")
        
        # Show first few rows to verify format
        print(f"\n=== SAMPLE OUTPUT ===")
        print(df.head(2).to_string(index=False))
    else:
        print("No results to save.")


if __name__ == "__main__":
    asyncio.run(main())
