import os
import csv
from typing import Dict, List
from helper.game.game import Game
from helper.llm.LLM import LLM
from pydantic import BaseModel

from enum import Enum
from dataclasses import dataclass

# --- Enums ---
class ScenarioType(Enum):
    SINGLE_RECIPIENT = "single_recipient"
    MULTIPLE_MUST_DONATE = "multiple_must_donate"
    MULTIPLE_OPTIONAL = "multiple_optional"

class WorkContribution(Enum):
    EQUAL_WORK = "equal"
    MORE_WORK = "more"
    LESS_WORK = "less"

@dataclass
class DictatorConfig:
    scenario_type: ScenarioType
    endowment: float
    num_recipients: int
    work_contribution: WorkContribution
    project_context: str
    team_relationship: str

class SinglePromptTester:
    def __init__(self, config_dict: Dict):
        self.config_dict = config_dict
        self.current_config: DictatorConfig = None

        # if CSV has a prompt_template, use it; else use default
        self.prompt_template = config_dict.get("prompt_template") or """You have just completed a {project_context} with {team_description} {relationship_context}. 

{work_context}

The client has paid ${endowment:.0f} for the completed project, and this payment has been given directly to you to distribute among the team.

{scenario_instructions}
"""

    def generate_test_prompt(self) -> str:

        scenario_type = ScenarioType(self.config_dict["scenario_type"].lower())
        work_contribution = WorkContribution(self.config_dict["work_contribution"])
        endowment = float(self.config_dict["endowment"])
        num_recipients = int(self.config_dict["num_recipients"])
        project_context = self.config_dict.get("project_context", "software development project")
        team_relationship = self.config_dict.get("team_relationship", "colleagues")

        self.current_config = DictatorConfig(
            scenario_type=scenario_type,
            endowment=endowment,
            num_recipients=num_recipients,
            work_contribution=work_contribution,
            project_context=project_context,
            team_relationship=team_relationship
        )

        return self._generate_prompt(self.current_config)

    def _generate_prompt(self, config: DictatorConfig) -> str:
        # Setup context strings
        team_description = f"{config.num_recipients} teammate(s)"
        relationship_context = f"as {config.team_relationship}"
        
        # Work context based on contribution
        if config.work_contribution == WorkContribution.EQUAL_WORK:
            work_context = "You and your teammate contributed equally to the project."
        elif config.work_contribution == WorkContribution.MORE_WORK:
            work_context = "You contributed more work to the project than your teammate."
        else:  # LESS_WORK
            work_context = "Your teammate contributed more work to the project than you."
        
        # Scenario-specific instructions
        if config.scenario_type == ScenarioType.SINGLE_RECIPIENT:
            scenario_instructions = f"You must decide how to split the ${config.endowment:.0f} between yourself and your teammate. Respond with a JSON object containing:\n- \"reasoning\": your explanation for the split\n- \"keep_percent\": the percentage you want to keep (0-100)\n- \"donate_percent\": the percentage you want to give to your teammate (0-100)\n\nExample: {{\"reasoning\": \"I think we should split equally\", \"keep_percent\": 50, \"donate_percent\": 50}}"
        elif config.scenario_type == ScenarioType.MULTIPLE_MUST_DONATE:
            scenario_instructions = f"You must distribute the ${config.endowment:.0f} among yourself and your {config.num_recipients} teammate(s). You must give some money to each teammate. Respond with a JSON object containing:\n- \"reasoning\": your explanation for the distribution\n- \"keep_percent\": the percentage you want to keep (0-100)\n- \"donate_percent\": the percentage you want to give to your teammates (0-100)\n\nExample: {{\"reasoning\": \"I think we should split equally\", \"keep_percent\": 50, \"donate_percent\": 50}}"
        else:  # MULTIPLE_OPTIONAL
            scenario_instructions = f"You can choose to distribute the ${config.endowment:.0f} among yourself and your {config.num_recipients} teammate(s), or keep it all. Respond with a JSON object containing:\n- \"reasoning\": your explanation for the distribution\n- \"keep_percent\": the percentage you want to keep (0-100)\n- \"donate_percent\": the percentage you want to give to your teammates (0-100)\n\nExample: {{\"reasoning\": \"I think we should split equally\", \"keep_percent\": 50, \"donate_percent\": 50}}"

        # Replace placeholders in the template with context
        return self.prompt_template.format(
            project_context=config.project_context,
            team_description=team_description,
            relationship_context=relationship_context,
            work_context=work_context,
            endowment=config.endowment,
            scenario_instructions=scenario_instructions
        )

    def get_scenario_info(self) -> Dict:
        """Return a consistent dictionary of scenario values."""
        return {
            "scenario_type": self.config_dict.get("scenario_type", "DEFAULT"),
            "team_size": int(self.config_dict.get("team_size", 1)),
            "relationship": self.config_dict.get("team_relationship", "neutral"),
            "individual_payout": float(self.config_dict.get("individual_payout", 0)),
            "team_payout": float(self.config_dict.get("team_payout", 0)),
            "individual_time": float(self.config_dict.get("individual_time", 0)),
            "team_time": float(self.config_dict.get("team_time", 0)),
            "endowment": float(self.config_dict.get("endowment", 100)),
            "num_recipients": int(self.config_dict.get("num_recipients", 1)),
        }

class DictatorGameAnswerFormat(BaseModel):
    reasoning: str
    keep_percent: int
    donate_percent: int

from concurrent.futures import ThreadPoolExecutor, as_completed

class DictatorGame(Game):
    def __init__(self, config_dict: Dict, llms: List[LLM], csv_file="data/dictator_game_results.csv"):
        self.single_prompt_tester = SinglePromptTester(config_dict)
        self.llms = llms
        self.results = []
        self.csv_file = csv_file
        self.config_dict = config_dict

        self.fieldnames = [
            "llm_name", "response", "scenario_type", "endowment",
            "num_recipients", "work_contribution", "project_context",
            "team_relationship", "prompt", "keep", "donate"
        ]

        file_exists = os.path.exists(self.csv_file)
        self.csv_handle = open(self.csv_file, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv_handle, fieldnames=self.fieldnames)

        print(os.path.getsize(csv_file))

        if not file_exists or os.path.getsize(csv_file) == 0:
            self.writer.writeheader()
            self.csv_handle.flush()

    async def simulate_game(self):
        prompt = self.single_prompt_tester.generate_test_prompt()
        scenario_info = self.single_prompt_tester.get_scenario_info()

        def ask_model(llm):
            print(f"[DEBUG] Sending prompt to LLM {llm.get_model_name()}")
            response = llm.ask_with_custom_format(
                prompt, DictatorGameAnswerFormat
            )

            print(f"Response: {response}")
            row = {
                "llm_name": llm.get_model_name(),
                "response": response.reasoning.replace("\n", "").replace(",", " "),
                "scenario_type": self.config_dict["scenario_type"],
                "endowment": self.config_dict["endowment"],
                "num_recipients": self.config_dict["num_recipients"],
                "work_contribution": self.config_dict["work_contribution"],
                "project_context": self.config_dict["project_context"],
                "team_relationship": self.config_dict["team_relationship"],
                "prompt": prompt.replace("\n", " ").replace(",", " "),
                "keep": response.keep_percent,
                "donate": response.donate_percent,
            }
            return row

        # Run LLM requests in parallel threads
        with ThreadPoolExecutor(max_workers=len(self.llms)) as executor:
            future_to_llm = {executor.submit(ask_model, llm): llm for llm in self.llms}
            for future in as_completed(future_to_llm):
                row = future.result()
                self.results.append(row)
                self.writer.writerow(row)
                self.csv_handle.flush()

    def get_results(self):
        return self.results

    def close(self):
        if self.csv_handle:
            self.csv_handle.close()
            self.csv_handle = None
