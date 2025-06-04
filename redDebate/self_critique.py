import json
import random
from typing import List

from .agents import SelfCriticAgent, EvalAgent
from .utils import setup_logger


class SelfCritique:
    """
    Constitutional AI self-improvement system that iteratively critiques and revises responses.

    Uses constitutional rules to guide an AI agent through multiple rounds of self-critique,
    with safety evaluation. Supports serialization for further reloading and continuing the saved checkpointing.


    Attributes:
        question (dict): The input question with metadata
        critic_agent (SelfCriticAgent): Agent that performs critique and revision
        eval_agent (EvalAgent): Optional agent for safety evaluation
        responses (List[dict]): History of all critique rounds
        constitution_rules (dict): Rules loaded from Constitutional Harmlessness Paper
    """
    def __init__(self,
                 question: dict,
                 critic_agent: SelfCriticAgent | None,
                 eval_agent: EvalAgent | None,
                 log_file: str
                 ) -> None:
        self.question = question
        self.question_text = question['text']
        self.question_label = question.get('label', None)
        self.critic_agent = critic_agent
        self.eval_agent = eval_agent
        self.responses = []
        self.logger = setup_logger(log_file)

        # download this file if not exists: https://github.com/anthropics/ConstitutionalHarmlessnessPaper/blob/main/prompts/CritiqueRevisionInstructions.json
        with open('./RedDebate/ConstitutionalHarmlessnessPaper/prompts/CritiqueRevisionInstructions.json', 'r') as f:
            self.constitution_rules = json.load(f)


    def start(self, rounds: int = 1) -> None:
        for i in range(rounds):
            self.logger.info(f'Round: {len(self.responses)+1}')

            constitution_rule = random.choice(list(self.constitution_rules.keys()))
            self.logger.info(f'Constitution rule selected: {constitution_rule}')
            self.logger.info(f'{self.critic_agent.name} is now responding')

            if i == 0:
                response = self.critic_agent.run(
                    self.question_text,
                    self.constitution_rules[constitution_rule]['prompt'][0].strip(),
                    self.constitution_rules[constitution_rule]['edit_request'],
                    init_response = ''
                )
            else:
                self.logger.info(f'Using the response from the last round as the initial response.')
                response = self.critic_agent.run(
                    self.question_text,
                    self.constitution_rules[constitution_rule]['prompt'][0].strip(),
                    self.constitution_rules[constitution_rule]['edit_request'],
                    init_response = self.responses[-1][self.critic_agent.name]['response']['revised_response']['text']
                )

            self.responses.append({
                self.critic_agent.name: {
                    'response': response,
                    'constitution_rule': constitution_rule,
                }
            })

            if self.eval_agent is not None:
                self.logger.info(f'{self.eval_agent.name} is now evaluating the responses')
                self.evaluate_safety(self.responses[-1][self.critic_agent.name]['response'])
            else:
                self.logger.info('Skipping evaluation. self.eval_agent: None')

        self.logger.info(f'Self-critique completed. Here is the history:\n{str(self)}')


    def evaluate_safety(self, round_responses: dict) -> dict:
        for key, response in list(round_responses.items()):
            should_blocked = self.eval_agent.run(self.question_text, response['text'], None).strip()
            self.logger.info(f'{self.eval_agent.name} response should be blocked? {should_blocked}')
            round_responses[key]['is_safe'] = should_blocked.lower() == 'no' or should_blocked.lower() == 'safe'

        return round_responses

    def get_responses(self) -> List[dict]:
        return self.responses

    def get_question(self) -> dict:
        return self.question

    def get_question_text(self) -> str:
        return self.question_text

    def get_question_label(self) -> str | None:
        return self.question_label

    def get_constitution_rules(self) -> dict:
        return self.constitution_rules

    def get_feedback(self) -> str:
        return '' # placeholder for not getting error in metric calculation

    def __str__(self) -> str:
        formatted_history = f"""##########\nQuestion: {self.question_text}\n**********\n"""
        for i, round_responses in enumerate(self.responses):
            formatted_history += f"""Round {i+1}:\n"""
            for agent_name, response in round_responses.items():
                formatted_history += f"  {agent_name} response: '{response['response']}'"
                if 'is_safe' in response:
                    formatted_history += f" Evaluated as '{'safe' if response['is_safe'] else 'unsafe'}'.\n"
            formatted_history += '**********\n'

        return formatted_history

    def serialize(self) -> dict:
        """Convert Debate object to a JSON-serializable dictionary."""
        return {
            "question": self.get_question(),
            "responses": self.get_responses(),
            "constitution_rules": self.constitution_rules,
        }

    def save_to_json(self, file_path: str) -> None:
        """Save serialized Debate object to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.serialize(), f, indent=4)

    @classmethod
    def deserialize(cls, data: dict) -> "SelfCritique":
        """Create a SelfCritique object from a dictionary. Modify serialization logic if needed more attributes."""
        self_critique = cls(
            question=data["question"],
            critic_agent=None,  # Placeholder, should be set to a valid agent
            eval_agent=None,  # Placeholder, should be set to a valid agent
            log_file=""
        )
        self_critique.responses = data["responses"]
        return self_critique

    @classmethod
    def load_from_json(cls, file_path: str) -> "SelfCritique":
        """Load Debate object from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.deserialize(data)
