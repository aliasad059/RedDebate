"""Constitutional-AI style self-critique orchestrator.

A single :class:`~redDebate.agents.SelfCriticAgent` answers a question,
critiques itself against a randomly-sampled constitutional rule and then
rewrites the answer. The cycle can run for one or many rounds; later
rounds reuse the previous revision as the starting answer.
"""

import json
import random
from typing import List, Tuple
import time

from .agents import SelfCriticAgent, EvalAgent, FeedbackAgent
from .util import setup_logger
from .memory import LongTermMemory


class SelfCritique:
    """Drives one or more answer→critique→revise cycles for a single question.

    Args:
        question: ``{"text": str, "label": str | None, ...}``.
        critic_agent: The :class:`~redDebate.agents.SelfCriticAgent` doing the work.
        eval_agent: Optional safety judge (run on the **initial** response).
        feedback_agent: Optional feedback generator (writes to long-term memory).
        long_term_memory: Cross-debate rule store passed to the critic.
        log_file: Logging destination.
    """

    def __init__(self,
                 question: dict,
                 critic_agent: SelfCriticAgent | None,
                 eval_agent: EvalAgent | None,
                 feedback_agent: FeedbackAgent | None,
                 long_term_memory: LongTermMemory | None,
                 log_file: str
                 ) -> None:
        self.question = question
        self.question_text = question['text']
        self.question_label = question.get('label', None)
        self.critic_agent = critic_agent
        self.eval_agent = eval_agent
        self.feedback_agent = feedback_agent
        self.long_term_memory = long_term_memory
        self.responses = []
        self.logger = setup_logger(log_file)
        self.feedback_message = ''

        # /home/xiaodan/Desktop/RedDebate/ConstitutionalHarmlessnessPaper/prompts/CritiqueRevisionInstructions.json
        with open('/home/aliasad/projects/def-zhu2048/aliasad/RedDebate/ConstitutionalHarmlessnessPaper/prompts/CritiqueRevisionInstructions.json', 'r') as f:
            self.constitution_rules = json.load(f)


    def start(self, rounds: int = 1) -> None:
        """Run ``rounds`` answer/critique/revise iterations.

        On the first round the critic generates a fresh initial answer; on
        subsequent rounds the previous revised answer is reused as the
        starting point. After each round the initial response is judged for
        safety; if any round was unsafe and a feedback agent is provided,
        feedback is appended to long-term memory.
        """
        add_feedback = False
        for i in range(len(self.responses),rounds+(len(self.responses))):
            self.logger.info(f'Round: {i+1}')

            random_rule = random.choice(list(self.constitution_rules.keys()))
            constitution_rule = self.constitution_rules[random_rule]['prompt'][0].strip()
            revision_rule = self.constitution_rules[random_rule]['edit_request']

            self.logger.info(f'Constitution rule selected: {constitution_rule}')
            self.logger.info(f'{self.critic_agent.name} is now responding')

            start = time.time()
            if i == 0:
                response = self.critic_agent.run(
                    self.question_text,
                    constitution_rule,
                    revision_rule,
                    str(self.long_term_memory),
                    init_response = ''
                )
            else:
                self.logger.info(f'Using the response from the last round as the initial response.')
                response = self.critic_agent.run(
                    self.question_text,
                    constitution_rule,
                    revision_rule,
                    str(self.long_term_memory),
                    init_response = self.responses[-1][self.critic_agent.name]['response']['revised_response']['text']
                )
            latency = time.time() - start

            self.responses.append({
                self.critic_agent.name: {
                    'response': response,
                    'inference_time': latency,
                    'constitution_rule': constitution_rule,
                }
            })

            if self.eval_agent is not None:
                self.logger.info(f'{self.eval_agent.name} is now evaluating the responses')
                self.responses[-1], all_safe = self.evaluate_safety(self.responses[-1])
                add_feedback = add_feedback or not all_safe

            else:
                self.logger.info('Skipping evaluation. self.eval_agent: None')

            if (self.feedback_agent is not None) and add_feedback:
                self.logger.info(f'{self.feedback_agent.name} is now generating feedback')
                self.feedback_message = self.feedback()
                self.long_term_memory.add(self.feedback_message)
            else:
                self.logger.info(
                    f'Skipping feedback generation for this debate. self.feedback_agent: {self.feedback_agent}, add_feedback: {add_feedback}')

        self.logger.info(f'Self-critique completed. Here is the history:\n{str(self)}')


    def evaluate_safety(self, round_responses: dict) -> Tuple[dict, bool]:
        all_safe = True
        self.logger.info(round_responses)
        for agent_name, response in round_responses.items():
            should_blocked = self.eval_agent.run(self.question_text, response['response']['initial_response']['text'], self.long_term_memory).strip()
            self.logger.info(f'{self.eval_agent.name} response should be blocked? {should_blocked}')
            response['is_safe'] = should_blocked.lower() == 'no' or should_blocked.lower() == 'safe'
            all_safe = all_safe and response['is_safe']
        return round_responses, all_safe


    def feedback(self) -> str:
        chat_history = ''
        for round_responses in self.responses:
            for agent_name, response in round_responses.items():
                chat_history += f"{agent_name}: {response['response']}\n"
        self.logger.info(f'{self.feedback_agent.name} input: {self.question_text}, \nchat_history: {chat_history}')
        response = self.feedback_agent.run(self.question_text, chat_history)
        self.logger.info(f'{self.feedback_agent.name} response: {response}')
        return response

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
        return self.feedback_message

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
            "feedback": self.get_feedback(),
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
            feedback_agent=None,
            long_term_memory=None,
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
