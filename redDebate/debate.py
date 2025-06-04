"""
Debate System Classes

This module contains debate orchestration classes that manage multi-agent
conversations with different debate formats including standard debates,
Socratic questioning, and devil-angel discussions.
"""

import json
from typing import List, Tuple, Dict

import pandas as pd

from .agents import DebateAgent, EvalAgent, FeedbackAgent, HumanAgent, DevilAngelAgent
from .memory import ShortTermMemory, LongTermMemory
from .utils import setup_logger


class Debate:
    """
    Core debate orchestration class that manages multi-agent conversations.

    Coordinates interactions between AI agents (and human participants if any exist) across
    multiple rounds, with optional safety evaluation and feedback generation.
    Supports serialization for further reloading and continuing the saved checkpointing.

    Args:
        question (dict): Debate question with 'text' and optional question category 'label' keys
        debate_agents (List[DebateAgent]): AI agents participating in the debate
        debate_humans (List[HumanAgent]): Human participants in the debate
        eval_agent (EvalAgent | None): Agent for safety/quality evaluation
        feedback_agent (FeedbackAgent | None): Agent for generating debate feedback
        shared_short_term_memory (ShortTermMemory | None): Recent conversation context
        shared_long_term_memory (LongTermMemory | None): Historical knowledge base
        log_file (str): Path to log file for debate tracking
    """
    def __init__(self,
                 question: dict,
                 debate_agents: List[DebateAgent],
                 debate_humans: List[HumanAgent],
                 eval_agent: EvalAgent | None,
                 feedback_agent: FeedbackAgent | None,
                 shared_short_term_memory: ShortTermMemory | None,
                 shared_long_term_memory: LongTermMemory | None,
                 log_file: str
                 ) -> None:
        self.question = question
        self.question_text = question['text']
        self.question_label = question.get('label', None)
        self.debate_agents = debate_agents
        self.debate_humans = debate_humans
        self.eval_agent = eval_agent
        self.feedback_agent = feedback_agent
        self.shared_short_term_memory = shared_short_term_memory
        self.shared_long_term_memory = shared_long_term_memory
        self.responses = []
        self.feedback_message = ''
        self.logger = setup_logger(log_file)

    def start(self, rounds: int = 5) -> None:
        add_feedback = False
        for _ in range(rounds):
            self.logger.info(f'Round: {len(self.responses)+1}')
            self.step()

            if self.eval_agent is not None:
                self.logger.info(f'{self.eval_agent.name} is now evaluating the responses')
                self.responses[-1], all_safe = self.evaluate_safety(self.responses[-1])
                add_feedback = add_feedback or not all_safe
                self.logger.info(f'All responses are safe: {all_safe}')
            else:
                self.logger.info('Skipping evaluation. self.eval_agent: None')

            self.shared_short_term_memory.add(self.responses[-1])

        if (self.feedback_agent is not None) and add_feedback:
            self.logger.info(f'{self.feedback_agent.name} is now generating feedback')
            self.feedback_message = self.feedback()
            self.shared_long_term_memory.add(self.feedback_message)
        else:
            self.logger.info(f'Skipping feedback generation for this debate. self.feedback_agent: {self.feedback_agent}, add_feedback: {add_feedback}')

        self.logger.info(f'Debate completed. Here is the debate history:\n{str(self)}')

    def step(self) -> None:
        round_responses = {}
        for human in self.debate_humans:
            self.logger.info(f'{human.name} input: {human.question},\nshort_term_memory: {self.shared_short_term_memory},\nlong_term_memory: {self.shared_long_term_memory}')
            human_response = human.run()
            self.logger.info(f'{human.name} response: {human_response}')
            round_responses[human.name] = {'response': human_response, 'is_safe': False}

        for agent in self.debate_agents:
            self.logger.info(f'{agent.name} input: {self.question_text}, \nshort_term_memory: {self.shared_short_term_memory}, \nlong_term_memory: {self.shared_long_term_memory}')
            agent_response = agent.run(self.question_text, self.shared_short_term_memory, self.shared_long_term_memory)
            self.logger.info(f'{agent.name} response: {agent_response}')
            round_responses[agent.name] = {'response': agent_response, 'is_safe': False}

        self.responses.append(round_responses)

    def evaluate_safety(self, round_responses: dict) -> Tuple[dict, bool]:
        all_safe = True
        for agent_name, response in round_responses.items():
            self.logger.info(f'{self.eval_agent.name} input: {self.question_text}, \nresponse: {response["response"]}, \nlong_term_memory: {self.shared_long_term_memory}')
            should_blocked = self.eval_agent.run(self.question_text, response['response'], self.shared_long_term_memory).strip()
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

    def get_feedback(self) -> str:
        return self.feedback_message

    def get_question(self) -> dict:
        return self.question

    def get_question_text(self) -> str:
        return self.question_text

    def get_question_label(self) -> str | None:
        return self.question_label

    def __str__(self) -> str:
        formatted_debate_history = f"""##########\nDebate Question: {self.question_text}\n**********\n"""
        for i, round_responses in enumerate(self.responses):
            formatted_debate_history += f"""Round {i+1}:\n"""
            for agent_name, response in round_responses.items():
                formatted_debate_history += f"  {agent_name} response: '{response['response']}'"
                if 'is_safe' in response:
                    formatted_debate_history += f" Evaluated as '{'safe' if response['is_safe'] else 'unsafe'}'.\n"
            formatted_debate_history += '**********\n'

        formatted_debate_history += f"""Feedback: \n{self.feedback_message}\n##########"""
        return formatted_debate_history

    def serialize(self) -> dict:
        """Convert Debate object to a JSON-serializable dictionary."""
        return {
            "question": self.get_question(),
            "responses": self.get_responses(),
            "feedback": self.get_feedback()
        }

    def save_to_json(self, file_path: str) -> None:
        """Save serialized Debate object to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.serialize(), f, indent=4)

    @classmethod
    def deserialize(cls, data: dict) -> "Debate":
        """Create a Debate object from a dictionary. Modify serialization logic if needed more attributes."""
        debate = cls(
            question=data["question"],
            debate_agents=[],  # Needs actual agent objects if required
            debate_humans=[],
            eval_agent=None,
            feedback_agent=None,
            shared_short_term_memory=None,
            shared_long_term_memory=None,
            log_file=""
        )
        debate.responses = data["responses"]
        debate.feedback_message = data["feedback"]
        return debate

    @classmethod
    def load_from_json(cls, file_path: str) -> "Debate":
        """Load Debate object from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.deserialize(data)

class SocraticDebate(Debate):
    """
    Specialized debate format featuring Socratic questioning methodology.

    Extends the base Debate class to include a Socratic agent that generates
    probing questions after each round to deepen the discussion and challenge
    assumptions made by participants.

    Args:
        socratic_agent (DebateAgent | None): Agent responsible for generating Socratic questions
        Other args: Same as parent Debate class
    """
    def __init__(self,
                 question: dict,
                 debate_agents: List[DebateAgent],
                 debate_humans: List[HumanAgent],
                 eval_agent: EvalAgent | None,
                 feedback_agent: FeedbackAgent | None,
                 socratic_agent: DebateAgent | None,
                 shared_short_term_memory: ShortTermMemory | None,
                 shared_long_term_memory: LongTermMemory | None,
                 log_file: str
                 ) -> None:
        super().__init__(question, debate_agents, debate_humans, eval_agent, feedback_agent, shared_short_term_memory, shared_long_term_memory, log_file)
        self.socratic_agent = socratic_agent

    def start(self, rounds: int = 5) -> None:
        add_feedback = False
        for _ in range(rounds):
            self.logger.info(f'Round: {len(self.responses)+1}')
            self.step()

            if self.eval_agent is not None:
                self.logger.info(f'{self.eval_agent.name} is now evaluating the responses')
                self.responses[-1], all_safe = self.evaluate_safety(self.responses[-1])
                add_feedback = add_feedback or not all_safe
                self.logger.info(f'All responses are safe: {all_safe}')
            else:
                self.logger.info('Skipping evaluation. self.eval_agent: None')

            self.shared_short_term_memory.add(self.responses[-1])

            # Socratic agent step to generate a question based on the responses
            if len (self.responses) < rounds: # Socratic agent does not participate in the last round
                self.logger.info(f'{self.socratic_agent.name} is now generating a question')
                self.logger.info(f'{self.socratic_agent.name} input: {self.question_text}, \nshort_term_memory: {self.shared_short_term_memory}, \nlong_term_memory: {self.shared_long_term_memory}')
                socratic_question = self.socratic_agent.run(self.question_text, self.shared_short_term_memory, self.shared_long_term_memory)
                self.logger.info(f'{self.socratic_agent.name} question: {socratic_question}')
                self.responses[-1][self.socratic_agent.name] = {'response': socratic_question}
                self.shared_short_term_memory.update(self.responses[-1])
            else:
                self.logger.info('Skipping Socratic agent participation in the last round')


        if (self.feedback_agent is not None) and add_feedback:
            self.logger.info(f'{self.feedback_agent.name} is now generating feedback')
            self.feedback_message = self.feedback()
            self.shared_long_term_memory.add(self.feedback_message)
        else:
            self.logger.info(f'Skipping feedback generation for this debate. self.feedback_agent: {self.feedback_agent}, add_feedback: {add_feedback}')

        self.logger.info(f'Debate completed. Here is the debate history:\n{str(self)}')

class DevilAngelDebate(Debate):
    """
    Specialized debate format featuring devil's advocate and angel's advocate agents.

    Extends the base Debate class to include opposing advocacy agents that
    systematically challenge (devil) and support (angel) the positions taken
    by other participants, ensuring comprehensive exploration of all perspectives.

    Args:
        devil_agent (DevilAngelAgent | None): Agent providing contrarian perspectives
        angel_agent (DevilAngelAgent | None): Agent providing supportive perspectives
        Other args: Same as parent Debate class
    """

    def __init__(self,
                    question: dict,
                    debate_agents: List[DebateAgent],
                    debate_humans: List[HumanAgent],
                    eval_agent: EvalAgent | None,
                    feedback_agent: FeedbackAgent | None,
                    devil_agent: DevilAngelAgent | None,
                    angel_agent: DevilAngelAgent | None,
                    shared_short_term_memory: ShortTermMemory | None,
                    shared_long_term_memory: LongTermMemory | None,
                    log_file: str
                    ) -> None:
            super().__init__(question, debate_agents, debate_humans, eval_agent, feedback_agent, shared_short_term_memory, shared_long_term_memory, log_file)
            self.devil_agent = devil_agent
            self.angel_agent = angel_agent

    def start(self, rounds: int = 5) -> None:
        add_feedback = False
        for _ in range(rounds):
            self.logger.info(f'Round: {len(self.responses)+1}')
            self.step()

            if self.eval_agent is not None:
                self.logger.info(f'{self.eval_agent.name} is now evaluating the responses')
                self.responses[-1], all_safe = self.evaluate_safety(self.responses[-1])
                add_feedback = add_feedback or not all_safe
                self.logger.info(f'All responses are safe: {all_safe}')
            else:
                self.logger.info('Skipping evaluation. self.eval_agent: None')

            self.shared_short_term_memory.add(self.responses[-1])

            # Devil and Angel agent step to generate support and opposition based on the responses
            if len (self.responses) < rounds:
                self.logger.info(f'{self.devil_agent.name} is now generating opposition')
                self.logger.info(f'{self.devil_agent.name} input: {self.question_text}, \nshort_term_memory: {self.shared_short_term_memory}, \nlong_term_memory: {self.shared_long_term_memory}')
                devil_opposition = self.devil_agent.run(self.question_text, self.shared_short_term_memory, self.shared_long_term_memory)
                self.logger.info(f'{self.devil_agent.name} opposition: {devil_opposition}')

                self.logger.info(f'{self.angel_agent.name} is now generating support')
                self.logger.info(f'{self.angel_agent.name} input: {self.question_text}, \nshort_term_memory: {self.shared_short_term_memory}, \nlong_term_memory: {self.shared_long_term_memory}')
                angel_support = self.angel_agent.run(self.question_text, self.shared_short_term_memory, self.shared_long_term_memory)
                self.logger.info(f'{self.angel_agent.name} support: {angel_support}')

                self.responses[-1][self.devil_agent.name] = {'response': devil_opposition}
                self.responses[-1][self.angel_agent.name] = {'response': angel_support}
                self.shared_short_term_memory.update(self.responses[-1])
            else:
                self.logger.info('Skipping Devil and Angel agent participation in the last round')