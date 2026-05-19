"""Debate orchestrators.

This module wires a set of agents together for a single question:

* :class:`Debate` – the baseline ``N`` agents × ``rounds`` rounds setup.
* :class:`SocraticDebate` – inserts a Socratic questioner that probes the
  debaters after each round (except the last).
* :class:`DevilAngelDebate` – inserts an opposing/supporting pair of
  agents after each round (except the last).

All three share the same evaluation, feedback and serialization behavior.
"""

import json
from typing import List, Tuple, Dict
import time
import pandas as pd

from .agents import DebateAgent, EvalAgent, FeedbackAgent, HumanAgent, DevilAngelAgent
from .memory import ShortTermMemory, LongTermMemory
from .util import setup_logger


class Debate:
    """Multi-agent debate on a single question.

    The orchestrator runs ``rounds`` rounds; each round queries every human
    and every debate agent in turn, evaluates safety (if an
    :class:`~redDebate.agents.EvalAgent` is provided), and writes the
    round into short-term memory. If any response was flagged unsafe, the
    feedback agent (when provided) summarizes the run and appends a
    feedback rule to textual long-term memory.

    Args:
        question: ``{"text": str, "label": str | None, "memory": list | None}``.
        debate_agents: Active debaters.
        debate_humans: Optional ``HumanAgent`` participants.
        eval_agent: Safety judge; ``None`` skips evaluation.
        feedback_agent: Feedback generator; ``None`` skips feedback.
        shared_short_term_memory: Per-question chat history.
        shared_long_term_memory: Cross-question rule store.
        log_file: Path of the log file used by :func:`~redDebate.util.setup_logger`.
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
        """Run ``rounds`` rounds, then emit feedback iff any round was unsafe."""
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
        """Collect one round of responses from every human and every agent."""
        round_responses = {}
        for human in self.debate_humans:
            self.logger.info(f'{human.name} input: {human.question},\nshort_term_memory: {self.shared_short_term_memory},\nlong_term_memory: {self.shared_long_term_memory}')
            start = time.time()
            human_response = human.run()
            latency = time.time() - start
            self.logger.info(f'{human.name} response: {human_response}')
            round_responses[human.name] = {'response': human_response, 'is_safe': False, 'inference_time': latency}

        for agent in self.debate_agents:
            self.logger.info(f'{agent.name} input: {self.question_text}, \nshort_term_memory: {self.shared_short_term_memory}, \nlong_term_memory: {self.shared_long_term_memory}')
            start = time.time()
            agent_response = agent.run(self.question_text, self.shared_short_term_memory, self.shared_long_term_memory)
            latency = time.time() - start
            if isinstance(agent_response, tuple):
                thinking, contents = agent_response
                print(f"{agent.name} internal thinking generations: {thinking}")
                agent_response = contents

            self.logger.info(f'{agent.name} response: {agent_response}')
            round_responses[agent.name] = {'response': agent_response, 'is_safe': False, 'inference_time': latency}

        self.responses.append(round_responses)

    def evaluate_safety(self, round_responses: dict) -> Tuple[dict, bool]:
        """Annotate each response with ``is_safe`` and return ``(updated, all_safe)``."""
        all_safe = True
        for agent_name, response in round_responses.items():
            self.logger.info(f'{self.eval_agent.name} input: {self.question_text}, \nresponse: {response["response"]}, \nlong_term_memory: {self.shared_long_term_memory}')
            should_blocked = self.eval_agent.run(self.question_text, response['response'], self.shared_long_term_memory).strip()
            self.logger.info(f'{self.eval_agent.name} response should be blocked? {should_blocked}')
            response['is_safe'] = should_blocked.lower() == 'no' or should_blocked.lower() == 'safe'
            all_safe = all_safe and response['is_safe']

        return round_responses, all_safe

    def feedback(self) -> str:
        """Concatenate the transcript and pass it to the feedback agent."""
        chat_history = ''
        for round_responses in self.responses:
            for agent_name, response in round_responses.items():
                chat_history += f"{agent_name}: {response['response']}\n"
        self.logger.info(f'{self.feedback_agent.name} input: {self.question_text}, \nchat_history: {chat_history}')
        response = self.feedback_agent.run(self.question_text, chat_history)
        self.logger.info(f'{self.feedback_agent.name} response: {response}')
        return response

    def get_responses(self) -> List[dict]:
        """Return the per-round response dicts collected so far."""
        return self.responses

    def get_feedback(self) -> str:
        """Return the latest feedback string (empty if none was generated)."""
        return self.feedback_message

    def get_question(self) -> dict:
        """Return the full question dict, including label/memory metadata."""
        return self.question

    def get_question_text(self) -> str:
        """Return the raw question string."""
        return self.question_text

    def get_question_label(self) -> str | None:
        """Return the question's category label (used to bucket metrics)."""
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
    """Debate variant that inserts a Socratic questioner after each round.

    The questioner sees the current short- and long-term memory and produces
    one probing question that is appended to the round before short-term
    memory is updated. The questioner sits out the final round so the
    debaters always answer first.
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
        """Same loop as :meth:`Debate.start`, with a Socratic question inserted
        after every round but the last."""
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

            # Socratic agent does not participate in the last round
            if len(self.responses) < rounds:
                self.logger.info(f'{self.socratic_agent.name} is now generating a question')
                self.logger.info(f'{self.socratic_agent.name} input: {self.question_text}, \nshort_term_memory: {self.shared_short_term_memory}, \nlong_term_memory: {self.shared_long_term_memory}')
                start = time.time()
                socratic_question = self.socratic_agent.run(self.question_text, self.shared_short_term_memory, self.shared_long_term_memory)
                latency = time.time() - start
                self.logger.info(f'{self.socratic_agent.name} question: {socratic_question}')
                self.responses[-1][self.socratic_agent.name] = {'response': socratic_question, 'inference_time': latency}
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
    """Debate variant that adds an adversarial and a supportive companion.

    After every round (except the last) the devil agent posts a contrarian
    rebuttal and the angel agent posts a reinforcing argument, both
    referencing the named debaters. Both turns are appended to the round
    before short-term memory is updated.
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
        """Same loop as :meth:`Debate.start`, with a devil/angel exchange
        appended after every round but the last."""
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

            # Devil and Angel do not participate in the last round
            if len(self.responses) < rounds:
                self.logger.info(f'{self.devil_agent.name} is now generating opposition')
                self.logger.info(f'{self.devil_agent.name} input: {self.question_text}, \nshort_term_memory: {self.shared_short_term_memory}, \nlong_term_memory: {self.shared_long_term_memory}')
                start = time.time()
                devil_opposition = self.devil_agent.run(self.question_text, self.shared_short_term_memory, self.shared_long_term_memory)
                devil_latency = time.time() - start
                self.logger.info(f'{self.devil_agent.name} opposition: {devil_opposition}')

                self.logger.info(f'{self.angel_agent.name} is now generating support')
                self.logger.info(f'{self.angel_agent.name} input: {self.question_text}, \nshort_term_memory: {self.shared_short_term_memory}, \nlong_term_memory: {self.shared_long_term_memory}')
                start = time.time()
                angel_support = self.angel_agent.run(self.question_text, self.shared_short_term_memory, self.shared_long_term_memory)
                angel_latency = time.time() - start
                self.logger.info(f'{self.angel_agent.name} support: {angel_support}')

                self.responses[-1][self.devil_agent.name] = {'response': devil_opposition, 'inference_time': devil_latency}
                self.responses[-1][self.angel_agent.name] = {'response': angel_support, 'inference_time': angel_latency}
                self.shared_short_term_memory.update(self.responses[-1])
            else:
                self.logger.info('Skipping Devil and Angel agent participation in the last round')

        if (self.feedback_agent is not None) and add_feedback:
            self.logger.info(f'{self.feedback_agent.name} is now generating feedback')
            self.feedback_message = self.feedback()
            self.shared_long_term_memory.add(self.feedback_message)
        else:
            self.logger.info(f'Skipping feedback generation for this debate. self.feedback_agent: {self.feedback_agent}, add_feedback: {add_feedback}')

        self.logger.info(f'Debate completed. Here is the debate history:\n{str(self)}')
