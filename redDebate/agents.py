"""
Agent Classes for Multi-Agent Debate and Evaluation System

This module contains various agent classes for implementing multi-agent debates,
evaluations, and self-criticism workflows.
"""

from openai import OpenAI
from .llm import AnyOpenAILLM, AnyHuggingFace, LlamaGuard
from .memory import ShortTermMemory, LongTermMemory

class HumanAgent:
    """
    Agent that allows human participation in debates through console input.

    This agent prompts a human user to provide responses during debate rounds,
    enabling human-AI collaborative discussions (currently for test purposes).

    Args:
        name (str): Display name for the human participant
        question_text (str): The debate question or topic
    """
    def __init__(self,
                 name: str,
                 question_text: str,
                ) -> None:
        self.name = name
        self.question = question_text
        self.debate_round = 0

    def run(self) -> str:
        self.debate_round += 1
        result = input(f'{self.name} enter your response here: ')
        return result


class DebateAgent:
    """
    AI agent that participates in structured debates using an LLM backend.

    This agent generates responses based on the debate question, current round,
    and available memory context from previous interactions.

    Args:
        name (str): Agent's identifier name
        base_llm: Language model instance for generating responses
    """
    def __init__(self,
                 name: str,
                 base_llm: AnyOpenAILLM | AnyHuggingFace,
                ) -> None:
        self.name = name
        self.base_llm = base_llm
        self.debate_round = 0

    def run(self, question_text: str, short_term_memory: ShortTermMemory, long_term_memory: LongTermMemory) -> str:
        self.debate_round += 1
        response = self.base_llm({
            "agent_name": self.name,
            "question": question_text,
            "debate_round": self.debate_round,
            "short_term_memory": str(short_term_memory),
            "long_term_memory": str(long_term_memory)
        })
        return response


class DevilAngelAgent:
    """
    Auxiliary agent that provides contrarian or supportive perspectives.

    This agent is designed to either challenge (devil's advocate) or support
    (angel's advocate) the positions of specified target agents in the debate.

    Args:
        name (str): Agent's identifier name
        base_llm: Language model instance for generating responses
        target_agents_names (list[str]): Names of agents to target with advocacy
    """
    def __init__(self,
                 name: str,
                 base_llm: AnyOpenAILLM | AnyHuggingFace,
                 target_agents_names: list[str],
                ) -> None:
        self.name = name
        self.base_llm = base_llm
        self.debate_round = 0
        self.target_agents_names = target_agents_names

    def run(self, question_text: str, short_term_memory: ShortTermMemory, long_term_memory: LongTermMemory) -> str:
        self.debate_round += 1
        response = self.base_llm({
            "agent_name": self.name,
            "question": question_text,
            "target_agents_names": " , ".join(self.target_agents_names),
            "debate_round": self.debate_round,
            "short_term_memory": str(short_term_memory),
            "long_term_memory": str(long_term_memory)
        })
        return response


class EvalAgent:
    """
    Agent responsible for evaluating and moderating other agents' responses.

    This agent can use different evaluation backends including LlamaGuard for
    safety, standard LLMs with a moderation prompt, or OpenAI's moderation API
    as a fallback safety check.

    Args:
        name (str): Agent's identifier name
        base_llm: Evaluation model (LlamaGuard, LLM, or None for OpenAI moderation)
    """
    def __init__(self,
                    name: str,
                 base_llm: AnyOpenAILLM | AnyHuggingFace | LlamaGuard | None,
                 ) -> None:
        self.name = name
        self.base_llm = base_llm

    def run(self, question_text: str, agent_response: str, long_term_memory: LongTermMemory | None) -> str:
        if isinstance(self.base_llm, LlamaGuard):
            response = self.base_llm(agent_response)
            return response
        elif isinstance(self.base_llm, AnyOpenAILLM) or isinstance(self.base_llm, AnyHuggingFace):
                response = self.base_llm({
                    "question": question_text,
                    "agent_response": agent_response,
                    "long_term_memory": str(long_term_memory)
                })
                return response
        elif self.base_llm is None:
            try:
                client = OpenAI()
                response = client.moderations.create(model="omni-moderation-latest", input=agent_response)
                if response.results[0].flagged:
                    return 'Yes'
                else:
                    return 'No'
            except Exception as e:
                print(f"Error checking safety with OpenAI moderation API: {e}")
                return 'Yes'
        else:
            raise ValueError("Invalid base_llm type")


class FeedbackAgent:
    """
    Agent that provides feedback and analysis on conversation history.

    This agent reviews the entire chat history and provides constructive feedback on safety failures.

    Args:
        name (str): Agent's identifier name
        base_llm: Language model instance for generating feedback
    """
    def __init__(self,
                 name: str,
                 base_llm: AnyOpenAILLM | AnyHuggingFace,
                ) -> None:
        self.name = name
        self.base_llm = base_llm

    def run(self, question_text: str, chat_history: str) -> str:
        response = self.base_llm({
            "question": question_text,
            "chat_history": chat_history
        })
        return response

class SelfCriticAgent:
    """
    Agent that implements constitutional AI through self-criticism and revision.

    This agent generates an initial response, critiques it against constitutional
    rules, and then revises the response based on the critique. Uses three
    separate LLMs for generation, criticism, and revision phases for the ease of implementation,
     however, all llms are given a same backend in the runtime

    Args:
        name (str): Agent's identifier name
        base_llm: LLM for initial response generation
        critic_llm: LLM for critiquing responses
        revision_llm: LLM for revising responses based on critique
    """
    def __init__(self,
                 name: str,
                 base_llm: AnyOpenAILLM | AnyHuggingFace,
                 critic_llm: AnyOpenAILLM | AnyHuggingFace,
                 revision_llm: AnyOpenAILLM | AnyHuggingFace,
                ) -> None:
        self.name = name
        self.base_llm = base_llm
        self.critic_llm = critic_llm
        self.revision_llm = revision_llm

    def run(self, question_text: str, constitutional_rule: str, revision_rule: str, init_response: str) -> dict:

        if not init_response:
            init_response = self.base_llm({
                "question": question_text,
            })
        else:
            print(f'Using the response from the last round as the initial response: {init_response}')


        critique = self.critic_llm({
            "question": question_text,
            "response": init_response,
            "constitutional_rule": constitutional_rule
        })
        revised_response = self.revision_llm({
            "question": question_text,
            "response": init_response,
            "critique": critique,
            "revision_rule": revision_rule
        })

        return {
            "initial_response": {'text': init_response},
            "critique": {'text': critique},
            "revised_response": {'text': revised_response}
        }