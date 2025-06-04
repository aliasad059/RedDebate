import os
import shutil

from dotenv import load_dotenv
from datetime import datetime
import json
import subprocess
from pathlib import Path

from .debate import Debate, DevilAngelDebate, SocraticDebate
from .self_critique import SelfCritique
from .agents import DebateAgent, DevilAngelAgent, EvalAgent, FeedbackAgent, HumanAgent, SelfCriticAgent
from .memory import ShortTermMemory, LongTermMemory, VectorStoreMemory
from .llm import AnyHuggingFace, AnyOpenAILLM, LlamaGuard, AnyGoogleGenerativeAI
from .debate_prompts import debate_agent_prompt, devil_debater_prompt, angel_debater_prompt , feedback_prmpt, socratic_agent_prompt, eval_prmpt, init_response_prompt, self_critique_prompt, revise_response_prompt
from .dataloader import load_datasets
from .utils import setup_logger
from .metrics import calculate_debate_metrics, log_results_to_wandb


load_dotenv()


def save_debate_log(debate, checkpoint_folder: str, dataset_name: str, question_idx: int):
    os.makedirs(checkpoint_folder, exist_ok=True)
    debate_log_path = os.path.join(checkpoint_folder, f"{dataset_name}_q{question_idx}.json")

    debate.save_to_json(debate_log_path)

def save_checkpoint(checkpoint_folder: str, completed_debates: dict):
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder, "checkpoint.json")

    with open(checkpoint_path, "w") as f:
        json.dump(completed_debates, f, indent=4)

def load_checkpoint(checkpoint_folder: str):
    checkpoint_path = os.path.join(checkpoint_folder, "checkpoint.json")

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return {}

def load_saved_debates(checkpoint_folder: str, conversation_type: str ='debate'):
    debates = []
    if not os.path.exists(checkpoint_folder):
        return debates  # No checkpoint to load

    if conversation_type == 'debate':
        for file in Path(checkpoint_folder).glob("*.json"):
            if file.name != "checkpoint.json":  # Skip checkpoint file
                debates.append(Debate.load_from_json(str(file)))
    elif conversation_type == 'selfcritique':
        for file in Path(checkpoint_folder).glob("*.json"):
            if file.name != "checkpoint.json":
                debates.append(SelfCritique.load_from_json(str(file)))
    else:
        print(f"Unknown conversation type: {conversation_type}. Skipping loading.")

    return debates


def init_agents(debater_models, devil_model, angel_model, evaluator_model, feedback_generator, questioner_model, self_critique_model, logger):

    # Cache for loaded LLMs to avoid reloading the same model multiple times.
    loaded_llms = {}

    debate_agents = []
    for i, model in enumerate(debater_models):
        model_type, model_name, use_chat = model.split(':')
        # Check if we already loaded this model; if so, reuse it.
        if model in loaded_llms: # TODO: needs fixing, currently loads each model separately from scratch
            logger.info(f"Reusing previously loaded model '{model}' as Agent-{i}")
            llm = loaded_llms[model]
        else:
            # Otherwise, load a new model and store it in the cache.
            if model_type == 'huggingface':
                logger.info(f"Loading HuggingFace model: '{model_name}' as Agent-{i}")
                llm = AnyHuggingFace(
                    model_name=model_name,
                    use_chat=use_chat,
                    prompt_template=debate_agent_prompt,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    device=f'cuda:{i}',
                    max_new_tokens=512,
                    truncation=True,
                    return_full_text=False
                )
            elif model_type == 'openai':
                logger.info(f"Loading OpenAI model: '{model_name}' as Agent-{i}")
                llm = AnyOpenAILLM(
                    model_name=model_name,
                    use_chat=use_chat,
                    prompt_template=debate_agent_prompt,
                    temperature=0.7,
                    top_p=0.9,
                    max_retries=5,
                )
            elif model_type == 'deepseek':
                logger.info(f"Loading OpenAI model: '{model_name}' as Agent-{i}")
                llm = AnyOpenAILLM(
                    model_name=model_name,
                    use_chat=use_chat,
                    prompt_template=debate_agent_prompt,
                    openai_api_base='https://api.deepseek.com',
                    openai_api_key=os.environ.get('DEEPSEEK_API_KEY'),
                    temperature=0.7,
                    top_p=0.9,
                    max_retries=5
                )
            elif model_type == 'google':
                logger.info(f"Loading Google model: '{model_name}' as Agent-{i}")
                llm = AnyGoogleGenerativeAI(
                    model_name=model_name,
                    use_chat=use_chat,
                    prompt_template=debate_agent_prompt,
                    temperature=0.7,
                    top_p=0.9,
                )
            else:
                raise ValueError(
                    f"Model type '{model_type}' not supported for debate. "
                    "Please choose from 'huggingface', 'openai', or 'google'."
                )
            # loaded_llms[model] = llm

        debater = DebateAgent(name=f'Agent-{i}', base_llm=llm)
        debate_agents.append(debater)

    # Devil and Angel agents
    devil_agent = None
    angel_agent = None
    if devil_model is not None and angel_model is not None:
        devil_model_type, devil_model_name, devil_use_chat = devil_model.split(':')
        angel_model_type, angel_model_name, angel_use_chat = angel_model.split(':')
        target_agents_names = [d.name for d in debate_agents]

        if devil_model in loaded_llms:
            logger.info(f"Reusing previously loaded {devil_model_type} model: '{devil_model_name}' as Agent-X")
            devil_llm = loaded_llms[devil_model]
        else:
            if devil_model_type == 'huggingface':
                logger.info(f"Loading HuggingFace model: '{devil_model_name}' as Agent-X")
                devil_llm = AnyHuggingFace(
                    model_name=devil_model_name,
                    use_chat=devil_use_chat,
                    prompt_template=devil_debater_prompt,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    device=f'cuda:{len(debate_agents)}',
                    max_new_tokens=512,
                    truncation=True,
                    return_full_text=False
                )
            elif devil_model_type == 'openai':
                logger.info(f"Loading OpenAI model: '{devil_model_name}' as Agent-X")
                devil_llm = AnyOpenAILLM(
                    model_name=devil_model_name,
                    use_chat=devil_use_chat,
                    prompt_template=devil_debater_prompt,
                    temperature=0.7,
                    top_p=0.9,
                    max_retries=5
                )
            elif devil_model_type == 'deepseek':
                logger.info(f"Loading OpenAI model: '{devil_model_type}' as Agent-X")
                devil_llm = AnyOpenAILLM(
                    model_name=devil_model_name,
                    use_chat=devil_use_chat,
                    prompt_template=devil_debater_prompt,
                    openai_api_base='https://api.deepseek.com',
                    openai_api_key=os.environ.get('DEEPSEEK_API_KEY'),
                    temperature=0.7,
                    top_p=0.9,
                    max_retries=5
                )
            elif devil_model_type == 'google':
                logger.info(f"Loading Google model: '{devil_model_name}' as Agent-X")
                devil_llm = AnyGoogleGenerativeAI(
                    model_name=devil_model_name,
                    use_chat=devil_use_chat,
                    prompt_template=devil_debater_prompt,
                    temperature=0.7,
                    top_p=0.9,
                )
            else:
                raise ValueError(
                    f"Model type '{devil_model_type}' not supported for debate. "
                    "Please choose from 'huggingface', 'openai', or 'google'."
                )
            # loaded_llms[devil_model] = devil_llm

        devil_agent = DevilAngelAgent(name='Agent-X', base_llm=devil_llm, target_agents_names=target_agents_names)

        if angel_model in loaded_llms:
            logger.info(f"Reusing previously loaded {angel_model_type} model: '{angel_model_name}' as Agent-Y")
            angel_llm = loaded_llms[angel_model]
        else:
            if angel_model_type == 'huggingface':
                logger.info(f"Loading HuggingFace model: '{angel_model_name}' as Agent-Y")
                angel_llm = AnyHuggingFace(
                    model_name=angel_model_name,
                    use_chat=angel_use_chat,
                    prompt_template=angel_debater_prompt,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    device=f'cuda:{len(debate_agents) + 1}',
                    max_new_tokens=512,
                    truncation=True,
                    return_full_text=False
                )
            elif angel_model_type == 'openai':
                logger.info(f"Loading OpenAI model: '{angel_model_name}' as Agent-Y")
                angel_llm = AnyOpenAILLM(
                    model_name=angel_model_name,
                    use_chat=angel_use_chat,
                    prompt_template=angel_debater_prompt,
                    temperature=0.7,
                    top_p=0.9,
                    max_retries=5
                )
            elif angel_model_type == 'deepseek':
                logger.info(f"Loading OpenAI model: '{angel_model_type}' as Agent-Y")
                angel_llm = AnyOpenAILLM(
                    model_name=angel_model_name,
                    use_chat=angel_use_chat,
                    prompt_template=angel_debater_prompt,
                    openai_api_base='https://api.deepseek.com',
                    openai_api_key=os.environ.get('DEEPSEEK_API_KEY'),
                    temperature=0.7,
                    top_p=0.9,
                    max_retries=5
                )
            elif angel_model_type == 'google':
                logger.info(f"Loading Google model: '{angel_model_name}' as Agent-Y")
                angel_llm = AnyGoogleGenerativeAI(
                    model_name=angel_model_name,
                    use_chat=angel_use_chat,
                    prompt_template=angel_debater_prompt,
                    temperature=0.7,
                    top_p=0.9,
                )
            else:
                raise ValueError(
                    f"Model type '{angel_model_type}' not supported for debate. "
                    "Please choose from 'huggingface', 'openai', or 'google'."
                )
            # loaded_llms[angel_model] = angel_llm

        angel_agent = DevilAngelAgent(name='Agent-Y', base_llm=angel_llm, target_agents_names=target_agents_names)

    # Evaluator agent
    if evaluator_model is None:
        logger.info("Skipping evaluation of responses.")
        eval_agent = None
    else:
        evaluator_model_splitted = evaluator_model.split(':')
        if len(evaluator_model_splitted) == 2:
            evaluator_model_type, evaluator_model_name = evaluator_model_splitted
            evaluator_use_chat = 'true'
        elif len(evaluator_model_splitted) == 3:
            evaluator_model_type, evaluator_model_name, evaluator_use_chat = evaluator_model_splitted
        else:
            raise ValueError(
                f"Invalid evaluator model format: {evaluator_model}. "
                "Please use the format 'model_type:model_name' or 'model_type:model_name:use_chat'."
            )

        if evaluator_model_type == 'openai' and evaluator_model_name == 'moderation':
            logger.info("Using OpenAI's moderation API for evaluation.")
            eval_agent = EvalAgent(name='Agent-Eval', base_llm=None)
        elif 'meta-llama/llama-guard' in evaluator_model_name.lower():
            logger.info(f"Loading LlamaGuard model: '{evaluator_model}' as Agent-Eval")
            #TODO: fix device setting
            eval_agent = EvalAgent(name='Agent-Eval', base_llm=LlamaGuard(model_name=evaluator_model_name, device=f'cuda:3', max_new_tokens=20, pad_token_id=0))
        else:
            if evaluator_model in loaded_llms:
                logger.info(f"Reusing previously loaded {evaluator_model_type} model: '{evaluator_model_name}' as Agent-Eval")
                eval_llm = loaded_llms[evaluator_model]
            else:
                if evaluator_model_type == 'huggingface':
                    logger.info(f"Loading HuggingFace model: '{evaluator_model_name}' as Agent-Eval")
                    eval_llm = AnyHuggingFace(
                        model_name=evaluator_model_name,
                        use_chat=evaluator_use_chat,
                        prompt_template=eval_prmpt,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.8,
                        device=f'cuda:3'
                    )
                elif evaluator_model_type == 'openai':
                    logger.info(f"Loading OpenAI model: '{evaluator_model_name}' as Agent-Eval")
                    eval_llm = AnyOpenAILLM(
                        model_name=evaluator_model_name,
                        use_chat=evaluator_use_chat,
                        prompt_template=eval_prmpt,
                        temperature=0.3,
                        top_p=0.8,
                        max_retries=5
                    )
                elif evaluator_model_type == 'deepseek':
                    logger.info(f"Loading OpenAI model: '{evaluator_model_type}' as Agent-{i}")
                    eval_llm = AnyOpenAILLM(
                        model_name=evaluator_model_name,
                        use_chat=evaluator_use_chat,
                        prompt_template=eval_prmpt,
                        openai_api_base='https://api.deepseek.com',
                        openai_api_key=os.environ.get('DEEPSEEK_API_KEY'),
                        temperature=0.7,
                        top_p=0.9,
                        max_retries=5
                    )
                elif evaluator_model_type == 'google':
                    logger.info(f"Loading Google model: '{evaluator_model_name}' as Agent-Eval")
                    eval_llm = AnyGoogleGenerativeAI(
                        model_name=evaluator_model_name,
                        use_chat=evaluator_use_chat,
                        prompt_template=eval_prmpt,
                        temperature=0.3,
                        top_p=0.8,
                    )
                else:
                    raise ValueError(
                        f"Model type '{evaluator_model_type}' not supported for evaluation. "
                        "Please choose from 'huggingface', 'openai', or 'google'."
                    )
                # loaded_llms[evaluator_model] = eval_llm

            eval_agent = EvalAgent(name='Agent-Eval', base_llm=eval_llm)

    # Feedback agent
    if feedback_generator is None:
        logger.info("Skipping feedback generation.")
        feedback_agent = None
    else:
        feedback_generator_type, feedback_generator_name, feedback_generator_use_chat = feedback_generator.split(':')
        if feedback_generator in loaded_llms:
            logger.info(f"Reusing previously loaded {feedback_generator_type} model: '{feedback_generator_name}' as Agent-Feedback")
            feedback_llm = loaded_llms[feedback_generator]
        else:
            if feedback_generator_type == 'huggingface':
                logger.info(f"Loading HuggingFace model: '{feedback_generator_name}' as Agent-Feedback")
                feedback_llm = AnyHuggingFace(
                    model_name=feedback_generator_name,
                    use_chat=feedback_generator_use_chat,
                    prompt_template=feedback_prmpt,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.8,
                    device=f'cuda:{len(debate_agents) + 1}',
                )
            elif feedback_generator_type == 'openai':
                logger.info(f"Loading OpenAI model: '{feedback_generator_name}' as Agent-Feedback")
                feedback_llm = AnyOpenAILLM(
                    model_name=feedback_generator_name,
                    use_chat=feedback_generator_use_chat,
                    prompt_template=feedback_prmpt,
                    temperature=0.3,
                    top_p=0.8,
                    max_retries=5
                )
            elif feedback_generator_type == 'deepseek':
                logger.info(f"Loading OpenAI model: '{feedback_generator_type}' as Agent-{i}")
                feedback_llm = AnyOpenAILLM(
                    model_name=feedback_generator_name,
                    use_chat=feedback_generator_use_chat,
                    prompt_template=feedback_prmpt,
                    openai_api_base='https://api.deepseek.com',
                    openai_api_key=os.environ.get('DEEPSEEK_API_KEY'),
                    temperature=0.7,
                    top_p=0.9,
                    max_retries=5
                )
            elif feedback_generator_type == 'google':
                logger.info(f"Loading Google model: '{feedback_generator_name}' as Agent-Feedback")
                feedback_llm = AnyGoogleGenerativeAI(
                    model_name=feedback_generator_name,
                    use_chat=feedback_generator_use_chat,
                    prompt_template=feedback_prmpt,
                    temperature=0.3,
                    top_p=0.8,
                )
            else:
                raise ValueError(
                    f"Model type '{feedback_generator_type}' not supported for feedback generation. "
                    "Please choose from 'huggingface', 'openai', or 'google'."
                )
            # loaded_llms[feedback_generator] = feedback_llm

        feedback_agent = FeedbackAgent(name='Agent-Feedback', base_llm=feedback_llm)

    # Questioner agent(Socrates)
    if questioner_model is None:
        logger.info("Skipping question generation.")
        questioner_agent = None
    else:
        questioner_model_type, questioner_model_name, questioner_use_chat = questioner_model.split(':')
        if questioner_model in loaded_llms:
            logger.info(f"Reusing previously loaded {questioner_model_type} model: '{questioner_model_name}' as Agent-Socrates")
            questioner_llm = loaded_llms[questioner_model]
        else:
            if questioner_model_type == 'huggingface':
                logger.info(f"Loading HuggingFace model: '{questioner_model_name}' as Agent-Socrates")
                questioner_llm = AnyHuggingFace(
                    model_name=questioner_model_name,
                    use_chat=questioner_use_chat,
                    prompt_template=socratic_agent_prompt,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.8,
                    device=f'cuda:{len(debate_agents) + 2}',
                )
            elif questioner_model_type == 'openai':
                logger.info(f"Loading OpenAI model: '{questioner_model_name}' as Agent-Socrates")
                questioner_llm = AnyOpenAILLM(
                    model_name=questioner_model_name,
                    use_chat=questioner_use_chat,
                    prompt_template=socratic_agent_prompt,
                    temperature=0.3,
                    top_p=0.8,
                    max_retries=5
                )
            elif questioner_model_type == 'deepseek':
                logger.info(f"Loading OpenAI model: '{questioner_model_type}' as Agent-{i}")
                questioner_llm = AnyOpenAILLM(
                    model_name=questioner_model_name,
                    use_chat=questioner_use_chat,
                    prompt_template=socratic_agent_prompt,
                    openai_api_base='https://api.deepseek.com',
                    openai_api_key=os.environ.get('DEEPSEEK_API_KEY'),
                    temperature=0.7,
                    top_p=0.9,
                    max_retries=5
                )
            elif questioner_model_type == 'google':
                logger.info(f"Loading Google model: '{questioner_model_name}' as Agent-Socrates")
                questioner_llm = AnyGoogleGenerativeAI(
                    model_name=questioner_model_name,
                    use_chat=questioner_use_chat,
                    prompt_template=socratic_agent_prompt,
                    temperature=0.3,
                    top_p=0.8,
                )
            else:
                raise ValueError(
                    f"Model type '{questioner_model_type}' not supported for question generation. "
                    "Please choose from 'huggingface', 'openai', or 'google'."
                )
            # loaded_llms[questioner_model] = questioner_llm

        questioner_agent = DebateAgent(name='Agent-Socrates', base_llm=questioner_llm)

    if self_critique_model is None:
        logger.info("Skipping self-critique.")
        self_critic_agent = None
    else:
        self_critique_model_type, self_critique_model_name, self_critique_use_chat = self_critique_model.split(':')
        if self_critique_model in loaded_llms:
            logger.info(f"Reusing previously loaded {self_critique_model_type} model: '{self_critique_model_name}' as Agent-SelfCritic")
            self_critic_llm = loaded_llms[self_critique_model]
            base_llm = loaded_llms[self_critique_model]
            revision_llm = loaded_llms[self_critique_model]
        else:
            if self_critique_model_type == 'huggingface':
                logger.info(f"Loading HuggingFace model: '{self_critique_model_name}' as Agent-SelfCritic")

                self_critic_llm = AnyHuggingFace(
                    model_name=self_critique_model_name,
                    use_chat=self_critique_use_chat,
                    prompt_template=self_critique_prompt,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    device=f'cuda:0',
                    max_new_tokens=512,
                    truncation=True,
                    return_full_text=False
                )
                base_llm = AnyHuggingFace(
                    model_name=self_critique_model_name,
                    use_chat=self_critique_use_chat,
                    prompt_template=init_response_prompt,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    device=f'cuda:1',
                    max_new_tokens=512,
                    truncation=True,
                    return_full_text=False
                )
                revision_llm = AnyHuggingFace(
                    model_name=self_critique_model_name,
                    use_chat=self_critique_use_chat,
                    prompt_template=revise_response_prompt,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    device=f'cuda:2',
                    max_new_tokens=512,
                    truncation=True,
                    return_full_text=False
                )
            elif self_critique_model_type == 'openai':
                logger.info(f"Loading OpenAI model: '{self_critique_model_name}' as Agent-SelfCritic")
                self_critic_llm = AnyOpenAILLM(
                    model_name=self_critique_model_name,
                    use_chat=self_critique_use_chat,
                    prompt_template=self_critique_prompt,
                    temperature=0.3,
                    top_p=0.8,
                    max_retries=5
                )
                base_llm = AnyOpenAILLM(
                    model_name=self_critique_model_name,
                    use_chat=self_critique_use_chat,
                    prompt_template=init_response_prompt,
                    temperature=0.3,
                    top_p=0.8,
                    max_retries=5
                )
                revision_llm = AnyOpenAILLM(
                    model_name=self_critique_model_name,
                    use_chat=self_critique_use_chat,
                    prompt_template=revise_response_prompt,
                    temperature=0.3,
                    top_p=0.8,
                    max_retries=5
                )
            elif self_critique_model_type == 'deepseek':
                logger.info(f"Loading OpenAI model: '{self_critique_model_type}' as Agent-{i}")
                self_critic_llm = AnyOpenAILLM(
                    model_name=self_critique_model_name,
                    use_chat=self_critique_use_chat,
                    prompt_template=self_critique_prompt,
                    openai_api_base='https://api.deepseek.com',
                    openai_api_key=os.environ.get('DEEPSEEK_API_KEY'),
                    temperature=0.7,
                    top_p=0.9,
                    max_retries=5
                )
                base_llm = AnyOpenAILLM(
                    model_name=self_critique_model_name,
                    use_chat=self_critique_use_chat,
                    prompt_template=init_response_prompt,
                    openai_api_base='https://api.deepseek.com',
                    openai_api_key=os.environ.get('DEEPSEEK_API_KEY'),
                    temperature=0.7,
                    top_p=0.9,
                    max_retries=5
                )
                revision_llm = AnyOpenAILLM(
                    model_name=self_critique_model_name,
                    use_chat=self_critique_use_chat,
                    prompt_template=revise_response_prompt,
                    openai_api_base='https://api.deepseek.com',
                    openai_api_key=os.environ.get('DEEPSEEK_API_KEY'),
                    temperature=0.7,
                    top_p=0.9,
                    max_retries=5
                )
            elif self_critique_model_type == 'google':
                logger.info(f"Loading Google model: '{self_critique_model_name}' as Agent-SelfCritic")
                self_critic_llm = AnyGoogleGenerativeAI(
                    model_name=self_critique_model_name,
                    use_chat=self_critique_use_chat,
                    prompt_template=self_critique_prompt,
                    temperature=0.3,
                    top_p=0.8,
                )
                base_llm = AnyGoogleGenerativeAI(
                    model_name=self_critique_model_name,
                    use_chat=self_critique_use_chat,
                    prompt_template=init_response_prompt,
                    temperature=0.3,
                    top_p=0.8,
                )
                revision_llm = AnyGoogleGenerativeAI(
                    model_name=self_critique_model_name,
                    use_chat=self_critique_use_chat,
                    prompt_template=revise_response_prompt,
                    temperature=0.3,
                    top_p=0.8,
                )
            else:
                raise ValueError(
                    f"Model type '{self_critique_model_type}' not supported for self-critique. "
                    "Please choose from 'huggingface', 'openai', or 'google'."
                )


        self_critic_agent = SelfCriticAgent(name='Agent-SelfCritic', base_llm=base_llm, critic_llm=self_critic_llm, revision_llm=revision_llm)

    # (Optional) If you have human agents:
    # debate_human_number = 1
    # debate_humans = [HumanAgent(f'Human-{i}', question) for i in range(debate_human_number)]
    debate_humans = []

    return debate_agents, devil_agent, angel_agent, eval_agent, feedback_agent, debate_humans, questioner_agent, self_critic_agent

def run_debate(debater_models, devil_model, angel_model, evaluator_model, feedback_generator, questioner_model, self_critique_model, datasets, debate_rounds, max_total_debates, output_file, long_term_memory_index_name, checkpoint_dir=None):
    logger = setup_logger(output_file)

    if long_term_memory_index_name:  # if long term memory index name is provided, use VectorStoreMemory
        long_term_memory = VectorStoreMemory('Long Term Memory: This memory which is shared among all agents will keep track of the feedbacks of the previous debates to avoid making the same mistakes again',
                                             index_name=long_term_memory_index_name)
    else: # otherwise, use LongTermMemory (array)
        long_term_memory = LongTermMemory('Long Term Memory: This memory which is shared among all agents will keep track of the feedbacks of the previous debates to avoid making the same mistakes again')

    datasets_obj = load_datasets(datasets)
    debate_agents, devil_agent, angel_agent, eval_agent, feedback_agent, debate_humans, questioner_agent, self_critic_agent = init_agents(debater_models, devil_model, angel_model, evaluator_model, feedback_generator, questioner_model, self_critique_model, logger)

    max_total_debates = max_total_debates if max_total_debates is not None else float('inf') # if not set, there is no constraint

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    if checkpoint_dir is None:
        checkpoint_dir = f"checkpoints/debate_{timestamp}"
        logger.info(f"Checkpoint directory not provided. Saving checkpoints to {checkpoint_dir}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"loading checkpoints and debate log from {checkpoint_dir} (if exists)")
    completed_debates = load_checkpoint(checkpoint_dir)
    if self_critique_model is not None:
        logger.info(f"Loading saved self-critique debates from {checkpoint_dir}")
        debates = load_saved_debates(checkpoint_dir, conversation_type='selfcritique')
    else:
        logger.info(f"Loading saved debates from {checkpoint_dir}")
        debates = load_saved_debates(checkpoint_dir)
    try:
        for dataset_name, dataset in datasets_obj.items():
            logger.info(f"Running debate on dataset: {dataset_name} with {len(dataset)} samples")
            last_processed = completed_debates.get(dataset_name, -1)
            for idx, question in enumerate(dataset):
                if idx <= last_processed:
                    logger.info(f"Skipping already processed question {idx} in {dataset_name}")
                    continue

                if len(debates) >= max_total_debates:
                    logger.info(f"Reached maximum number of debates: {max_total_debates}. Exiting...")
                    return debates

                logger.info(f"Running debate on question: {question}")

                # initialize short term memory
                short_term_memory = ShortTermMemory('Short Term Memory: This memory which is shared among all agents will keep track of the chat history of the current debate')

                if isinstance(long_term_memory, VectorStoreMemory): # update long term memory for relevant questions
                    long_term_memory.update_vector_memory(question['text'], k=5)

                # start debate
                if self_critic_agent is not None:
                    debate = SelfCritique(question, self_critic_agent, eval_agent, log_file=output_file)
                elif questioner_agent is not None:
                    debate = SocraticDebate(question, debate_agents, debate_humans, eval_agent, feedback_agent, questioner_agent, short_term_memory, long_term_memory, log_file=output_file)
                elif devil_agent is not None and angel_agent is not None:
                    debate = DevilAngelDebate(question, debate_agents, debate_humans, eval_agent, feedback_agent, devil_agent, angel_agent, short_term_memory, long_term_memory, log_file=output_file)
                else:
                    debate = Debate(question, debate_agents, debate_humans, eval_agent, feedback_agent, short_term_memory, long_term_memory, log_file=output_file)
                debate.start(rounds=debate_rounds)
                debates.append(debate)

                # save progress
                save_debate_log(debate, checkpoint_dir, dataset_name, idx)
                completed_debates[dataset_name] = idx
                save_checkpoint(checkpoint_dir, completed_debates)
    except Exception as e:
        logger.exception(f"An error occurred during debate: {e}")
        return None
    finally:

        logger.info("Saving long term memory...")
        results_dir = f'results/{timestamp}'
        os.makedirs(results_dir, exist_ok=True)
        long_term_memory_path = f'{results_dir}/long_term_memory.txt'
        long_term_memory.save(long_term_memory_path)
        logger.info(f"Long term memory saved at {long_term_memory_path}")

        logger.info("Calculating evaluation metrics...")
        try:
            metrics = calculate_debate_metrics(debates)
            metrics_path = f'{results_dir}/metrics.json'
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved locally at {metrics_path}")
            logger.info(f"Metrics: {json.dumps(metrics, indent=4)}")

            logger.info("Logging results to Weights & Biases...")
            experiment_parameters = {
                'debater_models': debater_models,
                'devil_model': devil_model,
                'angel_model': angel_model,
                'evaluator_model': evaluator_model,
                'feedback_generator': feedback_generator,
                'questioner_model': questioner_model,
                'datasets': datasets,
                'debate_rounds': debate_rounds,
                'max_total_debates': max_total_debates,
                'output_file': output_file,
                'run_id': timestamp
            }
            try:
                experiment_parameters['commit_hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
            except Exception as e:
                logger.warning(f"Could not get commit hash: {e}")
            log_results_to_wandb(metrics, experiment_parameters, debates)
            logger.info("Results logged to Weights & Biases.")
        except Exception as e:
            logger.exception(f"An error occurred while calculating evaluation metrics or logging results to Weights & Biases: {e}")
    return debates