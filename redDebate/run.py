"""Top-level orchestration for RedDebate.

:func:`run_debate` is the function called by ``main.py``. It:

1. Optionally picks a textual long-term memory backend (array vs. Pinecone vector store).
2. Loads the requested datasets.
3. Instantiates the right combination of agents
   (:func:`init_agents`) based on which model flags were supplied.
4. Iterates the dataset(s), selecting a debate flavour
   (``SelfCritique``/``SocraticDebate``/``DevilAngelDebate``/``PeerDebate``)
   depending on which optional agents exist.
5. Optionally activates continues long-term memory which accumulates feedback and triggers PEFT/LoRA fine-tuning
   of the HuggingFace debaters every ``train_steps`` samples.
6. Checkpoints progress to disk so a run can be resumed and finally
   logs metrics to Weights & Biases.

Helper functions handle JSON checkpoint I/O and parsing the
``<type>:<name>[:use_chat]`` model strings used on the CLI.
"""

import os
import shutil

from dotenv import load_dotenv
from datetime import datetime
import json
import subprocess
from pathlib import Path

from .debate import Debate, DevilAngelDebate, SocraticDebate
from .self_critique import SelfCritique
from .agents import DebateAgent, PEFTDebateAgent, DevilAngelAgent, EvalAgent, FeedbackAgent, HumanAgent, SelfCriticAgent
from .memory import ShortTermMemory, LongTermMemory, VectorStoreMemory
from .llm import AnyHuggingFace, AnyOpenAILLM, AnyVLLM, LlamaGuard, AnyGoogleGenerativeAI
from .debate_prompts import debate_agent_prompt, debate_agent_prompt_base, devil_debater_prompt, angel_debater_prompt , feedback_prmpt, socratic_agent_prompt, eval_prmpt, init_response_prompt, self_critique_prompt, revise_response_prompt
from .dataloader import load_datasets
from .util import setup_logger
from .metrics import calculate_debate_metrics, log_results_to_wandb


load_dotenv()


def save_debate_log(debate, checkpoint_folder: str, dataset_name: str, question_idx: int):
    """Serialize a finished debate to ``<folder>/<dataset>_q<idx>.json``."""
    os.makedirs(checkpoint_folder, exist_ok=True)
    debate_log_path = os.path.join(checkpoint_folder, f"{dataset_name}_q{question_idx}.json")

    debate.save_to_json(debate_log_path)

def save_checkpoint(checkpoint_folder: str, completed_debates: dict):
    """Persist the per-dataset 'last processed index' map to ``checkpoint.json``."""
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder, "checkpoint.json")

    with open(checkpoint_path, "w") as f:
        json.dump(completed_debates, f, indent=4)

def load_checkpoint(checkpoint_folder: str):
    """Return the persisted index map, or ``{}`` when no checkpoint exists."""
    checkpoint_path = os.path.join(checkpoint_folder, "checkpoint.json")

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return {}

def save_peft_training_data(checkpoint_folder: str, feedback_dict: list):
    """Persist the accumulated feedback list used as PEFT training data."""
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder, "feedback_dict.json")
    with open(checkpoint_path, "w") as f:
        json.dump(feedback_dict, f, indent=4)

def load_peft_training_data(checkpoint_folder: str) -> list | None:
    """Inverse of :func:`save_peft_training_data`; ``None`` if no file exists."""
    checkpoint_path = os.path.join(checkpoint_folder, "feedback_dict.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return None

def load_saved_debates(checkpoint_folder: str, conversation_type: str = 'debate'):
    """Reconstruct previously saved debates from a checkpoint directory.

    ``conversation_type`` selects which deserialiser to use
    (:meth:`Debate.load_from_json` vs. :meth:`SelfCritique.load_from_json`).
    Returns an empty list if the folder is missing or empty.
    """
    debates = []
    if not os.path.exists(checkpoint_folder):
        return debates  # No checkpoint to load

    skip_files = {"checkpoint.json", "feedback_dict.json"}

    if conversation_type == 'debate':
        for file in Path(checkpoint_folder).glob("*.json"):
            try:
                if file.name not in skip_files:
                    debates.append(Debate.load_from_json(str(file)))
            except Exception as e:
                print(f"Failed to load checkpoint {file}: {e}")
    elif conversation_type == 'selfcritique':
        for file in Path(checkpoint_folder).glob("*.json"):
            try:
                if file.name not in skip_files:
                    debates.append(SelfCritique.load_from_json(str(file)))
            except Exception as e:
                print(f"Failed to load checkpoint {file}: {e}")
    else:
        print(f"Unknown conversation type: {conversation_type}. Skipping loading.")

    return debates


def _parse_model_str(model_str):
    """Return (model_type, model_name, use_chat) from 'type:name[:use_chat]'."""
    first = model_str.index(':')
    last = model_str.rindex(':')
    model_type = model_str[:first]
    if first == last:
        return model_type, model_str[first + 1:], 'true'
    return model_type, model_str[first + 1:last], model_str[last + 1:]


def _create_llm(model_str, prompt_template, logger, agent_label='',
                temperature=0.7, top_p=0.9, device=None,
                max_new_tokens=None, truncation=True, return_full_text=False,
                do_sample=True, max_tokens=None):
    """Instantiate the right LLM from a 'type:name[:use_chat]' model string."""
    model_type, model_name, use_chat = _parse_model_str(model_str)
    if agent_label:
        logger.info(f"Loading {model_type} model: '{model_name}' as {agent_label}")

    if model_type == 'huggingface':
        kwargs = dict(model_name=model_name, use_chat=use_chat, prompt_template=prompt_template,
                      do_sample=do_sample, temperature=temperature, top_p=top_p, device=device)
        if max_new_tokens is not None:
            kwargs.update(max_new_tokens=max_new_tokens, truncation=truncation, return_full_text=return_full_text)
        return AnyHuggingFace(**kwargs)

    _openai_compat = {
        'openrouter': ('https://openrouter.ai/api/v1', 'OPENROUTER_API_KEY'),
        'fai':        ('https://api.featherless.ai/v1', 'FEATHERLESS_API_KEY'),
        'deepseek':   ('https://api.deepseek.com',      'DEEPSEEK_API_KEY'),
    }
    if model_type in ('openai', *_openai_compat):
        kwargs = dict(model_name=model_name, use_chat=use_chat, prompt_template=prompt_template, max_retries=5)
        if not (model_type == 'openai' and 'gpt-5' in model_name):
            kwargs.update(temperature=temperature, top_p=top_p)
        if model_type in _openai_compat:
            api_base, key_env = _openai_compat[model_type]
            kwargs.update(openai_api_base=api_base, openai_api_key=os.environ.get(key_env))
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
        return AnyOpenAILLM(**kwargs)

    if model_type == 'google':
        return AnyGoogleGenerativeAI(
            model_name=model_name, use_chat=use_chat, prompt_template=prompt_template,
            temperature=temperature, top_p=top_p)

    raise ValueError(f"Unsupported model type '{model_type}'. "
                     "Choose from: huggingface, openai, openrouter, fai, deepseek, google.")


def init_agents(debater_models, devil_model, angel_model, evaluator_model, feedback_generator, questioner_model, self_critique_model, logger, llamaguard_cuda_device, peft_memory=False):
    """Build the full agent ensemble from the user-supplied model strings.

    Each ``*_model`` argument is either ``None`` (skip that role) or a
    ``<type>:<name>[:use_chat]`` string. Debaters and devil/angel agents
    are placed on consecutive CUDA devices starting at ``cuda:0``;
    ``llamaguard_cuda_device`` selects the GPU used for the safety judge.
    When ``peft_memory`` is True, HuggingFace debaters are wrapped in
    :class:`~redDebate.agents.PEFTDebateAgent` so they can be fine-tuned
    on accumulated feedback.

    Returns:
        ``(debate_agents, devil_agent, angel_agent, eval_agent,
        feedback_agent, questioner_agent, self_critic_agent)`` — any of
        the optional members is ``None`` when no model was supplied.
    """

    debate_agents = []
    for i, model in enumerate(debater_models):
        model_type, model_name, _ = _parse_model_str(model)
        if model_type == 'huggingface' and 'base' in model_name:
            logger.info("Using the base prompt template.")
            llm = _create_llm(model, debate_agent_prompt_base, logger, f'Agent-{i}',
                              temperature=0.7, top_p=0.9, device=f'cuda:{i}')
        else:
            llm = _create_llm(model, debate_agent_prompt, logger, f'Agent-{i}',
                              temperature=0.7, top_p=0.9, device=f'cuda:{i}',
                              max_new_tokens=512, truncation=True, return_full_text=False,
                              max_tokens=2048 if model_type == 'openrouter' else None)
        agent_class = PEFTDebateAgent if (peft_memory and model_type == 'huggingface') else DebateAgent
        debate_agents.append(agent_class(name=f'Agent-{i}', base_llm=llm))

    devil_agent = angel_agent = None
    if devil_model is not None and angel_model is not None:
        n = len(debate_agents)
        target_names = [a.name for a in debate_agents]
        devil_llm = _create_llm(devil_model, devil_debater_prompt, logger, 'Agent-X',
                                temperature=0.7, top_p=0.9, device=f'cuda:{n}',
                                max_new_tokens=512, truncation=True, return_full_text=False)
        devil_agent = DevilAngelAgent(name='Agent-X', base_llm=devil_llm, target_agents_names=target_names)
        angel_llm = _create_llm(angel_model, angel_debater_prompt, logger, 'Agent-Y',
                                temperature=0.7, top_p=0.9, device=f'cuda:{n + 1}',
                                max_new_tokens=512, truncation=True, return_full_text=False)
        angel_agent = DevilAngelAgent(name='Agent-Y', base_llm=angel_llm, target_agents_names=target_names)

    eval_agent = None
    if evaluator_model is None:
        logger.info("Skipping evaluation of responses.")
    else:
        model_type, model_name, _ = _parse_model_str(evaluator_model)
        if model_type == 'openai' and model_name == 'moderation':
            logger.info("Using OpenAI's moderation API for evaluation.")
            eval_agent = EvalAgent(name='Agent-Eval', base_llm=None)
        elif 'meta-llama/llama-guard' in model_name.lower():
            logger.info(f"Loading LlamaGuard model: '{evaluator_model}' as Agent-Eval")
            eval_agent = EvalAgent(name='Agent-Eval',
                                   base_llm=LlamaGuard(model_name=model_name,
                                                       device=f'cuda:{llamaguard_cuda_device}',
                                                       max_new_tokens=20, pad_token_id=0))
        else:
            eval_llm = _create_llm(evaluator_model, eval_prmpt, logger, 'Agent-Eval',
                                   temperature=0.3, top_p=0.8, device=f'cuda:{llamaguard_cuda_device}')
            eval_agent = EvalAgent(name='Agent-Eval', base_llm=eval_llm)

    feedback_agent = None
    if feedback_generator is None:
        logger.info("Skipping feedback generation.")
    else:
        feedback_llm = _create_llm(feedback_generator, feedback_prmpt, logger, 'Agent-Feedback',
                                   temperature=0.3, top_p=0.8, device=f'cuda:{len(debate_agents) + 1}')
        feedback_agent = FeedbackAgent(name='Agent-Feedback', base_llm=feedback_llm)

    questioner_agent = None
    if questioner_model is None:
        logger.info("Skipping Socratic question generation.")
    else:
        questioner_llm = _create_llm(questioner_model, socratic_agent_prompt, logger, 'Agent-Socrates',
                                     temperature=0.3, top_p=0.8, device=f'cuda:{len(debate_agents) + 2}')
        questioner_agent = DebateAgent(name='Agent-Socrates', base_llm=questioner_llm)

    self_critic_agent = None
    if self_critique_model is None:
        logger.info("Skipping self-critique.")
    else:
        sc_type, sc_name, _ = _parse_model_str(self_critique_model)
        logger.info(f"Loading {sc_type} model: '{sc_name}' as Agent-SelfCritic")
        hf_params = dict(max_new_tokens=512, truncation=True, return_full_text=False) if sc_type == 'huggingface' else {}
        sc_temp, sc_top_p = (0.7, 0.9) if sc_type == 'huggingface' else (0.3, 0.8)
        self_critic_llm = _create_llm(self_critique_model, self_critique_prompt, logger, '',
                                      temperature=sc_temp, top_p=sc_top_p, device='cuda:0', **hf_params)
        base_llm = _create_llm(self_critique_model, init_response_prompt, logger, '',
                                temperature=sc_temp, top_p=sc_top_p, device='cuda:1', **hf_params)
        revision_llm = _create_llm(self_critique_model, revise_response_prompt, logger, '',
                                    temperature=sc_temp, top_p=sc_top_p, device='cuda:2', **hf_params)
        self_critic_agent = SelfCriticAgent(name='Agent-SelfCritic', base_llm=base_llm,
                                            critic_llm=self_critic_llm, revision_llm=revision_llm)

    return debate_agents, devil_agent, angel_agent, eval_agent, feedback_agent, questioner_agent, self_critic_agent

def run_debate(debater_models, devil_model, angel_model, evaluator_model, feedback_generator, questioner_model, self_critique_model, datasets, debate_rounds, max_total_debates, output_file, textual_memory_index, llamaguard_cuda_device, checkpoint_dir=None, peft_memory=False, peft_directory=None, train_steps=10, human_in_the_loop=0):
    """Run a full RedDebate experiment end-to-end.

    The flavour of debate is selected implicitly by which agents the user
    requested via the CLI:

    * ``--self_critique_model``               → :class:`SelfCritique`
    * ``--questioner_model``                  → :class:`SocraticDebate`
    * ``--devil_model`` and ``--angel_model`` → :class:`DevilAngelDebate`
    * otherwise (just ``--models``)           → :class:`Debate`

    Memory:

    * If ``textual_memory_index`` is provided, long-term memory is backed
      by a Pinecone :class:`VectorStoreMemory` and refreshed per question.
    * Otherwise an in-memory :class:`LongTermMemory` array is used (and
      can be pre-seeded from each question's ``memory`` field).
    * To enable this, the feedback generator agent must be provided to generate feedback; otherwise, the agents will run without textual long-term memory.

    Resumption: every finished debate is dumped under ``checkpoint_dir``;
    re-running with the same ``checkpoint_dir`` skips previously
    processed questions and reloads any earlier debates so metrics still
    cover the whole run. Final metrics and the long-term memory are
    written under ``results/<timestamp>/`` and (when wandb is available)
    logged to Weights & Biases.

    See ``main.py`` for the corresponding CLI arguments.
    """
    logger = setup_logger(output_file)

    if textual_memory_index:  # if textual memory index name is provided, use VectorStoreMemory
        long_term_memory = VectorStoreMemory('Long Term Memory: This memory which is shared among all agents will keep track of the feedbacks of the previous debates to avoid making the same mistakes again',
                                             index_name=textual_memory_index)
    else: # otherwise, use LongTermMemory (array)
        long_term_memory = LongTermMemory('Long Term Memory: This memory which is shared among all agents will keep track of the feedbacks of the previous debates to avoid making the same mistakes again')

    datasets_obj = load_datasets(datasets)
    debate_agents, devil_agent, angel_agent, eval_agent, feedback_agent, questioner_agent, self_critic_agent = init_agents(
        debater_models, devil_model, angel_model, evaluator_model, feedback_generator,
        questioner_model, self_critique_model, logger, llamaguard_cuda_device, peft_memory=peft_memory
    )

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

    # PEFT state: accumulated feedback and the base path for LoRA checkpoints
    peft_directory = peft_directory or ""
    hf_cache = os.environ.get('HF_HUB_CACHE', '')
    feedback_dict = load_peft_training_data(checkpoint_dir) or []

    def _maybe_train_peft():
        """Trigger LoRA training on all PEFTDebateAgents if enough feedback has accumulated."""
        if not peft_memory or len(feedback_dict) < train_steps:
            return
        if len(feedback_dict) % train_steps != 0:
            return
        logger.info(f"Triggering PEFT training with {len(feedback_dict)} accumulated feedback samples")
        for agent in debate_agents:
            if isinstance(agent, PEFTDebateAgent):
                save_dir = hf_cache + peft_directory + agent.base_llm.model_name + agent.name + "_LoRA"
                agent.trainPEFT(feedback_dict, save_dir)
        save_peft_training_data(checkpoint_dir, feedback_dict)

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
                else: # set predefined memory. For test purpose only, to set predefined memory of each question before the debate starts
                    try:
                        long_term_memory.set_memories(question['memory'])
                        logger.info(f"Using predefined memory")
                    except Exception as e:
                        # logger.info(f"Could not set predefined memory")
                        pass

                # start debate
                debate_humans = [HumanAgent(f'Human-{i}', question['text']) for i in range(human_in_the_loop)]
                if self_critic_agent is not None:
                    debate = SelfCritique(question, self_critic_agent, eval_agent, feedback_agent, long_term_memory, log_file=output_file)
                elif questioner_agent is not None:
                    debate = SocraticDebate(question, debate_agents, debate_humans, eval_agent, feedback_agent, questioner_agent, short_term_memory, long_term_memory, log_file=output_file)
                elif devil_agent is not None and angel_agent is not None:
                    debate = DevilAngelDebate(question, debate_agents, debate_humans, eval_agent, feedback_agent, devil_agent, angel_agent, short_term_memory, long_term_memory, log_file=output_file)
                else:
                    debate = Debate(question, debate_agents, debate_humans, eval_agent, feedback_agent, short_term_memory, long_term_memory, log_file=output_file)

                debate.start(rounds=debate_rounds)
                debates.append(debate)

                # Accumulate feedback for PEFT training
                if peft_memory and debate.get_feedback():
                    feedback_dict.append({"text": debate.get_feedback()})
                    _maybe_train_peft()

                # save progress
                save_debate_log(debate, checkpoint_dir, dataset_name, idx)
                completed_debates[dataset_name] = idx
                save_checkpoint(checkpoint_dir, completed_debates)

    except Exception as e:
        logger.exception(f"An error occurred during debate: {e}")
        return None
    finally:
        logger.info("Saving long term memory...")
        results_dir = f'results/{peft_directory + timestamp}' if peft_directory else f'results/{timestamp}'
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
                'peft_memory': peft_memory,
                'peft_directory': peft_directory,
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