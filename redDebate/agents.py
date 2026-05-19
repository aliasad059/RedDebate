"""Agent abstractions for the RedDebate framework.

Each agent is a thin wrapper around one (or several) LLM clients from
:mod:`redDebate.llm` plus the bookkeeping needed for its role inside a
debate (current round, target agents, etc.). The orchestrator in
:mod:`redDebate.run` instantiates the appropriate combination of agents
based on which CLI flags the user passed.

Agent roles:

* :class:`HumanAgent` – interactive ``stdin`` participant.
* :class:`DebateAgent` – standard debating LLM.
* :class:`PEFTDebateAgent` – debate agent that LoRA-fine-tunes itself on
  accumulated feedback (long-term memory via weights).
* :class:`DevilAngelAgent` – the antagonist/protagonist pair used by
  :class:`~redDebate.debate.DevilAngelDebate`.
* :class:`EvalAgent` – safety judge (LlamaGuard, generic LLM judge, or the
  OpenAI moderation API).
* :class:`FeedbackAgent` – generates textual feedback after a debate.
* :class:`SelfCriticAgent` – three-stage (answer → critique → revise) agent
  used by :class:`~redDebate.self_critique.SelfCritique`.
"""

import gc
import os
import torch
from openai import OpenAI
from .llm import AnyOpenAILLM, AnyHuggingFace, AnyVLLM, LlamaGuard
from .memory import ShortTermMemory, LongTermMemory
from typing import Union, Tuple


class HumanAgent:
    """Interactive participant that reads its response from ``stdin``.

    Used by the ``--human_in_the_loop`` CLI flag. The agent is queried
    before the LLM debaters each round so the human's response is part of
    the short-term memory the LLMs see.
    """

    def __init__(self,
                 name: str,
                 question_text: str,
                ) -> None:
        self.name = name
        self.question = question_text
        self.debate_round = 0

    def run(self) -> str:
        """Prompt the user via ``input()`` and return whatever they typed."""
        self.debate_round += 1
        result = input(f'{self.name} enter your response here: ')
        return result


class DebateAgent:
    """Standard debating LLM.

    Calls ``base_llm`` with the current question, the running short-term
    memory, the shared textual long-term memory and a debate-round counter.
    """

    def __init__(self,
                 name: str,
                 base_llm: AnyOpenAILLM | AnyHuggingFace | AnyVLLM,
                ) -> None:
        self.name = name
        self.base_llm = base_llm
        self.debate_round = 0

    def run(self, question_text: str, short_term_memory: ShortTermMemory, long_term_memory: LongTermMemory) -> Union[str, Tuple[str, str]]:
        """Produce a single response for the current round.

        Returns the raw model output. For thinking models that emit a
        ``<think>...</think>`` block, the LLM wrapper returns a
        ``(thinking, response)`` tuple instead.
        """
        self.debate_round += 1
        response = self.base_llm({
            "agent_name": self.name,
            "question": question_text,
            "debate_round": self.debate_round,
            "short_term_memory": str(short_term_memory),
            "long_term_memory": str(long_term_memory)
        })
        return response


class PEFTDebateAgent(DebateAgent):
    """DebateAgent that can fine-tune itself using LoRA on accumulated feedback.

    Requires an AnyHuggingFace base_llm. After training, the LoRA weights are
    merged back into the base model and the inference pipeline is rebuilt, so
    subsequent calls use the updated model transparently.
    """

    # GPU assignment per model — adjust to match your cluster topology. Note: By default, the previously selected device of the inference pipeline will be used.
    _DEVICE_MAP = {
        'mistralai/Mistral-7B-Instruct-v0.2': 0,
        'google/gemma-3-12b-it': 0,
        'microsoft/Phi-3.5-mini-instruct': 2,
        'deepseek-ai/DeepSeek-R1-Distill-Llama-8B': 2,
    }

    def trainPEFT(self, feedback_dict: list, save_directory: str):
        """Fine-tune the underlying HuggingFace model with LoRA on accumulated feedback.

        After training completes the model is saved as a merged (non-PEFT) checkpoint
        and the agent's inference pipeline is rebuilt from that checkpoint, so the
        agent continues to work as normal without any caller changes.

        Args:
            feedback_dict: list of {"text": <feedback_string>} dicts to train on.
            save_directory: local path where the merged model will be saved.
        """
        if not isinstance(self.base_llm, AnyHuggingFace):
            raise TypeError("PEFTDebateAgent.trainPEFT requires an AnyHuggingFace base_llm")

        # Lazy imports so peft/datasets are only required when actually training
        from peft import get_peft_model, LoraConfig, TaskType, PeftModel
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

        class _NoParallelTrainer(Trainer):
            """Skips DataParallel wrapping to avoid crashes with multi-GPU PEFT models."""
            def _wrap_model(self, model, training=True):
                return model

        llm = self.base_llm
        llm.logger.info(f"[{self.name}] Starting PEFT training for {llm.model_name}")

        # Snapshot the attributes we need to rebuild AnyHuggingFace after training
        model_path = llm.model_path
        model_name = llm.model_name
        use_chat_str = llm._use_chat_str
        prompt_template = llm.prompt_template
        pipe_kwargs = llm.kwargs
        task = llm.task

        # Free the inference model to reclaim VRAM before loading for training
        llm.model.cpu()
        del llm.model, llm.tokenizer, llm.llm, llm.pipe
        gc.collect()
        torch.cuda.empty_cache()

        # Prefer the device set when the inference pipeline was built (e.g. 'cuda:0')
        # so training lands on the same GPU.  Fall back to _DEVICE_MAP, then cuda:1.
        device_str = pipe_kwargs.get('device', None)
        if device_str is None:
            fallback_id = self._DEVICE_MAP.get(model_name, 1)
            device_str = f'cuda:{fallback_id}'
        device_id = int(device_str.split(':')[1]) if ':' in device_str else 1
        torch.cuda.set_device(device_id)

        llm.logger.info(f"[{self.name}] Loading model for PEFT training on {device_str} (model: {model_name})")

        hf_cache = os.environ.get('HF_HUB_CACHE')
        hf_token = os.environ.get('HF_HOME')

        train_tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=hf_cache, token=hf_token)
        train_tokenizer.pad_token = train_tokenizer.eos_token

        dtype = torch.bfloat16 if 'gemma' in model_name.lower() else torch.float16
        # Load directly onto the target GPU (device_map avoids the CPU-staging step
        # that would temporarily double peak memory usage).
        train_model = AutoModelForCausalLM.from_pretrained(
            model_path, cache_dir=hf_cache, token=hf_token,
            torch_dtype=dtype, device_map={'': f'cuda:{device_id}'}
        )

        # LoRA target modules vary by model architecture
        if 'phi-3.5' in model_name.lower() or 'phi3.5' in model_name.lower():
            target_modules = ["o_proj", "qkv_proj"]
        elif 'gemma' in model_name.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
        )

        train_model.config.use_cache = False
        train_model.enable_input_require_grads()
        train_model = get_peft_model(train_model, lora_config)
        train_model.gradient_checkpointing_enable()

        def _tokenize(examples):
            tokenized = train_tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=512
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = Dataset.from_list(feedback_dict).map(_tokenize, batched=True)

        training_args = TrainingArguments(
            output_dir=save_directory,
            num_train_epochs=5,
            per_device_train_batch_size=1,
            learning_rate=5e-5,
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        )

        trainer = _NoParallelTrainer(
            model=train_model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        trainer.train()

        # Merge LoRA weights into the base model and save as a standard checkpoint
        merged = train_model.merge_and_unload()
        merged.save_pretrained(save_directory)
        train_tokenizer.save_pretrained(save_directory)
        llm.logger.info(f"[{self.name}] Training complete, merged model saved to {save_directory}")

        del train_model, merged, train_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        # Rebuild the inference pipeline from the merged checkpoint. Because the
        # merged model is a plain HuggingFace model, AnyHuggingFace can load it
        # without knowing anything about PEFT.
        self.base_llm = AnyHuggingFace(
            model_name=save_directory,
            use_chat=use_chat_str,
            prompt_template=prompt_template,
            **pipe_kwargs,
        )
        llm.logger.info(f"[{self.name}] Inference pipeline rebuilt from {save_directory}")


class DevilAngelAgent:
    """Adversarial or supportive companion targeting one or more debaters.

    Same call signature as :class:`DebateAgent` but additionally passes the
    list of ``target_agents_names`` so the prompt template can render
    "challenge agents X, Y, ..." (devil) or "support agents X, Y, ..."
    (angel) instructions.
    """

    def __init__(self,
                 name: str,
                 base_llm: AnyOpenAILLM | AnyHuggingFace | AnyVLLM,
                 target_agents_names: list[str],
                ) -> None:
        self.name = name
        self.base_llm = base_llm
        self.debate_round = 0
        self.target_agents_names = target_agents_names

    def run(self, question_text: str, short_term_memory: ShortTermMemory, long_term_memory: LongTermMemory) -> str:
        """Generate a devil/angel turn that references the named debaters."""
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
    """Safety judge: decides whether an agent response should be blocked.

    The behavior depends on the type of ``base_llm``:

    * :class:`~redDebate.llm.LlamaGuard` – LlamaGuard is called directly.
    * Generic LLM (OpenAI/HF/vLLM) – an evaluation prompt is rendered and
      the model's textual verdict is returned.
    * ``None`` – the OpenAI ``omni-moderation-latest`` API is queried and
      the moderation flag is mapped to ``Yes``/``No``.
    """

    def __init__(self,
                 name: str,
                 base_llm: AnyOpenAILLM | AnyHuggingFace | AnyVLLM | LlamaGuard | None,
                 ) -> None:
        self.name = name
        self.base_llm = base_llm

    def run(self, question_text: str, agent_response: str, long_term_memory: LongTermMemory | None) -> str:
        """Return a textual verdict; downstream code looks for ``no``/``safe``."""
        if isinstance(self.base_llm, LlamaGuard):
            return self.base_llm(agent_response)
        elif isinstance(self.base_llm, (AnyOpenAILLM, AnyHuggingFace, AnyVLLM)):
            response = self.base_llm({
                "question": question_text,
                "agent_response": agent_response,
                "long_term_memory": str(long_term_memory)
            })
            return response
        elif self.base_llm is None:
            try:
                client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
                response = client.moderations.create(model="omni-moderation-latest", input=agent_response)
                return 'Yes' if response.results[0].flagged else 'No'
            except Exception as e:
                print(f"Error checking safety with OpenAI moderation API: {e}")
                return 'Yes'
        else:
            raise ValueError("Invalid base_llm type")


class FeedbackAgent:
    """Generates a bulleted critique of a finished debate transcript.

    The output is appended verbatim to long-term memory and shown to all
    agents in subsequent debates as "rules to follow".
    """

    def __init__(self,
                 name: str,
                 base_llm: AnyOpenAILLM | AnyHuggingFace | AnyVLLM,
                ) -> None:
        self.name = name
        self.base_llm = base_llm

    def run(self, question_text: str, chat_history: str) -> str:
        """Return feedback text for the given debate transcript."""
        response = self.base_llm({
            "question": question_text,
            "chat_history": chat_history
        })
        return response


class SelfCriticAgent:
    """Three-stage Constitutional-AI style agent.

    On each step the agent (1) produces an initial answer (or reuses the
    revised answer from the previous round), (2) critiques it against a
    randomly-sampled constitutional rule, (3) rewrites it according to a
    matching revision rule. All three sub-calls go through separate LLM
    wrappers so they can use distinct prompt templates and devices.
    """

    def __init__(self,
                 name: str,
                 base_llm: AnyOpenAILLM | AnyHuggingFace | AnyVLLM,
                 critic_llm: AnyOpenAILLM | AnyHuggingFace | AnyVLLM,
                 revision_llm: AnyOpenAILLM | AnyHuggingFace | AnyVLLM,
                ) -> None:
        self.name = name
        self.base_llm = base_llm
        self.critic_llm = critic_llm
        self.revision_llm = revision_llm

    def run(self, question_text: str, constitutional_rule: str, revision_rule: str, long_term_memory: str, init_response: str) -> dict:
        """Run one answer → critique → revise cycle.

        Args:
            question_text: User question.
            constitutional_rule: Critique instructions sampled from the
                ConstitutionalHarmlessnessPaper rule set.
            revision_rule: Matching revision instructions.
            long_term_memory: Rendered long-term memory string.
            init_response: If non-empty, used as the starting answer
                (typically the revised answer from the previous round);
                otherwise a fresh initial answer is generated.

        Returns:
            ``{"initial_response", "critique", "revised_response"}`` with each
            value being a ``{"text": ...}`` dict.
        """
        if not init_response:
            init_response = self.base_llm({
                "question": question_text,
                "long_term_memory": long_term_memory,
            })
            if isinstance(init_response, tuple):  # in case the agent returns (thinking, response)
                thinking, init_response = init_response
        else:
            print(f'Using the response from the last round as the initial response: {init_response}')

        critique = self.critic_llm({
            "question": question_text,
            "response": init_response,
            "constitutional_rule": constitutional_rule
        })
        if isinstance(critique, tuple):
            thinking, critique = critique

        revised_response = self.revision_llm({
            "question": question_text,
            "response": init_response,
            "critique": critique,
            "revision_rule": revision_rule
        })
        if isinstance(revised_response, tuple):
            thinking, revised_response = revised_response

        return {
            "initial_response": {'text': init_response},
            "critique": {'text': critique},
            "revised_response": {'text': revised_response}
        }
