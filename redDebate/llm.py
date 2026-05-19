"""Unified LLM wrappers that hide provider-specific boilerplate.

Each wrapper accepts a LangChain :class:`PromptTemplate` and exposes a
``__call__(prompt_inputs)`` interface so debate agents can stay backend-
agnostic. The orchestrator in :mod:`redDebate.run` picks the right wrapper
based on the ``<type>:`` prefix in the model string passed on the CLI.

Supported backends:

* :class:`AnyOpenAILLM` – OpenAI plus OpenAI-compatible endpoints
  (OpenRouter, Featherless, DeepSeek).
* :class:`AnyGoogleGenerativeAI` – Google Generative AI / Gemini.
* :class:`AnyHuggingFace` – local HuggingFace ``transformers`` pipelines.
* :class:`AnyVLLM` – vLLM-backed local inference (experimental).
* :class:`LlamaGuard` – Meta's LlamaGuard moderation model used as a judge.
"""

import time
import re
import os
from typing import Union, Tuple

import torch
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_openai import ChatOpenAI, OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import VLLM

from .util import setup_logger


# loads a model from OpenAI
class AnyOpenAILLM:
    """OpenAI / OpenAI-compatible chat or completion wrapper.

    ``use_chat`` is a string (``"true"``/``"false"``) so it can come straight
    from the CLI ``<type>:<name>:<use_chat>`` triple. When True the wrapper
    builds a ``ChatOpenAI`` client, otherwise a legacy ``OpenAI`` completion
    client. All extra kwargs (temperature, top_p, max_tokens, openai_api_base,
    openai_api_key, …) are forwarded to the underlying LangChain client, which
    is why this class also works with OpenAI-compatible endpoints like
    OpenRouter or Featherless.
    """

    def __init__(self, model_name: str, use_chat: str, prompt_template: PromptTemplate, *args, **kwargs):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.use_chat = use_chat.lower() == 'true'

        if self.use_chat:
            self.llm = ChatOpenAI(model=self.model_name, cache=False, **kwargs)
        else:
            self.llm = OpenAI(model=self.model_name, **kwargs)

        self.chain = self.prompt_template | self.llm

    def __call__(self, prompt_inputs: Union[str, dict]):
        """Invoke the prompt | model chain with up to 5 retries on failure."""
        for _ in range(5):  # Retry up to 5 times
            try:
                if self.use_chat:
                    return self.chain.invoke(prompt_inputs).content
                else:
                    return self.chain.invoke(prompt_inputs)
            except Exception as e:
                print(f"Error invoking model: {e}")
                time.sleep(1)
                continue
        raise RuntimeError("Failed to invoke model after 3 attempts.")


# loads a model from GoogleGenerativeAI
class AnyGoogleGenerativeAI:
    """Google Generative AI / Gemini wrapper. Mirrors :class:`AnyOpenAILLM`."""

    def __init__(self, model_name: str, use_chat: str, prompt_template: PromptTemplate, *args, **kwargs):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.use_chat = use_chat.lower() == 'true'

        if self.use_chat:
            self.llm = ChatGoogleGenerativeAI(model=self.model_name, cache=False, **kwargs)
        else:
            self.llm = GoogleGenerativeAI(model=self.model_name, **kwargs)

        self.chain = self.prompt_template | self.llm

    def __call__(self, prompt_inputs: Union[str, dict]):
        if self.use_chat:
            return self.chain.invoke(prompt_inputs).content
        else:
            return self.chain.invoke(prompt_inputs)


# loads a model from HuggingFace
class AnyHuggingFace:
    """Local HuggingFace ``transformers`` pipeline behind a LangChain interface.

    Accepts either a HuggingFace Hub model id (e.g.
    ``"mistralai/Mistral-7B-Instruct-v0.2"``) or a full local checkpoint
    path (e.g. ``/scratch/.../org/model``). In the latter case the canonical
    ``org/model`` id is derived from the path so that model-family checks
    (dtype, thinking-mode handling) still work.

    All inference kwargs (``temperature``, ``top_p``, ``max_new_tokens``,
    ``do_sample``, ``return_full_text``, …) are forwarded to the underlying
    ``transformers.pipeline`` call.
    """

    def __init__(self, model_name: str, use_chat: str, prompt_template: PromptTemplate, *args, **kwargs):
        # If a full local path is given (e.g. /scratch/.../org/model), keep the full path for
        # loading but also derive the canonical HF model ID from the last two components so that
        # model-type detection (dtype, thinking mode) works correctly.
        parts = model_name.split('/')
        if model_name.startswith('/') and len(parts) >= 3:
            self.model_name = parts[-2] + '/' + parts[-1]
            self.model_path = model_name
        else:
            self.model_name = model_name
            self.model_path = model_name

        hf_cache = os.environ.get('HF_HUB_CACHE')
        hf_token = os.environ.get('HF_HOME')

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir=hf_cache, token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Derive target device from kwargs so we can load directly there (avoids
        # CPU staging + move, which temporarily doubles memory usage).
        device = kwargs.get('device', None)
        dtype = torch.bfloat16 if 'gemma' in self.model_name.lower() else torch.float16
        load_kwargs: dict = dict(cache_dir=hf_cache, token=hf_token, torch_dtype=dtype)
        if device is not None:
            load_kwargs['device_map'] = {'': device}

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)

        self._use_chat_str = use_chat  # stored for rebuilding after PEFT training
        self.use_chat = use_chat.lower() == 'true'
        self.task = kwargs.get('task', 'text-generation')

        # Disable internal thinking mode for models that expose it via apply_chat_template
        thinking_model_indicators = ['thinking', 'reasoning', 'qwen3']
        if any(indicator in self.model_name.lower() for indicator in thinking_model_indicators):
            print(f"Thinking model detected: {self.model_name}. Disabling thinking mode.")
            original_apply_chat_template = self.tokenizer.apply_chat_template
            def apply_chat_template_no_thinking(*args, **kw):
                kw['enable_thinking'] = False
                return original_apply_chat_template(*args, **kw)
            self.tokenizer.apply_chat_template = apply_chat_template_no_thinking

        # When device_map is set on the model, pipeline must not receive `device`
        # (HuggingFace raises an error if both are specified).
        pipe_kwargs = {k: v for k, v in kwargs.items() if k != 'device'} if device is not None else kwargs
        self.pipe = pipeline(self.task, model=self.model, tokenizer=self.tokenizer, **pipe_kwargs)
        if self.use_chat:
            self.llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=self.pipe), cache=False)
        else:
            self.llm = HuggingFacePipeline(pipeline=self.pipe)

        self.prompt_template = prompt_template
        self.chain = self.prompt_template | self.llm
        self.kwargs = kwargs  # kept so agents can rebuild this object after PEFT training
        self.logger = setup_logger("debate.log")

    def separate_thinking_and_response(self, text: str) -> Tuple[str, str]:
        """Split ``<think>...</think>`` chain-of-thought from the final answer."""
        last_close = text.rfind('</think>')
        if last_close != -1:
            thinking_content = text[:last_close].replace('<think>', '').strip()
            response_content = text[last_close + len('</think>'):].strip()
        else:
            thinking_content = ""
            response_content = text.strip()
        return thinking_content, response_content

    def __call__(self, prompt_inputs: Union[str, dict]) -> Union[str, Tuple[str, str]]:
        """Invoke the chain and post-process the output.

        Returns the raw text, or ``(thinking, response)`` when the model
        emits a chain-of-thought ``<think>...</think>`` block.
        """
        if self.use_chat:
            full_response = self.chain.invoke(prompt_inputs).content
        else:
            full_response = self.chain.invoke(prompt_inputs)

        if '<think>' in full_response or '</think>' in full_response:
            thinking, response = self.separate_thinking_and_response(full_response)
            return thinking, response
        return full_response


# loads llama-guard from HuggingFace
class LlamaGuard:
    """Meta LlamaGuard wrapper used as a safety judge.

    Unlike the generic LLM wrappers this one accepts a single ``text``
    string and returns LlamaGuard's verdict ("safe" / "unsafe …") directly,
    bypassing LangChain entirely. ``kwargs`` are forwarded as
    ``model.generate`` kwargs (e.g. ``max_new_tokens``, ``pad_token_id``).
    """

    def __init__(self, model_name: str, device: str = 'cuda', torch_dtype=torch.bfloat16, **kwargs):
        self.model_name = model_name

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device(device)
        self.generate_kwargs = kwargs

    def __call__(self, text: str):
        """Classify ``text`` and return the model's verdict string."""
        chat = [{"role": "user", "content": text}]
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids=input_ids, **self.generate_kwargs)
        prompt_len = input_ids.shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


# TODO: fix independency error
class AnyVLLM:
    """Experimental vLLM wrapper. The :class:`AnyHuggingFace` path is preferred
    for now; this class is kept for benchmarking and not used by default.
    """

    def __init__(self, model_name: str, prompt_template: PromptTemplate, *args, **kwargs):
        self.model_name = model_name

        self.llm = VLLM(
                model=model_name,
                trust_remote_code=True,  # mandatory for hf models
                max_new_tokens=128,
                top_k=10,
                top_p=0.95,
                temperature=0.8,
                dtype='float16',
            )

        self.prompt_template = prompt_template
        self.chain = self.prompt_template | self.llm

    def __call__(self, prompt_inputs: Union[str, dict]):
        return self.chain.invoke(prompt_inputs)
