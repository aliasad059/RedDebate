from typing import Union

import torch
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_openai import ChatOpenAI, OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import VLLM

# loads a model from OpenAI
class AnyOpenAILLM:
    def __init__(self, model_name: str, use_chat: str, prompt_template: PromptTemplate, *args, **kwargs):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.use_chat = use_chat.lower() == 'true'

        if self.use_chat:
            self.llm = ChatOpenAI(model=self.model_name, cache=False, **kwargs)
        else:
            self.llm = OpenAI(model=self.model_name, **kwargs)

        # Create a chain
        self.chain = self.prompt_template | self.llm

    def __call__(self, prompt_inputs: Union[str, dict]):
        if self.use_chat:
            return self.chain.invoke(prompt_inputs).content
        else:
            return self.chain.invoke(prompt_inputs)

# loads a model from GoogleGenerativeAI
class AnyGoogleGenerativeAI:
    def __init__(self, model_name: str, use_chat: str, prompt_template: PromptTemplate, *args, **kwargs):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.use_chat = use_chat.lower() == 'true'

        if self.use_chat:
            self.llm = ChatGoogleGenerativeAI(model=self.model_name, cache=False, **kwargs)
        else:
            self.llm = GoogleGenerativeAI(model=self.model_name, **kwargs)

        # Create a chain
        self.chain = self.prompt_template | self.llm

    def __call__(self, prompt_inputs: Union[str, dict]):
        if self.use_chat:
            return self.chain.invoke(prompt_inputs).content
        else:
            return self.chain.invoke(prompt_inputs)

# loads a model from HuggingFace
class AnyHuggingFace:
    def __init__(self, model_name: str, use_chat: str, prompt_template: PromptTemplate, *args, **kwargs):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.use_chat = use_chat.lower() == 'true'

        self.task = kwargs.get('task', 'text-generation')
        self.pipe = pipeline(self.task, model=self.model, tokenizer=self.tokenizer, **kwargs)
        if self.use_chat:
            self.llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=self.pipe), cache=False)
        else:
            self.llm = HuggingFacePipeline(pipeline=self.pipe)

        self.prompt_template = prompt_template
        self.chain = self.prompt_template | self.llm

    def __call__(self, prompt_inputs: Union[str, dict]):
        if self.use_chat:
            return self.chain.invoke(prompt_inputs).content
        else:
            return self.chain.invoke(prompt_inputs)

# loads llama-guard from HuggingFace
class LlamaGuard:
    def __init__(self, model_name: str, device: str = 'cuda', torch_dtype=torch.bfloat16, **kwargs):
        self.model_name = model_name

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device(device)
        self.generate_kwargs = kwargs

    def __call__(self, text: str):
        chat = [{"role": "user", "content": text}]
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids=input_ids, **self.generate_kwargs)
        prompt_len = input_ids.shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
