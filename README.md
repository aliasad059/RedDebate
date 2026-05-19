# RedDebate

**Safer Responses Through Multi-Agent Red Teaming Debates** *(ICML 2026)*

> *"It is better to change an opinion than to persist in a wrong one."* — Socrates

RedDebate is a fully automated framework that lets a pool of Large Language
Models **red-team each other through debate**. Rather than treating a
model's reply as a final answer, RedDebate frames it as a claim to be
tested, challenged, and iteratively refined by other agents. An
independent evaluator flags unsafe behaviour, a feedback agent distils the
incident into a safety lesson, and **long-term memory** carries those
lessons into every future debate — so the system keeps improving itself
without any human in the loop.

<div align="center">
  <img src="assets/overview.png" alt="RedDebate framework overview" width="500">
</div>


*RedDebate: multiple agents debate a red-teaming prompt across several rounds, an evaluator flags unsafe patterns, and distilled safety insights are stored in memory to prevent future mistakes.*

---

## Why debate?

Existing AI-safety pipelines lean on costly human red-teaming or
single-model self-assessment, both of which struggle to scale and miss
subtle failure modes. Training-time alignment (RLHF, supervised safety
tuning, rule-based guardrails) is also bounded by training-signal coverage
and is vulnerable to reward hacking and distribution shift, while many
unsafe behaviours only emerge **at inference time** under novel or
adversarial prompts.

RedDebate is an **inference-time** complement to those methods: by forcing
LLMs to *defend, challenge and revise* one another's reasoning, the
framework surfaces hidden assumptions and counterexamples that no single
agent (and no static guardrail) would catch on its own.

## What's in the framework

RedDebate is built around three orthogonal axes that can be combined freely
on the command line:

**Debate scenarios**

- **Peer Refinement Debate (PReD)** — two or more debaters argue and revise.
- **Devil–Angel Refinement Debate (DAReD)** — an adversary and an advocate join the debate to stress-test each round.
- **Socratic Refinement Debate (SReD)** — a Socratic questioner probes the debaters between rounds.
- **Self-Critique (SC)** — single-agent constitutional answer → critique → revise loop, included as a baseline.
- **Best-of-N** — additional single-model sampling baseline.

**Long-term memory modules**

- **Textual Long-Term Memory (TLTM)** — feedback rules stored verbatim, either as an in-memory list or in a Pinecone vector index for semantic retrieval.
- **Continuous Long-Term Memory (CLTM)** — feedback is consolidated into the debaters' weights via on-the-fly LoRA fine-tuning.
- **Unified Long-Term Memory (TLTM + CLTM)** — both run together.
- **Guardrails Long-Term Memory (GLTM)** — feedback is compiled into NeMo Guardrails Colang flows and enforced as dialog rails at inference time. See [`guardrails_memory/README.md`](guardrails_memory/README.md).

**Datasets & judges**

Safety datasets out of the box: HarmBench, CoSafe, Aegis 2, WildJailbreak,
XSTest and TriviaQA. The evaluator can be LlamaGuard, the OpenAI moderation API, or
any LLM-as-judge.

## Running the code

The full CLI surface — model strings, dataset strings, every flag for
every debate flavour and every memory backend, plus checkpointing,
resumption and Weights & Biases logging — is documented in
**[RUN.md](RUN.md)**.

Quick start (a peer debate between three HuggingFace models on HarmBench,
LlamaGuard as judge, GPT-4o-mini as the feedback generator):

```bash
python main.py \
  --models huggingface:mistralai/Mistral-7B-Instruct-v0.2:true \
           huggingface:meta-llama/Llama-3.2-3B-Instruct:true \
           huggingface:microsoft/Phi-3.5-mini-instruct:true \
  --evaluator huggingface:meta-llama/Llama-Guard-3-8B:false \
  --feedback_generator openai:gpt-4o-mini:true \
  --datasets harmbench:datasets/harmbench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --debate_rounds 3 \
  --textual_memory_index red-debate-memory
```

See [RUN.md](RUN.md) for installation, environment variables, and recipes
for every supported debate / memory combination.

## Repository layout

```
main.py                          # CLI entry point
redDebate/
  run.py                         # orchestration, checkpointing, W&B logging
  agents.py                      # Debate / DevilAngel / SelfCritic / PEFT agents
  debate.py                      # Debate, SocraticDebate, DevilAngelDebate
  self_critique.py               # Constitutional self-critique driver
  memory.py                      # ShortTerm / LongTerm / VectorStore memories
  llm.py                         # Provider-agnostic LLM wrappers + LlamaGuard
  dataloader.py                  # Safety-dataset loaders
  debate_prompts.py              # PromptTemplates for every agent role
  metrics.py                     # Error / agreement / confusion / diversity metrics
  bon.py                         # Standalone Best-of-N baseline
guardrails_memory/               # GLTM (NeMo Guardrails) pipeline + paper-ready flows
datasets/                        # HarmBench, CoSafe, ... + preprocess scripts
RUN.md                           # Full technical / CLI documentation
```

## Citation

If you find this work interesting or use the code in your research, please consider citing our paper — it really helps!

```bibtex
@inproceedings{
  asad2026reddebate,
  title     = {RedDebate: Safer Responses Through Multi-Agent Red Teaming Debates},
  author    = {Ali Asad and Stephen Obadinma and Radin Shayanfar and Xiaodan Zhu},
  booktitle = {Forty-third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=79fSrxFKKx}
}
```

## Contact & Contributing

Questions, ideas, or feedback? Feel free to reach out at **ali.asad@queensu.ca** — happy to chat. Pull requests are also very welcome!
