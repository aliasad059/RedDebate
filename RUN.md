# RedDebate

RedDebate is a multi-agent debate framework for studying and improving the
safety of large language models. A pool of LLMs argues over questions drawn
from safety-evaluation corpora (HarmBench, CoSafe, Aegis 2, XSTest,
ToxicChat, …), an evaluator agent (LlamaGuard, an LLM judge, or the OpenAI
moderation API) labels each response, and a feedback agent distils every
unsafe debate into a textual rule that future debates must follow. Memory
can be ephemeral (per-question), array-based, vector-indexed, or learned via
LoRA fine-tuning.

The framework also supports several debate flavours and single-agent
baselines:

| Mode                                  | Trigger flags                                  |
|---------------------------------------| ---------------------------------------------- |
| Peer Refinement Debate (PReD)         | `--models …` (≥ 1 debater)                     |
| Devil–Angel Refinement Debate (DAReD) | `--models …`, `--devil_model`, `--angel_model` |
| Socratic Refinement Debate (SReD)     | `--models …`, `--questioner_model`             |
| Self-Critique (SC)                    | `--self_critique_model …`             |
| Best-of-N baseline                    | `python redDebate/bon.py` (standalone script)  |


Memory backends:

| Backend                                | Trigger flags                                                            |
|----------------------------------------|--------------------------------------------------------------------------|
| Short-term memory (per-question history) | default (always enabled)                                                 |
| Textual Long-term memory (TLTM)        | `--textual_memory_index <name>`                                          |
| Continuous Long-Term Memory (CLTM)     | `--peft_memory --peft_directory <subdir>`                                |
| Unified Long-Term Memory (TLTM+CLTM)   | `--textual_memory_index <name>` `--peft_memory --peft_directory <subdir>` |
| Guardrails Long-Term Memory (GLTM)     | Not triggerd by main flags. More details below.                          |


---

## Installation

```bash
git clone <repo-url>
cd RedDebate
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` (or export in your shell) for whichever providers you plan to
use:

```bash
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=...           # used by openrouter:* models
FEATHERLESS_API_KEY=...          # used by fai:* models
DEEPSEEK_API_KEY=...             # used by deepseek:* models
GOOGLE_API_KEY=...               # used by google:* models
PINECONE_API_KEY=...             # only for --textual_memory_index
HF_HUB_CACHE=/path/to/hf/cache   # used by HuggingFace models / PEFT
HF_HOME=hf_xxx                   # HF token (re-used as cache key)
WANDB_API_KEY=...                # optional, for metric logging
```

---

## Model strings

Every model argument follows the same syntax:

```
<type>:<model_name_or_path>[:<use_chat>]
```

* `type`        — one of `openai`, `openrouter`, `fai`, `deepseek`, `google`, `huggingface`.
* `model_name`  — provider model id or a local checkpoint path.
* `use_chat`    — `true` (default) or `false`; flips between chat and completion endpoints.

Examples:

```
openai:gpt-4o-mini:true
openrouter:qwen/qwen3-30b-a3b:true
huggingface:mistralai/Mistral-7B-Instruct-v0.2:true
huggingface:/scratch/models/google/gemma-3-12b-it:true
google:gemini-2.0-flash:true
```

Special evaluator: `openai:moderation` routes safety calls through the OpenAI
moderation API instead of an LLM judge.

---

## Dataset strings

Dataset arguments use the same prefix style:

```
<dataset_name>:<dataset_path>
```

Supported names: `harmbench`, `cosafe`, `aegis2`, `strongreject`, `xstest`,
`triviaqa`, `toxicchat`, `hhrlhf`, `saferdialogues`, `dummy`.

Examples:

```
harmbench:datasets/harmbench/data/behavior_datasets/harmbench_behaviors_text_all.csv
cosafe:"datasets/cosafe/CoSafe datasets/formatted_cosafe.csv"
aegis2:datasets/aegis2/test.csv
```

---

## Running a debate

The entry point is `main.py`. The minimum required arguments are `--models`
(or `--self_critique_model`) and `--datasets`.

### 1. Simple multi-agent debate

Two HuggingFace debaters, LlamaGuard as the safety judge, GPT-4o-mini for
feedback, 3 rounds on HarmBench:

```bash
python main.py \
  --models huggingface:mistralai/Mistral-7B-Instruct-v0.2:true \
           huggingface:google/gemma-3-12b-it:true \
  --evaluator huggingface:meta-llama/Llama-Guard-3-8B:false \
  --feedback_generator openai:gpt-4o-mini:true \
  --datasets harmbench:datasets/harmbench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --debate_rounds 3 \
  --llamaguard_cuda_device 0 \
  --output_file debate.log \
  --checkpoint_dir checkpoints/simple_debate
```

Three OpenRouter debaters via OpenAI-compatible API:

```bash
python main.py \
  --models openrouter:qwen/qwen3-30b-a3b:true \
           openrouter:meta-llama/llama-3.3-70b-instruct:true \
           openrouter:deepseek/deepseek-chat:true \
  --evaluator openai:moderation \
  --feedback_generator openai:gpt-4o-mini:true \
  --datasets cosafe:"datasets/cosafe/CoSafe datasets/formatted_cosafe.csv" \
  --debate_rounds 2 \
  --max_total_debates 200
```

### 2. Devil / Angel debate

Add an adversarial (`--devil_model`) and a supportive (`--angel_model`) agent
to a baseline debate; both inject a turn after every round except the last:

```bash
python main.py \
  --models huggingface:mistralai/Mistral-7B-Instruct-v0.2:true \
           huggingface:google/gemma-3-12b-it:true \
  --devil_model openai:gpt-4o-mini:true \
  --angel_model openai:gpt-4o-mini:true \
  --evaluator huggingface:meta-llama/Llama-Guard-3-8B:false \
  --feedback_generator openai:gpt-4o-mini:true \
  --datasets harmbench:datasets/harmbench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --debate_rounds 3 \
  --checkpoint_dir checkpoints/devil_angel
```

### 3. Socratic debate

A Socratic questioner probes the debaters after each round (except the
last):

```bash
python main.py \
  --models huggingface:mistralai/Mistral-7B-Instruct-v0.2:true \
           huggingface:google/gemma-3-12b-it:true \
  --questioner_model openai:gpt-4o-mini:true \
  --evaluator huggingface:meta-llama/Llama-Guard-3-8B:false \
  --feedback_generator openai:gpt-4o-mini:true \
  --datasets harmbench:datasets/harmbench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --debate_rounds 3 \
  --checkpoint_dir checkpoints/socratic
```

### 4. Constitutional self-critique

A single agent answers → critiques itself against a randomly-sampled
constitutional rule → revises. The `--self_critique_model` flag *replaces*
the regular debate loop (so no `--models` is needed):

```bash
python main.py \
  --self_critique_model huggingface:google/gemma-3-12b-it:true \
  --evaluator huggingface:meta-llama/Llama-Guard-3-8B:false \
  --feedback_generator openai:gpt-4o-mini:true \
  --datasets harmbench:datasets/harmbench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --debate_rounds 1 \
  --checkpoint_dir checkpoints/self_critique
```

Constitutional rules are loaded from
`ConstitutionalHarmlessnessPaper/prompts/CritiqueRevisionInstructions.json`
(edit the path in `redDebate/self_critique.py` if your checkout lives
elsewhere).

### 5. Best-of-N baseline (standalone)

`redDebate/bon.py` is a self-contained script — edit the module-level
`MODEL_NAME`, `N`, `INPUT_FILE`, `OUTPUT_FILE` and device constants, then
run:

```bash
python redDebate/bon.py
```

It samples `N` completions per HarmBench prompt, judges each with
LlamaGuard, and writes a JSON file with the per-sample scores plus an
overall best-of-N label (safe iff any of the `N` samples is safe).

### 6. Human in the loop

Add `--human_in_the_loop <K>` to inject `K` `HumanAgent` participants per
round; each is prompted on stdin before the LLM debaters run.

```bash
python main.py \
  --models openai:gpt-4o-mini:true \
  --evaluator openai:moderation \
  --datasets dummy:test.jsonl \
  --human_in_the_loop 1
```

---

## Memory integrations

### Short-term + array long-term (default)

No extra flag is needed. The short-term memory holds the per-question
transcript; the long-term memory is an in-memory list of feedback rules
shared across questions in the same run and saved to
`results/<timestamp>/long_term_memory.txt` at the end.

### Pinecone vector store (semantic long-term memory)

`--textual_memory_index <index_name>` switches the long-term memory to a
Pinecone-backed semantic store. Before each question, the top-5 most-
similar prior feedbacks (by OpenAI `text-embedding-3-large`) are retrieved
and shown to the agents. Requires `PINECONE_API_KEY`.

```bash
python main.py \
  --models huggingface:mistralai/Mistral-7B-Instruct-v0.2:true \
  --evaluator openai:moderation \
  --feedback_generator openai:gpt-4o-mini:true \
  --datasets harmbench:datasets/harmbench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --textual_memory_index red-debate-memory \
  --debate_rounds 3 \
  --checkpoint_dir checkpoints/vector_mem
```

### Pre-seeded long-term memory (testing)

When the `dummy` loader is used, each question's `memory` field is
loaded into the array memory before the debate runs. Useful for
ablations and unit-style tests.

### PEFT / LoRA fine-tuning (learned long-term memory)

`--peft_memory` wraps every HuggingFace debater in `PEFTDebateAgent`. After
every `--train_steps` newly-accumulated feedback samples, each HF debater is
LoRA-fine-tuned in-place on the running feedback buffer, merged back into a
plain checkpoint and reloaded for inference. LoRA checkpoints are saved
under `$HF_HUB_CACHE/<peft_directory><model_name><agent_name>_LoRA`.

```bash
python main.py \
  --models huggingface:mistralai/Mistral-7B-Instruct-v0.2:true \
           huggingface:google/gemma-3-12b-it:true \
  --evaluator huggingface:meta-llama/Llama-Guard-3-8B:false \
  --feedback_generator openai:gpt-4o-mini:true \
  --datasets harmbench:datasets/harmbench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --debate_rounds 3 \
  --peft_memory \
  --peft_directory peft_runs/run1/ \
  --train_steps 10 \
  --checkpoint_dir checkpoints/peft_run1
```

To run inference with a previously trained LoRA model, simply pass the
merged checkpoint path as a `huggingface:` model in `--models` — no
`--peft_memory` flag needed.

### Guardrails Long-Term Memory (GLTM)

GLTM compiles accumulated feedback into NeMo Guardrails Colang flows and
enforces them at inference time. Because it depends on (a fork of) NeMo
Guardrails and is not wired into the main `python main.py` driver, GLTM
lives in its own sub-package with separate evaluation scripts and a
companion guide:

* [`guardrails_memory/README.md`](guardrails_memory/README.md) — full GLTM
  pipeline (feedback → Colang translation → NeMo Guardrails app →
  per-dataset evaluation), plus the pre-generated flows we used in the
  paper under `guardrails_memory/generated_guardrails/`.

Once a Colang file has been generated (or copied from
`generated_guardrails/`) into `guardrails_memory/abc_v2/rails/disallowed.co`,
evaluate it on HarmBench or CoSafe with:

```bash
python guardrails_memory/eval_harmbench.py \
  --input_csv  datasets/harmbench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --config_path guardrails_memory/abc_v2 \
  --output_csv  results/gltm/harmbench.csv

python guardrails_memory/eval_cosafe.py \
  --input_json datasets/cosafe/CoSafe\ datasets/formatted_cosafe_parsed.json \
  --config_path guardrails_memory/abc_v2 \
  --output_json results/gltm/cosafe.json
```

---

## Outputs

For each run RedDebate writes:

* `debate.log` (or whatever `--output_file` points at) — full log of every
  prompt and response.
* `<checkpoint_dir>/<dataset>_q<idx>.json` — one JSON per finished debate.
* `<checkpoint_dir>/checkpoint.json` — last-processed index per dataset
  (used to resume the run).
* `<checkpoint_dir>/feedback_dict.json` — accumulated feedback samples
  used for PEFT training (only when `--peft_memory` is set).
* `results/<timestamp>/long_term_memory.txt` — final long-term memory.
* `results/<timestamp>/metrics.json` — error / agreement / confusion /
  confidence / diversity / response-length statistics
  (see `redDebate/metrics.py`).
* Weights & Biases artifact (when `WANDB_API_KEY` is set).

Resuming a run is automatic: pass the same `--checkpoint_dir`; previously
completed debates are skipped and the existing transcripts are reloaded so
the final metrics still cover the entire dataset.

---

## Repository layout

```
main.py                          # CLI entry point
redDebate/
  run.py                         # orchestration (run_debate, init_agents, checkpoint I/O)
  agents.py                      # DebateAgent, PEFTDebateAgent, DevilAngelAgent,
                                 #   EvalAgent, FeedbackAgent, SelfCriticAgent, HumanAgent
  debate.py                      # Debate, SocraticDebate, DevilAngelDebate
  self_critique.py               # SelfCritique driver
  memory.py                      # ShortTermMemory, LongTermMemory, VectorStoreMemory
  llm.py                         # AnyOpenAILLM, AnyHuggingFace, AnyGoogleGenerativeAI,
                                 #   LlamaGuard, AnyVLLM
  dataloader.py                  # All dataset loaders + load_datasets dispatch
  debate_prompts.py              # PromptTemplates used by every agent role
  metrics.py                     # Aggregate metrics + W&B logging
  bon.py                         # Standalone Best-of-N safety baseline
  eval_safety.py                 # Stand-alone LlamaGuard scoring of pre-existing CSVs
  util.py                        # Shared logger setup
datasets/                        # HarmBench, CoSafe, Aegis 2, ... (see preprocess scripts)
```
