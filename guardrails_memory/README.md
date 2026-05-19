# Guardrails Long-Term Memory (GLTM)

The Guardrails Long-Term Memory variant stores accumulated debate feedback as
**NeMo Guardrails dialog rails** (Colang flows) rather than as textual rules or
LoRA weights. At inference time every user prompt is matched against the
generated flows; if it triggers a flow, the bot refuses with a pre-defined
response, otherwise the request is forwarded to the underlying LLM.

GLTM is kept in this separate sub-package because it depends on NeMo
Guardrails (and a small fork of it), which is hard to integrate with the
interactive debate loop. The pipeline is therefore *offline*: a debate run
produces feedbacks → those feedbacks are converted to Colang flows by an
LLM translator → the flows are loaded into a NeMo Guardrails app that is
then evaluated against the dataset.

This directory also ships **the exact Colang flows we generated for the
HarmBench and CoSafe experiments** so you can reproduce the paper numbers
without re-running the translation step.

---

## Directory layout

```
guardrails_memory/
├── README.md                       # this file
├── prompts/
│   └── json.txt                    # feedback → JSON intent translator prompt
├── abc_v2/                         # base NeMo Guardrails config (Colang 2.x)
│   ├── config.yml                  # model + rails config
│   ├── main.co
│   ├── rails.co
│   ├── prompts.yml
│   └── rails/
│       └── disallowed.co           # drop generated flows here before running
├── generated_guardrails/           # pre-built flows used in the paper
│   ├── harmbench/
│   │   ├── PReD_MistralLLamaPhi/code.co
│   │   ├── SReD_MistralLLamaPhi/code.co
│   │   └── SReD_GemmaQwenR1/code.co
│   └── cosafe/
│       ├── PReD_MistralLLamaPhi/code.co
│       ├── SReD_MistralLLamaPhi/code.co
│       └── SReD_GemmaQwenR1/code.co
├── eval_harmbench.py               # GLTM evaluation driver for HarmBench
└── eval_cosafe.py                  # GLTM evaluation driver for CoSafe
```

---

## Prerequisites

GLTM relies on a small fork of NeMo Guardrails that ships the Colang 2.x
extensions used by the generated flows. Install it from source:

```bash
git clone --branch reddebate https://github.com/radinshayanfar/NeMo-Guardrails.git
cd NeMo-Guardrails
pip install -e .
```

You will also need the providers used by `abc_v2/config.yml` — by default
the config talks to a local OpenAI-compatible endpoint on
`http://localhost:5001/v1`; edit `engine`, `model` and `parameters.base_url`
to point at OpenAI / OpenRouter / a local vLLM server / etc.

---

## Pipeline

```
debate feedback (LongTermMemory) ──► [LLM translator] ──► generated Colang flows
                                                                  │
                                                                  ▼
                                                       abc_v2/rails/disallowed.co
                                                                  │
                                                                  ▼
                                                        NeMo Guardrails app
                                                                  │
                                                                  ▼
                                       eval_harmbench.py / eval_cosafe.py
```

### 1. Collect feedback from a debate run

Run any of the standard RedDebate flavours (PReD / DAReD / SReD / SC) with a
feedback generator, as documented in the [main README](../README.md). The
resulting `results/<timestamp>/long_term_memory.txt` (or the per-feedback
entries inside the saved debate JSONs) is the input to the next step.

### 2. Translate feedback into Colang flows

Use the prompts in `prompts/` to ask an LLM (we used GPT-4o-mini) to convert
each feedback entry into either:

* `prompts/code.txt` — direct Colang code (preferred; produces a complete
  `flow user … / bot refuse … / @active flow` triple).
* `prompts/json.txt` — an intermediate JSON that captures only the trigger
  expression and a few example utterances, useful when you want to merge
  many feedbacks into a single normalized rule set before emitting Colang.

Each prompt expects the placeholders `<<input>>` (the original user prompt
that produced the unsafe debate) and `<<feedback>>` (the feedback text) to
be filled in before sending it to the translator LLM.

Concatenate every produced Colang flow into a single `.co` file. Examples
of what these files look like — and what we used in the paper — are under
`generated_guardrails/`.

### 3. Plug the flows into the NeMo Guardrails app

Copy (or symlink) the generated `code.co` over the placeholder file in the
base config:

```bash
cp generated_guardrails/harmbench/SReD_GemmaQwenR1/code.co abc_v2/rails/disallowed.co
```

`abc_v2/rails/disallowed.co` is the only file the base app reads for
dialog rails; replacing it switches which feedbacks are enforced.

### 4. Evaluate on a dataset

Two drivers are provided. Both share the same CLI surface and write the
LLM's final response back into the dataset under an `llm_response` column /
field. They cache progress between runs so a long evaluation can be
resumed after a crash.

**HarmBench** (CSV input, CSV output):

```bash
python guardrails_memory/eval_harmbench.py \
  --input_csv  datasets/harmbench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --config_path guardrails_memory/abc_v2 \
  --output_csv  results/gltm/harmbench_SReD_GemmaQwenR1.csv
```

Optional: pass `--split_file <path>` to limit evaluation to the row indices
listed (one integer per line); this is how the parallel HarmBench splits
in `datasets/harmbench/.../splited/` are consumed.

**CoSafe** (JSON in/out, multi-turn-aware):

```bash
python guardrails_memory/eval_cosafe.py \
  --input_json datasets/cosafe/CoSafe\ datasets/formatted_cosafe_parsed.json \
  --config_path guardrails_memory/abc_v2 \
  --output_json results/gltm/cosafe_SReD_GemmaQwenR1.json \
  --limit 200
```

`eval_cosafe.py` injects an `Evaluator` helper into `RailsConfig` so the
prior conversation turns are formatted as `user said …` / `bot said …`
actions and made available to the Colang flow runtime. `--limit` caps the
number of samples (useful for smoke tests); `--split_file` works the same
way as in the HarmBench driver.

### 5. Score the safety of the responses

The outputs of step 4 are plain CSVs / JSONs with an extra `llm_response`
column / field. Score them with the same LlamaGuard pipeline you use for
the rest of the framework, e.g. `redDebate/eval_safety.py`.

---

## Reproducing the paper

To reproduce one of the reported GLTM results, skip step 1–2 and reuse the
pre-built flows under `generated_guardrails/<dataset>/<run>/code.co`:

- **HarmBench, PReD** (Mistral/Llama/Phi) — `generated_guardrails/harmbench/PReD_MistralLLamaPhi/code.co`
- **HarmBench, SReD** (Mistral/Llama/Phi) — `generated_guardrails/harmbench/SReD_MistralLLamaPhi/code.co`
- **HarmBench, SReD** (Gemma/Qwen/R1) — `generated_guardrails/harmbench/SReD_GemmaQwenR1/code.co`
- **CoSafe, PReD** (Mistral/Llama/Phi) — `generated_guardrails/cosafe/PReD_MistralLLamaPhi/code.co`
- **CoSafe, SReD** (Mistral/Llama/Phi) — `generated_guardrails/cosafe/SReD_MistralLLamaPhi/code.co`
- **CoSafe, SReD** (Gemma/Qwen/R1) — `generated_guardrails/cosafe/SReD_GemmaQwenR1/code.co`

For each variant: copy the `code.co` over `abc_v2/rails/disallowed.co`,
run the corresponding `eval_*.py` driver against the matching dataset, and
score the resulting responses.

---

## Tips & gotchas

* **Backend LLM:** `abc_v2/config.yml` defaults to a local
  `http://localhost:5001/v1` endpoint. If you don't have a local server,
  switch `engine` to `openai`/`openrouter` and set `parameters.base_url`
  accordingly. The model under guardrails has to be a chat model.
* **Colang version:** `colang_version: "2.x"` is required — the older
  1.x syntax is **not** supported by the generated flows.
* **Cache files:** both drivers write `cache.json` (HarmBench) or
  `cosafe_cache.json` (CoSafe) while running, and delete the file once
  evaluation finishes. If you abort the run, simply re-run the same
  command; the cache is picked up and only the missing rows are
  re-processed.
* **Empty completions:** when the underlying LLM call fails or returns an
  empty string, the driver retries up to 3 times before writing
  `<<<ERROR>>> …` into `llm_response`. Filter those rows out before
  computing safety metrics.
