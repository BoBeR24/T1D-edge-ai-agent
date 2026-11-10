# DiabetesAgent

A tool-calling LLM agent for type-1 diabetes data, built to run against **locally
hosted models on modest hardware**. It answers questions about continuous glucose
monitor (CGM) traces, meals, and insulin by *querying* the data through tools
rather than reading it out of the context window.

The repository contains three things:

| | What it is | Entry point |
|---|---|---|
| **Agent** | LangGraph agent + ~30 diabetes tools, as an interactive CLI | `python main.py` |
| **Web UI** | HTTP API and a React chat front-end over the same agent | `python server.py` + `bun dev` |
| **Benchmark** | A 72-task OhioT1DM suite for scoring how well a model drives those tools | `python -m benchmark` |

Everything works against any **OpenAI-compatible endpoint** — LM Studio, Ollama,
vLLM, or the OpenAI API itself.

---

## Table of contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Configuring a model endpoint](#configuring-a-model-endpoint)
- [Running the agent (CLI)](#running-the-agent-cli)
- [Running the web UI](#running-the-web-ui)
- [Running the benchmark](#running-the-benchmark)
- [Analysing benchmark results](#analysing-benchmark-results)
- [The OhioT1DM dataset](#the-ohiot1dm-dataset)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Requirements

- **Python 3.12+**
- **An OpenAI-compatible LLM endpoint** (see below). A tool-calling-capable model
  is required — the agent is useless without function calling.
- **Bun** — only for the web UI ([install](https://bun.sh))
- **The OhioT1DM dataset** — only for the benchmark, and it must be requested
  separately (see [below](#the-ohiot1dm-dataset))

A first run downloads the `all-MiniLM-L6-v2` sentence-transformer (~90 MB) for the
knowledge base. After that everything runs offline apart from the LLM endpoint.

## Setup

```bash
git clone <your-fork-url> DiabetesAgent
cd DiabetesAgent

python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # then edit .env
```

> `requirements.txt` installs the Spoonacular client from GitHub, so `git` must be
> on your PATH.

### Configuration

All configuration lives in [`config.py`](config.py), and **every value can be
overridden by an environment variable of the same name**. `config.py` reads `.env`
at import time, so the usual workflow is to edit `.env` and never touch the Python.

`.env` is git-ignored. **No credentials belong in `config.py`** — it is committed.

Nothing in `.env` is mandatory. Features whose credentials are missing say so
explicitly when you call them rather than failing silently, so you can run the
agent with an LLM endpoint alone.

## Configuring a model endpoint

The agent talks to one OpenAI-compatible `/v1` endpoint. Two settings control it:

```bash
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1   # must end at /v1
MODEL_NAME=qwen3.6-35b-a3b                    # an id the server actually serves
```

The client appends `/chat/completions` itself, so **stop the URL at `/v1`**.
The variable is named for LM Studio for historical reasons, but any backend works:

<details>
<summary><b>LM Studio</b> (default)</summary>

Load a model, then start the server from the *Developer* tab (or `lms server start`).
`MODEL_NAME` must match the model id shown in LM Studio.

```bash
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=lm-studio        # ignored by LM Studio, but must be non-empty
MODEL_NAME=qwen3.6-35b-a3b
```
</details>

<details>
<summary><b>Ollama</b></summary>

Ollama exposes an OpenAI-compatible API at `/v1`. Use a tool-calling model.

```bash
ollama pull qwen3:8b
ollama serve
```
```bash
LM_STUDIO_BASE_URL=http://127.0.0.1:11434/v1
LM_STUDIO_API_KEY=ollama           # ignored, but must be non-empty
MODEL_NAME=qwen3:8b
```
</details>

<details>
<summary><b>vLLM</b></summary>

```bash
vllm serve Qwen/Qwen3-8B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
```
```bash
LM_STUDIO_BASE_URL=http://127.0.0.1:8000/v1
LM_STUDIO_API_KEY=vllm
MODEL_NAME=Qwen/Qwen3-8B-Instruct
```

Tool calling must be enabled explicitly — without `--enable-auto-tool-choice` the
model will never emit a tool call and every task scores zero.
</details>

<details>
<summary><b>OpenAI or another hosted provider</b></summary>

```bash
LM_STUDIO_BASE_URL=https://api.openai.com/v1
LM_STUDIO_API_KEY=sk-...           # a real key
MODEL_NAME=gpt-4.1
```

Sending patient data to a third-party API has obvious privacy implications. The
project is designed for local inference precisely to avoid this.
</details>

### A second endpoint for benchmark judging

The benchmark can route three roles — the **agent under test**, the **LLM judge**,
and the **chat simulator** for multi-turn tasks — to different endpoints. This lets
a small local model be graded by a large remote one.

The second endpoint is configured separately:

```bash
OPENCODE_ZEN_BASE_URL=https://opencode.ai/zen/go/v1
OPENCODE_ZEN_API_KEY=your-key-here
ZEN_DEFAULT_MODEL=deepseek         # alias; see ZEN_MODELS in config.py
```

Select it per role with `--agent-provider`, `--judge-provider`, and
`--simulator-provider` (each `local` or `zen`). It is any OpenAI-compatible
endpoint — point `OPENCODE_ZEN_BASE_URL` at whatever you like.

## Running the agent (CLI)

```bash
source venv/bin/activate
python main.py
```

Options:

```bash
python main.py --model qwen3:8b        # override MODEL_NAME for this session
python main.py --session <session-id>  # resume a saved chat
```

Responses stream token-by-token, and each turn prints latency, time-to-first-token,
and token counts. Conversations are saved automatically and carry multi-turn memory.

The agent has roughly 30 tools spanning:

- **Live CGM** — latest reading, logbook, history (needs LibreLinkUp credentials)
- **Stored CGM** — range queries, summaries, spike detection, nearest-reading lookup
- **Glucose statistics** — time in range, average glucose, estimated A1C
- **Logging** — meals, insulin, glucose, with automatic CGM context attached
- **Knowledge base** — semantic search over logged events, past reports, and patterns
- **Insulin** — insulin-to-carb ratios and dose calculation
- **Nutrition** — food and glycemic-index lookup (Spoonacular)

### Live CGM data (optional)

To sync from a real LibreLink account, set `LIBRE_EMAIL` / `LIBRE_PASSWORD` in
`.env`. Without them the live-CGM tools report that they're unconfigured and
everything else works normally. The benchmark never uses this path.

### Scheduled reports

```bash
python run_scheduler.py
```

Generates a glucose report on the `REPORT_CRON` schedule (default: Mondays 09:00)
into `REPORTS_DIR`. Reports contain real readings and are git-ignored.

## Running the web UI

The UI is a React app talking to the Python HTTP API. **Both must be running.**

**1. Start the API** (terminal 1):

```bash
source venv/bin/activate
python server.py                    # binds 127.0.0.1:8000
```

Useful flags: `--host`, `--port`, `--model`, `--cors-origin`.

**2. Start the front-end** (terminal 2):

```bash
cd web
bun install
bun dev
```

Open the URL Bun prints (<http://localhost:3001> by default; override with `PORT`).

The front-end targets `http://localhost:8000/api`, set as `API_BASE` in
[`web/src/App.tsx`](web/src/App.tsx) — change it there if you move the API, and
pass the front-end's origin to `server.py --cors-origin`.

For a production build: `bun run build` (outputs to `web/dist`), then `bun start`.

The UI provides chat with streamed tool calls and metrics, a report browser, and a
session browser. The API exposes `/api/chat`, `/api/report[s]`, `/api/sessions`,
plus `/api/cgm/*` and `/api/kb/*` routes.

## Running the benchmark

The benchmark measures how reliably a model drives the agent's tools to reach
*verified* answers. Ground truth is computed directly from each patient's data, so
scoring is grounded in real numbers rather than a hand-written reference response.

> Requires the OhioT1DM dataset — see [below](#the-ohiot1dm-dataset).

```bash
source venv/bin/activate
python -m benchmark --model qwen3:8b
```

Results are written to `benchmark_results_v2/<model>/` as:

- `scores_<model>_<timestamp>.jsonl` — per-task scores
- `traces_<model>_<timestamp>.jsonl` — full tool-call traces
- `summary_<model>_<timestamp>.{json,md}` — aggregate summary

### The task suite

72 tasks (`benchmark/tasks_v2.json`) over four OhioT1DM patients (540, 544, 559,
584), spanning 17 categories — lookup, summary, analysis, time-in-range,
time-above-range, hypoglycemia severity, correlation, pattern analysis, missed-bolus
detection, weekly trend, food/glycemic index, memory, logging, clarification,
patient education, insulin-adjustment dialogue, and out-of-distribution probes.

They are graded easy/medium/hard; 63 are single-turn and 9 are multi-turn dialogues
driven by a simulated user. Each patient gets a freshly built, isolated SQLite
database and a patient-specific system prompt, so runs cannot contaminate each other.

Out-of-distribution tasks ask for data outside the patient's coverage window and
check the model says so instead of fabricating values.

### Scoring

Each task is scored on four weighted dimensions (per-task weights in the task file):

| Dimension | What it measures |
|---|---|
| `tool_correctness` | Were the right tools called? |
| `argument_correctness` | Were they called with the right arguments (e.g. date ranges)? |
| `numeric_groundedness` | Do the numbers match ground truth computed from the DB? |
| `structured_conclusion` | Did the model emit a well-formed `log_conclusion`? |

An optional **LLM judge** adds prose dimensions — groundedness, completeness,
clarity, and clinical caution:

```bash
python -m benchmark --model qwen3:8b --judge-model gpt-4.1 --judge-provider zen
```
### Skills (oracle mode)

A *skill* is a short, step-by-step procedure for one type of task, stored in a
`SKILL.md` file. In **oracle mode** the benchmark picks the skill from the task's
`category`, appends its instructions to the system prompt, and shows the model only
the tools the skill declares (plus `log_conclusion`). Tasks whose category has no
skill run exactly as the baseline. It is called "oracle" because the category is
known in the benchmark but not in real use: results are the upper bound for any
automatic skill-selection mechanism.

```bash
python -m benchmark --model qwen3:8b --skills oracle                     # skills on
python -m benchmark --model qwen3:8b --skills none                       # baseline (default)
```

Before each task that uses a skill the log prints
`[benchmark] Oracle skill '<name>' for <task_id>`, and the trace records `"skill"`.

#### Adding one or more skills

For each new skill:

1. **Create the skill file** `benchmark/skills/<skill-name>/SKILL.md`:

   ```markdown
   ---
   name: <skill-name>
   description: One line saying when the skill applies.
   tools: [tool_a, tool_b]
   ---
   Goal: ...

   Steps:
   1. ...
   ```

   - `name` must match the folder name.
   - `tools` lists the only data tools the model will see. Do not list
     `log_conclusion`: it is always added.
   - The body (below the header) is what the model reads.

2. **Map the category** in `benchmark/skill_loader.py`, one line per category:

   ```python
   ORACLE_SKILL_FOR_CATEGORY: dict[str, str] = {
       "pattern_analysis": "dawn-phenomenon",
       "<category>": "<skill-name>",
   }
   ```

   Categories are the `category` values in `benchmark/tasks_v2.json`. Each category
   maps to one skill; several categories may share the same skill.

3. **Check the tools exist.** Every name in `tools` must be a tool returned by
   `_build_benchmark_tools` in `benchmark/runner.py`; otherwise the run stops with
   `Skill '<name>' declares unknown tools`. If the skill needs a new tool, add it
   there first and, if the tasks should accept it, add it to their
   `expected_tools` / `acceptable_tools` in `benchmark/tasks_v2.json`.

4. **Check the skill loads:**

   ```bash
   python -c "from benchmark.skill_loader import load_oracle_skill as f; s=f('<category>'); print(s['name'], s['tools'])"
   ```

5. **Measure it** on the category's tasks, with and without the skill, same model
   and settings:

   ```bash
   python -m benchmark --model <model> --skills oracle --task-ids <ids> --output benchmark_results_<skill>_oracle
   python -m benchmark --model <model> --skills none   --task-ids <ids> --output benchmark_results_<skill>_none
   ```

When writing a skill: keep it short and procedural; name the exact tool, arguments
and ISO date format; prefer tools that return computed results over raw data;
require both a `log_conclusion` call with the task's keys and a non-empty reply to
the user; and **use only made-up dates and values in examples**: real benchmark
values would leak the ground truth.

### Common invocations

```bash
# A subset of tasks, for a quick smoke test
python -m benchmark --model qwen3:8b --task-ids v2_540_lookup_001 v2_food_gi_001

# One patient only
python -m benchmark --model qwen3:8b --patients 540

# Local agent, remote judge and multi-turn simulator
python -m benchmark --model qwen3:8b \
    --judge-provider zen --simulator-provider zen --zen-model deepseek

# No-tools control arm: dump the whole dataset into context instead of
# giving the model query tools (tool/argument dimensions are dropped)
python -m benchmark --model qwen3:8b --no-tools --agent-timeout 900
```

Other flags: `--output`, `--tasks-path`, `--agent-max-tokens`, `--keep-db`,
`--skills {none,oracle}`.
Full list via `python -m benchmark --help`.

> `--no-tools` prefills the model's context with the patient's entire dataset
> (~134k tokens). It needs a large context window and a generous `--agent-timeout`.

### Sweep scripts

`scripts/` holds the multi-model sweeps used to produce the committed results. They
drive LM Studio's CLI to load each model in turn, so they need adapting to your
setup — read before running.

```bash
./scripts/run_all_benchmarks_v2.sh           # with-tools arm
./scripts/run_all_benchmarks_v2_notools.sh   # no-tools arm
./scripts/run_exp4_edge.sh                   # reduced context window
```

Each script runs from the repository root no matter where you invoke it.

## Analysing benchmark results

Committed results from previous runs live in `benchmark_results_v2/` (with tools),
`benchmark_results_v2_notools/` (no tools), and `benchmark_results_v2_edge/`
(reduced context), so the analysis below works before you run anything yourself.

```bash
# Per-model charts + summary table for one results directory
python -m analysis.analyze

# Cross-cutting comparison: model size, quantization, tools vs no-tools
python -m analysis.compare

# Regenerate every figure (assets/) and LaTeX table (tables/)
python -m analysis.paper_figures
```

`analysis.paper_figures` writes two variants: `assets/` + `tables/` under standard
scoring, and `assets_strict/` + `tables_strict/` where failed runs score 0 instead
of being excluded.

Also in `analysis/`:

- `collect_hw_metrics.py` — TTFT and prefill/decode throughput from an Ollama host
  (set `HW_METRICS_HOST`); appends to `results/hardware_metrics.json`
- `compute_patient_ratios.py` — derives insulin-to-carb ratios from OhioT1DM
- `merge_rerun.py` — merges a partial re-run back into a model's result files

## The OhioT1DM dataset

The benchmark and the OhioT1DM analysis scripts need the **OhioT1DM dataset**,
which is **not included in this repository**.

You can request the dataset here:

<https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html>

Once you have it, extract it so the paths look like this:

```
data/OhioT1DM/
├── 2018/
│   ├── train/559-ws-training.xml
│   └── test/559-ws-testing.xml
└── 2020/
    ├── train/540-ws-training.xml
    └── test/540-ws-testing.xml
```

`data/` is git-ignored — keep it that way. The benchmark reads the four *testing*
files for patients 540, 544, 559, and 584; to use other patients, edit
`PATIENT_REGISTRY` in `benchmark/cli.py` (each entry needs the XML path, the real
coverage window, and the insulin type).

The XML loader is `agent/data/ohio_t1dm.py`, which builds the SQLite database the
tools query.

## Project layout

```
.
├── agent/                  Core agent
│   ├── graph.py              LangGraph agent loop
│   ├── tools.py              ~30 diabetes tools
│   ├── llm.py                OpenAI-compatible client factory
│   ├── reporter.py           Glucose report generation
│   ├── data/
│   │   ├── cgm_store.py      SQLite CGM time-series store
│   │   └── ohio_t1dm.py      OhioT1DM XML -> SQLite loader
│   └── kb/store.py           Vector knowledge base (Qdrant + embeddings)
├── benchmark/              Benchmark harness  (python -m benchmark)
│   ├── cli.py                Entry point, patient registry, system prompts
│   ├── runner.py             Executes tasks against the agent
│   ├── evaluator.py          Deterministic scoring + LLM judge
│   ├── isolation.py          Per-run isolated DB/KB/session state
│   ├── user_simulator.py     Simulated user for multi-turn tasks
│   ├── skill_loader.py       Category -> skill mapping, SKILL.md parser
│   ├── skills/               One folder per skill, each with a SKILL.md
│   └── tasks_v2.json         The 72-task suite
├── analysis/               Scoring analysis and figure generation
├── scripts/                Multi-model sweep scripts
├── web/                    React front-end (Bun)
├── docs/                   Benchmark design notes and task-authoring guide
├── tests/                  Test suite
├── config.py               Central config — all values env-overridable
├── main.py                 Interactive CLI
├── server.py               HTTP API
└── run_scheduler.py        Scheduled report generation
```

Deeper documentation lives in [`docs/BENCHMARK.md`](docs/BENCHMARK.md) (design and
scoring) and [`docs/BENCHMARK_TASK_GUIDE.md`](docs/BENCHMARK_TASK_GUIDE.md) (how to
author new tasks).

## Testing

```bash
source venv/bin/activate
python -m pytest              # 118 tests
```

Tests mock the LLM endpoint, LibreLinkUp, and the scheduler, so they need no
network access and no running model server.

The exception is the benchmark-isolation tests, which read an OhioT1DM XML file —
they fail without the dataset in place.

## Troubleshooting

**`Connection refused` / the agent hangs on every turn**
The LLM endpoint isn't reachable. Confirm the server is running and that
`LM_STUDIO_BASE_URL` ends at `/v1`. Quick check:
`curl $LM_STUDIO_BASE_URL/models`

**The model never calls a tool, and every benchmark task scores 0**
The model either lacks tool-calling support or the server has it disabled. Pick a
tool-calling model; on vLLM pass `--enable-auto-tool-choice`.

**`model not found`**
`MODEL_NAME` must match an id the server serves — not the file name. List them with
`curl $LM_STUDIO_BASE_URL/models`.

**Benchmark fails with a missing-XML error**
The OhioT1DM dataset isn't in `data/`. See [above](#the-ohiot1dm-dataset).

**LibreLinkUp tools say credentials aren't set**
Expected unless you set `LIBRE_EMAIL` / `LIBRE_PASSWORD`. Nothing else is affected.

**Web UI loads but every message errors**
`server.py` isn't running, or CORS is blocking it. Start the API and pass your
front-end origin: `python server.py --cors-origin http://localhost:3001`.

**Timeouts on `--no-tools` runs**
Prefilling ~134k tokens is slow. Raise `--agent-timeout` and make sure the model is
loaded with a large enough context window.

## Disclaimer

**This is research software, not a medical device.** It is not validated for
clinical use and must not be used to make treatment decisions. Insulin dosing
output in particular is illustrative only. Always consult a qualified healthcare
professional. The authors accept no liability for any use of this software.

## License

Released under the [MIT License](LICENSE).

The OhioT1DM dataset is **not** covered by this license and is not distributed
here;
