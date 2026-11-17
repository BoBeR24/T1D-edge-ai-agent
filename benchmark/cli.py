#!/usr/bin/env python3
"""
DiabetesAgent Multi-Patient Benchmark CLI (v2)

Runs the concrete, multi-patient OhioT1DM task suite (``benchmark/tasks_v2.json``).

Unlike ``benchmark.py`` (single patient 540, single isolated DB), this runner groups
tasks by ``patient_id`` and builds a fresh isolated environment per patient — each with
that patient's real OhioT1DM data and a patient-specific system prompt (correct id,
coverage dates, and insulin type). Tasks without a ``patient_id`` (e.g. the cached
food/glycemic-index tasks, which don't touch CGM data) run under a default patient DB.

Every task in the v2 suite carries verified ground-truth values derived directly from
the patient's data, so scoring is grounded in real numbers rather than a single
hand-tuned trajectory.

Examples:
    python -m benchmark --model qwen3.5-8k:latest
    python -m benchmark --task-ids v2_540_lookup_001 v2_food_gi_001
    python -m benchmark --patients 540 584 --judge-model gpt-4.1
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark import runner as benchmark_runner
from benchmark.evaluator import evaluate_all
from benchmark.isolation import BenchmarkIsolation
from benchmark.reporter import write_report
from benchmark.runner import load_tasks, run_benchmark, select_tasks
from benchmark.spoonacular_cache import CACHED_SPOONACULAR_TOOLS
from config import MODEL_NAME, ZEN_DEFAULT_MODEL, ZEN_MODELS

# Ships with the package, so the default works regardless of the caller's cwd.
DEFAULT_TASKS_PATH = str(Path(__file__).resolve().parent / "tasks_v2.json")

# Tools the no-tools agent does NOT have (it keeps only log_conclusion plus the
# in-context data dump). A task requiring any of these — KB write/recall tools or
# the Spoonacular food-lookup tools — cannot be completed in no-tools mode.
NO_TOOLS_UNAVAILABLE_TOOLS = {
    "log_meal",
    "log_insulin",
    "log_glucose",
    "search_logged_events",
    "get_icr_summary",
} | {t.name for t in CACHED_SPOONACULAR_TOOLS}

# Patient -> (XML source, human-readable coverage window, insulin type).
# Coverage windows are taken from the real OhioT1DM timestamps (verified against the
# built DB), so the system prompt never tells the agent the wrong dates.
PATIENT_REGISTRY: dict[str, dict[str, str]] = {
    "540": {
        "xml_path": "data/OhioT1DM/2020/test/540-ws-testing.xml",
        "coverage": "July 4-14, 2027",
        "insulin_type": "Humalog",
    },
    "544": {
        "xml_path": "data/OhioT1DM/2020/test/544-ws-testing.xml",
        "coverage": "June 24 - July 4, 2027",
        "insulin_type": "Humalog",
    },
    "584": {
        "xml_path": "data/OhioT1DM/2020/test/584-ws-testing.xml",
        "coverage": "June 29 - July 9, 2025",
        "insulin_type": "Novalog",
    },
    # 559 is included specifically as a positive dawn-phenomenon case (~3 AM 8.0 ->
    # ~6 AM 11.6 mmol/L); 540/544/584 are dawn-negative, so the category has both.
    "559": {
        "xml_path": "data/OhioT1DM/2018/test/559-ws-testing.xml",
        "coverage": "January 18-27, 2022",
        "insulin_type": "Novalog",
    },
}

# Patient whose DB backs patient-independent tasks (food/GI). These tasks only use the
# cached Spoonacular tools, so the CGM data behind them is irrelevant.
DEFAULT_PATIENT = "540"


def _system_prompt_for(patient_id: str, no_tools: bool = False) -> str:
    info = PATIENT_REGISTRY[patient_id]
    if no_tools:
        # No-tools mode: the full dataset is appended to this prompt (see
        # benchmark_runner.serialize_patient_data), so the agent answers from
        # context rather than querying tools. log_conclusion is its only tool.
        return (
            f"You are running a controlled diabetes-data benchmark on OhioT1DM patient "
            f"{patient_id}. Dataset coverage is approximately {info['coverage']} (UTC). "
            "The patient's COMPLETE dataset (CGM, meals, bolus, basal) is provided "
            "below in the system prompt. You do NOT have data-query tools — read the "
            "provided data directly to answer. Do not assume real-time/live sensor "
            "data exists. If a requested date is outside the provided data, state "
            "clearly that there is no data for that period — do not fabricate values. "
            "When relevant, cross-reference glucose, meals, bolus, and basal data "
            "before concluding. Be concise but accurate. "
            "IMPORTANT: When you have finished your analysis, you MUST call "
            "log_conclusion with a JSON 'findings' field containing your key "
            "structured conclusions. Example: findings='{\"glucose_mmol_l\": 5.0, "
            "\"timestamp\": \"2027-07-04T09:01:45+00:00\"}'."
        )
    return (
        f"You are running a controlled diabetes-data benchmark on OhioT1DM patient "
        f"{patient_id}. Dataset coverage is approximately {info['coverage']} (UTC). "
        "Use tools with explicit start/end ISO date ranges whenever possible. "
        "Do not assume real-time/live sensor data exists. If a requested date is "
        "outside the dataset, query it, then state clearly that there is no data for "
        "that period — do not fabricate values. "
        "When relevant, cross-reference glucose, meals, bolus, and basal data before "
        "concluding. Be concise but accurate. "
        "IMPORTANT: When you have finished your analysis, you MUST call log_conclusion "
        "with a JSON 'findings' field containing your key structured conclusions. "
        "Example: findings='{\"glucose_mmol_l\": 5.0, "
        "\"timestamp\": \"2027-07-04T09:01:45+00:00\"}'."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the multi-patient OhioT1DM benchmark suite (v2)"
    )
    parser.add_argument("--model", default=None, help="Model name to evaluate")
    parser.add_argument(
        "--tasks-path", default=DEFAULT_TASKS_PATH, help="Path to v2 tasks JSON"
    )
    parser.add_argument(
        "--task-ids", nargs="+", default=None, help="Specific task IDs to run"
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help=f"Patient ids to run (default: all known {sorted(PATIENT_REGISTRY)})",
    )
    parser.add_argument(
        "--output", default="benchmark_results_v2", help="Output directory"
    )
    parser.add_argument(
        "--judge-model", default=None, help="Optional judge model for prose scoring"
    )
    parser.add_argument(
        "--simulator-model",
        default=None,
        help="Optional model override for the LLM/evaluator chat simulator "
        "(otherwise the model named in each task's user_simulator is used)",
    )
    # Per-role endpoint selection. "local" -> LM Studio (default), "zen" -> the
    # remote OpenCode Zen endpoint configured in config.py. Each role can be routed
    # independently, e.g. a local agent judged by a remote zen model.
    provider_choices = ["local", "zen"]
    parser.add_argument(
        "--agent-provider",
        choices=provider_choices,
        default="local",
        help="Endpoint for the agent under test (default: local)",
    )
    parser.add_argument(
        "--judge-provider",
        choices=provider_choices,
        default="local",
        help="Endpoint for the LLM judge (default: local)",
    )
    parser.add_argument(
        "--simulator-provider",
        choices=provider_choices,
        default="local",
        help="Endpoint for the chat/user simulator (default: local)",
    )
    parser.add_argument(
        "--zen-model",
        choices=sorted(ZEN_MODELS),
        default=ZEN_DEFAULT_MODEL,
        help=(
            "Which zen model any zen-routed role uses "
            f"({', '.join(f'{k}={v}' for k, v in ZEN_MODELS.items())}). "
            "A per-role --model/--judge-model/--simulator-model still overrides this."
        ),
    )
    parser.add_argument("--keep-db", action="store_true", help="Keep temp DBs")
    parser.add_argument(
        "--agent-timeout",
        type=int,
        default=None,
        help="Agent LLM request timeout in seconds (default: benchmark_runner."
        "BENCHMARK_AGENT_TIMEOUT). Raise for no-tools runs where the full-dataset "
        "prefill is slow.",
    )
    parser.add_argument(
        "--agent-max-tokens",
        type=int,
        default=None,
        help="Cap on total completion tokens (reasoning + answer) per agent step "
        "(default: benchmark_runner.BENCHMARK_AGENT_MAX_TOKENS). Bounds runaway "
        "reasoning; a task that hits the cap is truncated and scored as-is.",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Run without data-query tools: dump the patient's full dataset into the "
        "context window and give the agent only log_conclusion. For with-tools vs "
        "without-tools comparisons. (Tool/argument scoring dimensions are dropped.)",
    )
    parser.add_argument(
        "--skills",
        choices=["none", "oracle"],
        default="none",
        help="Skill mode. 'oracle': for categories mapped in benchmark/skill_loader.py, "
        "append the matching SKILL.md to the system prompt and expose only that "
        "skill's tools (+ log_conclusion). 'none' (default) = baseline.",
    )
    return parser


def _task_required_tools(task: dict[str, Any]) -> set[str]:
    """All tools a task expects/accepts, including those named in multi-turn scripts."""
    tools = set(task.get("expected_tools", [])) | set(task.get("acceptable_tools", []))
    for step in task.get("user_simulator", {}).get("script", []):
        tools |= set(step.get("expected_tools", [])) | set(
            step.get("acceptable_tools", [])
        )
    return tools


def _filter_no_tools_doable(
    tasks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep only tasks a no-tools agent can actually complete.

    Doable means: the task is about a patient whose data is dumped into context, and
    it needs no tools beyond log_conclusion + the (now in-context) read-only data.
    Tasks that require KB write/recall tools (logging, memory) or the Spoonacular
    food-lookup tools (food_glycemic, which also have no patient data to inject) are
    dropped — the agent has no way to do them without tools.
    """
    doable: list[dict[str, Any]] = []
    skipped: list[str] = []
    for task in tasks:
        required = _task_required_tools(task)
        if not task.get("patient_id") or (required & NO_TOOLS_UNAVAILABLE_TOOLS):
            skipped.append(task["task_id"])
        else:
            doable.append(task)
    return doable, skipped


def _group_tasks_by_patient(
    tasks: list[dict[str, Any]], patients_filter: set[str] | None
) -> dict[str, list[dict[str, Any]]]:
    """Group tasks by patient_id. Tasks with no patient_id go to DEFAULT_PATIENT."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        pid = task.get("patient_id") or DEFAULT_PATIENT
        if pid not in PATIENT_REGISTRY:
            print(
                f"[benchmark-v2] WARNING: task {task['task_id']} references unknown "
                f"patient '{pid}'; skipping."
            )
            continue
        if patients_filter and pid not in patients_filter:
            continue
        grouped[pid].append(task)
    return grouped


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = str(args.output)

    # Per-role model: an explicit per-role flag always wins; otherwise a zen-routed
    # role uses the chosen --zen-model and a local role uses the default MODEL_NAME.
    def _role_model(explicit: str | None, provider: str) -> str | None:
        if explicit:
            return explicit
        return args.zen_model if provider == "zen" else None

    model_name = args.model or (
        args.zen_model if args.agent_provider == "zen" else MODEL_NAME
    )
    judge_model = _role_model(args.judge_model, args.judge_provider)
    simulator_model = _role_model(args.simulator_model, args.simulator_provider)

    # Surface the per-role endpoint routing and fail fast if zen is requested
    # without credentials configured.
    roles = {
        "agent": args.agent_provider,
        "judge": args.judge_provider,
        "simulator": args.simulator_provider,
    }
    zen_roles = [r for r, p in roles.items() if p == "zen"]
    if zen_roles:
        from config import OPENCODE_ZEN_API_KEY

        print(
            f"[benchmark-v2] OpenCode Zen endpoint ({ZEN_MODELS[args.zen_model]}) "
            f"for: {', '.join(sorted(zen_roles))}"
        )
        if not OPENCODE_ZEN_API_KEY:
            print(
                "[benchmark-v2] ERROR: 'zen' provider selected but no API key. "
                "Set OPENCODE_ZEN_API_KEY (env var) or OPENCODE_ZEN_API_KEY in config.py."
            )
            return 1

    print(f"[benchmark-v2] Loading tasks from {args.tasks_path}...")
    try:
        all_tasks = load_tasks(args.tasks_path)
    except FileNotFoundError:
        print(f"[benchmark-v2] ERROR: Tasks file not found: {args.tasks_path}")
        return 1

    tasks = select_tasks(all_tasks, args.task_ids)

    # In no-tools mode, drop tasks that need tools the agent no longer has (logging,
    # memory, food-lookup) so it only runs tasks it can actually complete.
    if args.no_tools:
        tasks, skipped = _filter_no_tools_doable(tasks)
        if skipped:
            print(
                f"[benchmark-v2] NO-TOOLS: skipping {len(skipped)} task(s) needing "
                f"unavailable tools (logging/memory/food): {sorted(skipped)}"
            )
        if not tasks:
            print("[benchmark-v2] No no-tools-doable tasks in the selection.")
            return 0

    patients_filter = set(args.patients) if args.patients else None
    grouped = _group_tasks_by_patient(tasks, patients_filter)

    if not grouped:
        print("[benchmark-v2] No tasks matched the selection.")
        return 0

    total = sum(len(v) for v in grouped.values())
    print(
        f"[benchmark-v2] {total} task(s) across {len(grouped)} patient(s): "
        + ", ".join(f"{p}({len(t)})" for p, t in sorted(grouped.items()))
    )

    if args.no_tools:
        print(
            "[benchmark-v2] NO-TOOLS mode: full patient dataset injected into context; "
            "only log_conclusion is available. Tool/argument scoring dimensions are "
            "dropped so scores reflect answer quality."
        )

    # Preserve the module-level prompt so we can restore it afterwards.
    original_prompt = benchmark_runner.BENCHMARK_SYSTEM_PROMPT

    all_traces: list[dict[str, Any]] = []
    all_scores: list[dict[str, Any]] = []
    per_patient_mean: dict[str, float] = {}

    try:
        for patient_id in sorted(grouped):
            patient_tasks = grouped[patient_id]
            info = PATIENT_REGISTRY[patient_id]
            print(
                f"\n[benchmark-v2] === Patient {patient_id} "
                f"({info['coverage']}, {info['insulin_type']}) — "
                f"{len(patient_tasks)} task(s) ==="
            )

            # Per-patient system prompt (correct id / dates / context).
            benchmark_runner.BENCHMARK_SYSTEM_PROMPT = _system_prompt_for(
                patient_id, no_tools=args.no_tools
            )

            with BenchmarkIsolation(
                xml_path=info["xml_path"], keep=args.keep_db
            ) as isolation:
                db_path = isolation.db_path
                traces = run_benchmark(
                    tasks=patient_tasks,
                    db_path=db_path,
                    model_name=model_name,
                    kb_store=isolation.kb_store,
                    provider=args.agent_provider,
                    simulator_provider=args.simulator_provider,
                    simulator_model=simulator_model,
                    no_tools=args.no_tools,
                    agent_timeout=args.agent_timeout,
                    agent_max_tokens=args.agent_max_tokens,
                    skills_mode=args.skills,
                )
                scores = evaluate_all(
                    traces=traces,
                    tasks=patient_tasks,
                    db_path=db_path,
                    judge_model=judge_model,
                    kb_store=isolation.kb_store,
                    judge_provider=args.judge_provider,
                    no_tools=args.no_tools,
                )

            # Tag scores/traces with patient for the combined report.
            for s in scores:
                s["patient_id"] = patient_id
            for t in traces:
                t["patient_id"] = patient_id

            all_traces.extend(traces)
            all_scores.extend(scores)

            valid = [s for s in scores if "normalized_score" in s]
            if valid:
                mean = sum(s["normalized_score"] for s in valid) / len(valid)
                per_patient_mean[patient_id] = mean
                print(
                    f"[benchmark-v2] Patient {patient_id} mean score: "
                    f"{mean:.3f} ({len(valid)} task(s))"
                )
    finally:
        benchmark_runner.BENCHMARK_SYSTEM_PROMPT = original_prompt

    # Combined report across all patients.
    print(f"\n[benchmark-v2] Writing combined artifacts to {output_dir}...")
    trace_file, score_file, md_file, json_file = write_report(
        traces=all_traces,
        scores=all_scores,
        model_name=model_name,
        xml_path="multi-patient (" + ",".join(sorted(grouped)) + ")",
        db_path="(per-patient isolated)",
        judge_model=judge_model,
        output_dir=output_dir,
    )
    print(f"[benchmark-v2] Wrote traces:   {trace_file}")
    print(f"[benchmark-v2] Wrote scores:   {score_file}")
    print(f"[benchmark-v2] Wrote markdown: {md_file}")
    print(f"[benchmark-v2] Wrote JSON:     {json_file}")

    valid_all = [s for s in all_scores if "normalized_score" in s]
    if valid_all:
        overall = sum(s["normalized_score"] for s in valid_all) / len(valid_all)
        print("\n[benchmark-v2] Per-patient means:")
        for pid in sorted(per_patient_mean):
            print(f"  patient {pid}: {per_patient_mean[pid]:.3f}")
        print(
            f"[benchmark-v2] Overall normalized score: {overall:.3f} / 1.000 "
            f"({len(valid_all)} task(s))"
        )

    print("[benchmark-v2] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
