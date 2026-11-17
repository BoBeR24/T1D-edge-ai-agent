from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _slugify_model(model_name: str) -> str:
    """Convert a model name to a filesystem-safe slug."""
    slug = re.sub(r"[^\w\-]", "-", model_name)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug[:64]

# ---------------------------------------------------------------------------
# Trace writer
# ---------------------------------------------------------------------------


def write_traces(
    traces: list[dict[str, Any]],
    output_dir: str,
    model_name: str,
    run_tag: str = "",
) -> str:
    """Write per-task trace files as JSONL. Returns the path to the JSONL file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    trace_file = out_path / f"traces_{run_tag}.jsonl"
    with trace_file.open("w", encoding="utf-8") as f:
        for trace in traces:
            record = {
                "model": model_name,
                **trace,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return str(trace_file)


# ---------------------------------------------------------------------------
# Score writer
# ---------------------------------------------------------------------------


def write_scores(
    scores: list[dict[str, Any]],
    output_dir: str,
    run_tag: str = "",
) -> str:
    """Write per-task score cards as JSONL. Returns the path to the JSONL file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    score_file = out_path / f"scores_{run_tag}.jsonl"
    with score_file.open("w", encoding="utf-8") as f:
        for score in scores:
            f.write(json.dumps(score, ensure_ascii=False) + "\n")

    return str(score_file)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def generate_markdown_summary(
    *,
    traces: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    model_name: str,
    xml_path: str,
    db_path: str,
    judge_model: str | None,
) -> str:
    """Generate a human-readable markdown summary report."""
    generated_at = datetime.now(timezone.utc).isoformat()

    lines: list[str] = [
        "# DiabetesAgent Benchmark Results",
        "",
        f"- **Generated**: {generated_at}",
        f"- **Model**: {model_name}",
        f"- **Judge Model**: {judge_model or 'none'}",
        f"- **Dataset XML**: {xml_path}",
        f"- **Dataset DB**: {db_path}",
        "",
        "---",
        "",
        "## Overall Scores",
        "",
    ]

    # Aggregate by task
    valid_scores = [s for s in scores if "normalized_score" in s]
    if valid_scores:
        overall = sum(s["normalized_score"] for s in valid_scores) / len(valid_scores)
        lines.append(f"- **Overall normalized score**: {overall:.3f} / 1.000")
        lines.append(f"- **Tasks evaluated**: {len(valid_scores)}")
    else:
        lines.append("No valid scores available.")

    # Latency summary
    latencies = [t.get("latency_ms", 0) for t in traces if "latency_ms" in t]
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        max_lat = max(latencies)
        lines.append(f"- **Average latency**: {avg_lat:.0f} ms")
        lines.append(f"- **Max latency**: {max_lat:.0f} ms")

    # Tool usage summary
    total_tool_calls = sum(t.get("tool_call_count", 0) for t in traces)
    total_turns = sum(t.get("turn_count", 0) for t in traces)
    lines.append(f"- **Total tool calls**: {total_tool_calls}")
    lines.append(f"- **Total agent turns**: {total_turns}")

    # Token usage summary (traces produced after the token-logging patch)
    tok_traces = [t for t in traces if t.get("llm_calls")]
    if tok_traces:
        n = len(tok_traces)
        avg_in = sum(t["input_tokens_total"] for t in tok_traces) / n
        avg_out = sum(t["output_tokens_total"] for t in tok_traces) / n
        avg_calls = sum(t["llm_calls"] for t in tok_traces) / n
        avg_first = sum(t["first_call_input_tokens"] for t in tok_traces) / n
        max_in = max(t["max_input_tokens"] for t in tok_traces)
        lines.append(f"- **Avg input tokens per task**: {avg_in:.0f}")
        lines.append(f"- **Avg output tokens per task**: {avg_out:.0f}")
        lines.append(f"- **Avg LLM calls per task**: {avg_calls:.1f}")
        lines.append(
            f"- **Avg first-call input tokens** (prompt + tool schemas + question): {avg_first:.0f}"
        )
        lines.append(f"- **Max input tokens in a single call**: {max_in}")

    lines.extend(["", "---", "", "## Per-Task Results", ""])

    score_by_id = {s["task_id"]: s for s in scores}
    for trace in traces:
        tid = trace["task_id"]
        score = score_by_id.get(tid, {})

        lines.append(f"### {tid}")
        lines.append(f"**Prompt**: {trace.get('prompt', '')}")
        lines.append(f"**Latency**: {trace.get('latency_ms', 'N/A')} ms")
        lines.append(f"**Tool calls**: {trace.get('tool_call_count', 0)}")
        lines.append(f"**Agent turns**: {trace.get('turn_count', 0)}")
        if trace.get("llm_calls"):
            lines.append(
                f"**Tokens**: in {trace['input_tokens_total']} / "
                f"out {trace['output_tokens_total']} over {trace['llm_calls']} LLM call(s); "
                f"first call in {trace['first_call_input_tokens']}"
            )

        if "normalized_score" in score:
            lines.append(f"**Normalized score**: {score['normalized_score']:.3f}")
        else:
            lines.append("**Normalized score**: N/A")

        # Break down scores
        score_details = score.get("scores", {})
        if score_details:
            lines.append("")
            lines.append("| Dimension | Score | Detail |")
            lines.append("|-----------|-------|--------|")
            for dim, detail in score_details.items():
                s = detail.get("score", "N/A")
                if isinstance(s, float):
                    s = f"{s:.3f}"
                d = detail.get("detail", "")
                lines.append(
                    f"| {dim} | {s} | {d[:120]}... |"
                    if len(d) > 120
                    else f"| {dim} | {s} | {d} |"
                )

        # Final response excerpt
        response = trace.get("final_response", "")
        lines.append("")
        lines.append("**Response excerpt**:")
        excerpt = response[:400] + "..." if len(response) > 400 else response
        lines.append(f"> {excerpt}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Tool Call Summary")
    lines.append("")

    for trace in traces:
        tid = trace["task_id"]
        calls = trace.get("tool_calls", [])
        if calls:
            lines.append(f"### {tid}")
            for call in calls:
                name = call.get("name", "unknown")
                args = json.dumps(call.get("args", {}), ensure_ascii=False)
                lines.append(f"- `{name}({args})`")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------


def generate_json_summary(
    *,
    traces: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    model_name: str,
    xml_path: str,
    db_path: str,
    judge_model: str | None,
) -> dict[str, Any]:
    """Generate a machine-readable JSON summary."""
    valid_scores = [s for s in scores if "normalized_score" in s]
    overall = (
        sum(s["normalized_score"] for s in valid_scores) / len(valid_scores)
        if valid_scores
        else 0.0
    )

    latencies = [t.get("latency_ms", 0) for t in traces if "latency_ms" in t]

    per_task = []
    score_by_id = {s["task_id"]: s for s in scores}
    for trace in traces:
        tid = trace["task_id"]
        score = score_by_id.get(tid, {})
        per_task.append(
            {
                "task_id": tid,
                "prompt": trace.get("prompt", ""),
                "latency_ms": trace.get("latency_ms"),
                "tool_call_count": trace.get("tool_call_count", 0),
                "turn_count": trace.get("turn_count", 0),
                "llm_calls": trace.get("llm_calls"),
                "input_tokens_total": trace.get("input_tokens_total"),
                "output_tokens_total": trace.get("output_tokens_total"),
                "first_call_input_tokens": trace.get("first_call_input_tokens"),
                "max_input_tokens": trace.get("max_input_tokens"),
                "normalized_score": score.get("normalized_score"),
                "scores": score.get("scores", {}),
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "judge_model": judge_model,
        "xml_path": xml_path,
        "db_path": db_path,
        "overall_normalized_score": round(overall, 4),
        "task_count": len(traces),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2)
        if latencies
        else None,
        "max_latency_ms": max(latencies) if latencies else None,
        "total_tool_calls": sum(t.get("tool_call_count", 0) for t in traces),
        "total_input_tokens": sum(t.get("input_tokens_total") or 0 for t in traces),
        "total_output_tokens": sum(t.get("output_tokens_total") or 0 for t in traces),
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def write_report(
    *,
    traces: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    model_name: str,
    xml_path: str,
    db_path: str,
    judge_model: str | None,
    output_dir: str,
) -> tuple[str, str, str, str]:
    """Write all artifacts: traces JSONL, scores JSONL, markdown summary, JSON summary.
    Returns paths to all four files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_tag = f"{_slugify_model(model_name)}_{ts}"

    trace_file = write_traces(traces, output_dir, model_name, run_tag)
    score_file = write_scores(scores, output_dir, run_tag)

    md_text = generate_markdown_summary(
        traces=traces,
        scores=scores,
        model_name=model_name,
        xml_path=xml_path,
        db_path=db_path,
        judge_model=judge_model,
    )
    md_file = out_path / f"summary_{run_tag}.md"
    md_file.write_text(md_text, encoding="utf-8")

    json_summary = generate_json_summary(
        traces=traces,
        scores=scores,
        model_name=model_name,
        xml_path=xml_path,
        db_path=db_path,
        judge_model=judge_model,
    )
    json_file = out_path / f"summary_{run_tag}.json"
    json_file.write_text(
        json.dumps(json_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return str(trace_file), str(score_file), str(md_file), str(json_file)
