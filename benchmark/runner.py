from __future__ import annotations

import importlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agent.llm import get_llm
from agent.tools import (
    get_cgm_readings,
    get_icr_summary,
    get_nearest_cgm_reading,
    log_glucose,
    log_insulin,
    log_meal,
    search_logged_events,
)
from benchmark.spoonacular_cache import CACHED_SPOONACULAR_TOOLS
from benchmark.skill_loader import load_oracle_skill
from benchmark.user_simulator import (
    ContinueAction,
    EvaluatorSimulator,
    RejectAction,
    build_simulator,
)

# Agent LLM timeout (seconds). Generous because no-tools mode injects the whole
# dataset, so prefill of a large prompt can take many minutes on local models.
BENCHMARK_AGENT_TIMEOUT = 1800

# Cap on total completion tokens (reasoning + answer) per agent step. Reasoning
# models (e.g. Gemma 4, Qwen3) can otherwise run away — transcribing the whole
# no-tools dataset in their chain-of-thought for tens of thousands of tokens. When
# the cap is hit the response is truncated (finish_reason == "length"); the agent
# node logs it and the task is scored on whatever (if anything) it managed to log.
BENCHMARK_AGENT_MAX_TOKENS = 30000

BENCHMARK_SYSTEM_PROMPT = (
    "You are running a controlled diabetes-data benchmark on OhioT1DM patient 540. "
    "Dataset coverage is approximately July 4-14, 2027 (UTC). "
    "Use tools with explicit start/end ISO date ranges whenever possible. "
    "Do not assume real-time/live sensor data exists. "
    "When relevant, cross-reference glucose, meals, bolus, and basal data before concluding. "
    "Be concise but accurate. "
    "IMPORTANT: When you have finished your analysis, you MUST call log_conclusion "
    "with a JSON 'findings' field containing your key structured conclusions. "
    'Example: findings=\'{"glucose_mmol_l": 5.0, "timestamp": "2027-07-04T09:01:45+00:00"}\'.'
)


# ---------------------------------------------------------------------------
# DB helpers (same pattern as experiment.py)
# ---------------------------------------------------------------------------


class BenchmarkDB:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def query_rows(
        self, sql: str, params: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def query_one(
        self, sql: str, params: tuple[Any, ...] = ()
    ) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        conn = self._connect()
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()


def _coerce_iso(value: str) -> str:
    text = (value or "").strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Full-context (no-tools) patient data dump
# ---------------------------------------------------------------------------


def serialize_patient_data(db_path: str) -> str:
    """Serialize the entire patient dataset to a compact text block.

    Used by the "no tools" benchmark mode: instead of giving the agent query
    tools, the same data the tools would expose is dumped straight into the
    context window. Format is compact CSV-style to keep the token count down,
    and the rows are exactly the columns the DB-backed tools return so the two
    modes differ only in *access method*, not in the data itself.
    """
    db = BenchmarkDB(db_path)

    sections: list[str] = [
        "PATIENT DATASET (provided in full because no query tools are available). "
        "All timestamps are UTC ISO-8601. Use only the values below; do not "
        "fabricate readings outside this data.",
    ]

    meta = db.query_one(
        "SELECT patient_id, weight, insulin_type FROM patient_metadata LIMIT 1"
    )
    if meta:
        sections.append(
            "== Patient metadata ==\n"
            f"patient_id={meta.get('patient_id')} "
            f"weight={meta.get('weight')} "
            f"insulin_type={meta.get('insulin_type')}"
        )

    def _block(title: str, header: str, rows: list[dict[str, Any]], cols: list[str]) -> str:
        lines = [f"== {title} ({len(rows)} rows) ==", header]
        for r in rows:
            lines.append(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols))
        return "\n".join(lines)

    cgm = db.query_rows(
        "SELECT factory_timestamp_utc, glucose_mmol_l FROM cgm_readings "
        "ORDER BY factory_timestamp_utc ASC"
    )
    sections.append(
        _block(
            "CGM readings",
            "factory_timestamp_utc,glucose_mmol_l",
            cgm,
            ["factory_timestamp_utc", "glucose_mmol_l"],
        )
    )

    meals = db.query_rows(
        "SELECT timestamp_utc, meal_type, carbs_g FROM meal ORDER BY timestamp_utc ASC"
    )
    sections.append(
        _block(
            "Meals",
            "timestamp_utc,meal_type,carbs_g",
            meals,
            ["timestamp_utc", "meal_type", "carbs_g"],
        )
    )

    bolus = db.query_rows(
        "SELECT ts_begin_utc, ts_end_utc, bolus_type, dose_units FROM bolus "
        "ORDER BY ts_begin_utc ASC"
    )
    sections.append(
        _block(
            "Bolus doses",
            "ts_begin_utc,ts_end_utc,bolus_type,dose_units",
            bolus,
            ["ts_begin_utc", "ts_end_utc", "bolus_type", "dose_units"],
        )
    )

    basal = db.query_rows(
        "SELECT timestamp_utc, rate_units_per_hr FROM basal ORDER BY timestamp_utc ASC"
    )
    sections.append(
        _block(
            "Scheduled basal rates",
            "timestamp_utc,rate_units_per_hr",
            basal,
            ["timestamp_utc", "rate_units_per_hr"],
        )
    )

    temp_basal = db.query_rows(
        "SELECT ts_begin_utc, ts_end_utc, rate_units_per_hr FROM temp_basal "
        "ORDER BY ts_begin_utc ASC"
    )
    sections.append(
        _block(
            "Temp basal rates",
            "ts_begin_utc,ts_end_utc,rate_units_per_hr",
            temp_basal,
            ["ts_begin_utc", "ts_end_utc", "rate_units_per_hr"],
        )
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Benchmark-safe tools
# ---------------------------------------------------------------------------


def _build_benchmark_tools(db_path: str, no_tools: bool = False) -> list[Any]:
    db = BenchmarkDB(db_path)

    @tool
    def get_meals(start_iso: str, end_iso: str) -> str:
        """Query meal events between UTC ISO timestamps (inclusive)."""
        try:
            start = _coerce_iso(start_iso)
            end = _coerce_iso(end_iso)
        except Exception as exc:
            return f"Invalid timestamp input: {exc}"

        rows = db.query_rows(
            """
            SELECT timestamp_utc, meal_type, carbs_g
            FROM meal
            WHERE timestamp_utc >= ? AND timestamp_utc <= ?
            ORDER BY timestamp_utc ASC
            """,
            (start, end),
        )
        if not rows:
            return f"No meals found from {start} to {end}."
        return json.dumps(rows)

    @tool
    def get_bolus_doses(start_iso: str, end_iso: str) -> str:
        """Query bolus insulin doses overlapping a UTC ISO time range."""
        try:
            start = _coerce_iso(start_iso)
            end = _coerce_iso(end_iso)
        except Exception as exc:
            return f"Invalid timestamp input: {exc}"

        rows = db.query_rows(
            """
            SELECT ts_begin_utc, ts_end_utc, bolus_type, dose_units
            FROM bolus
            WHERE ts_end_utc >= ? AND ts_begin_utc <= ?
            ORDER BY ts_begin_utc ASC
            """,
            (start, end),
        )
        if not rows:
            return f"No bolus doses found from {start} to {end}."
        return json.dumps(rows)

    @tool
    def get_basal_rates(start_iso: str, end_iso: str) -> str:
        """Query scheduled basal and temp basal rates overlapping a UTC ISO time range."""
        try:
            start = _coerce_iso(start_iso)
            end = _coerce_iso(end_iso)
        except Exception as exc:
            return f"Invalid timestamp input: {exc}"

        rows = db.query_rows(
            """
            SELECT 'basal' AS source,
                   timestamp_utc AS ts_begin_utc,
                   timestamp_utc AS ts_end_utc,
                   rate_units_per_hr
            FROM basal
            WHERE timestamp_utc >= ? AND timestamp_utc <= ?

            UNION ALL

            SELECT 'temp_basal' AS source,
                   ts_begin_utc,
                   ts_end_utc,
                   rate_units_per_hr
            FROM temp_basal
            WHERE ts_end_utc >= ? AND ts_begin_utc <= ?

            ORDER BY ts_begin_utc ASC
            """,
            (start, end, start, end),
        )
        if not rows:
            return f"No basal data found from {start} to {end}."
        return json.dumps(rows)

    @tool
    def get_patient_info() -> str:
        """Return patient metadata (id, weight, insulin_type)."""
        row = db.query_one(
            """
            SELECT patient_id, weight, insulin_type
            FROM patient_metadata
            LIMIT 1
            """
        )
        if row is None:
            return "No patient metadata found."
        return json.dumps(row)

    @tool
    def get_cgm_summary_from_db_range(
        start_iso: str, end_iso: str, low: float = 3.9, high: float = 10.0
    ) -> str:
        """Return glucose summary stats from SQLite for an explicit UTC range."""
        try:
            start = _coerce_iso(start_iso)
            end = _coerce_iso(end_iso)
        except Exception as exc:
            return f"Invalid timestamp input: {exc}"

        rows = db.query_rows(
            """
            SELECT glucose_mmol_l
            FROM cgm_readings
            WHERE factory_timestamp_utc >= ? AND factory_timestamp_utc <= ?
            ORDER BY factory_timestamp_utc ASC
            """,
            (start, end),
        )
        if not rows:
            return f"No CGM readings found from {start} to {end}."

        values = [float(r["glucose_mmol_l"]) for r in rows]
        avg = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        in_range_count = sum(1 for v in values if low <= v <= high)
        tir = (in_range_count / len(values)) * 100
        avg_mg_dl = avg * 18.0
        est_a1c = (avg_mg_dl + 46.7) / 28.7
        # Population SD to match the benchmark reference queries.
        std_dev = (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5
        cv_percent = (std_dev / avg) * 100 if avg else 0.0

        return (
            f"SQLite CGM summary ({start} to {end}, {len(values)} readings): "
            f"avg {avg:.1f} mmol/L, min {min_val:.1f}, max {max_val:.1f}, "
            f"std dev {std_dev:.1f} mmol/L, CV {cv_percent:.1f}%, "
            f"time in range {low:.1f}–{high:.1f} mmol/L: {tir:.1f}%, "
            f"estimated A1C: {est_a1c:.2f}%."
        )

    @tool
    def get_cgm_spikes_from_db_range(
        start_iso: str,
        end_iso: str,
        min_rise_mmol_l: float = 2.2,
        min_drop_mmol_l: float = 2.2,
        window_minutes: int = 120,
    ) -> str:
        """Detect notable glucose spikes and dips from SQLite for an explicit UTC range."""
        try:
            start = _coerce_iso(start_iso)
            end = _coerce_iso(end_iso)
        except Exception as exc:
            return f"Invalid timestamp input: {exc}"

        rows = db.query_rows(
            """
            SELECT factory_timestamp_utc, glucose_mmol_l
            FROM cgm_readings
            WHERE factory_timestamp_utc >= ? AND factory_timestamp_utc <= ?
            ORDER BY factory_timestamp_utc ASC
            LIMIT 5000
            """,
            (start, end),
        )
        if len(rows) < 2:
            return (
                f"Not enough CGM readings from {start} to {end} to detect excursions."
            )

        spikes: list[tuple[str, str, float, float, float]] = []
        dips: list[tuple[str, str, float, float, float]] = []

        for i, start_row in enumerate(rows):
            start_ts = str(start_row["factory_timestamp_utc"])
            start_val = float(start_row["glucose_mmol_l"])
            found_spike = False
            found_dip = False

            for end_row in rows[i + 1 :]:
                end_ts = str(end_row["factory_timestamp_utc"])
                end_val = float(end_row["glucose_mmol_l"])
                delta = end_val - start_val

                if (not found_spike) and delta >= min_rise_mmol_l:
                    spikes.append((start_ts, end_ts, start_val, end_val, delta))
                    found_spike = True

                if (not found_dip) and delta <= -min_drop_mmol_l:
                    dips.append((start_ts, end_ts, start_val, end_val, delta))
                    found_dip = True

                if found_spike and found_dip:
                    break

        if not spikes and not dips:
            return (
                f"No spikes/dips found from {start} to {end} "
                f"with rise >= {min_rise_mmol_l:.1f} mmol/L or drop >= {min_drop_mmol_l:.1f} mmol/L."
            )

        spike_lines = [
            f"{s[0]} -> {s[1]}: {s[2]:.1f} → {s[3]:.1f} mmol/L (rise {s[4]:+.1f})"
            for s in spikes[:10]
        ]
        dip_lines = [
            f"{d[0]} -> {d[1]}: {d[2]:.1f} → {d[3]:.1f} mmol/L (drop {d[4]:+.1f})"
            for d in dips[:10]
        ]

        blocks = [
            (
                f"Spikes: {len(spikes)} detected "
                f"(threshold +{min_rise_mmol_l:.1f} mmol/L)."
                + (
                    "\n" + "\n".join(f"- {line}" for line in spike_lines)
                    if spike_lines
                    else ""
                )
            ),
            (
                f"Dips: {len(dips)} detected "
                f"(threshold -{min_drop_mmol_l:.1f} mmol/L)."
                + (
                    "\n" + "\n".join(f"- {line}" for line in dip_lines)
                    if dip_lines
                    else ""
                )
            ),
        ]

        return f"SQLite CGM excursions ({start} to {end}):\n" + "\n\n".join(blocks)

    def _range_values(start_iso: str, end_iso: str):
        """Fetch (timestamp, glucose_mmol_l) tuples for an inclusive UTC range.
        Returns (rows, error_message). On success error_message is ''."""
        try:
            start = _coerce_iso(start_iso)
            end = _coerce_iso(end_iso)
        except Exception as exc:
            return None, f"Invalid timestamp input: {exc}"
        rows = db.query_rows(
            """
            SELECT factory_timestamp_utc, glucose_mmol_l
            FROM cgm_readings
            WHERE factory_timestamp_utc >= ? AND factory_timestamp_utc <= ?
            ORDER BY factory_timestamp_utc ASC
            """,
            (start, end),
        )
        return rows, ""

    @tool
    def get_time_in_range_from_db_range(
        start_iso: str, end_iso: str, low: float = 3.9, high: float = 10.0
    ) -> str:
        """Percentage of readings in the target range (low–high mmol/L) for an explicit UTC range."""
        rows, err = _range_values(start_iso, end_iso)
        if err:
            return err
        if not rows:
            return f"No CGM readings found from {start_iso} to {end_iso}."
        values = [float(r["glucose_mmol_l"]) for r in rows]
        in_range = sum(1 for v in values if low <= v <= high)
        tir = 100.0 * in_range / len(values)
        return (
            f"Time in range {low:.1f}–{high:.1f} mmol/L ({start_iso} to {end_iso}, "
            f"{len(values)} readings): {tir:.1f}% ({in_range} readings in range)."
        )

    @tool
    def get_average_glucose_from_db_range(start_iso: str, end_iso: str) -> str:
        """Average (and min/max) glucose in mmol/L for an explicit UTC range."""
        rows, err = _range_values(start_iso, end_iso)
        if err:
            return err
        if not rows:
            return f"No CGM readings found from {start_iso} to {end_iso}."
        values = [float(r["glucose_mmol_l"]) for r in rows]
        avg = sum(values) / len(values)
        return (
            f"Average glucose ({start_iso} to {end_iso}, {len(values)} readings): "
            f"{avg:.1f} mmol/L (min {min(values):.1f}, max {max(values):.1f})."
        )

    @tool
    def get_estimated_a1c_from_db_range(start_iso: str, end_iso: str) -> str:
        """Estimated A1C (ADAG formula) and mean glucose for an explicit UTC range."""
        rows, err = _range_values(start_iso, end_iso)
        if err:
            return err
        if not rows:
            return f"No CGM readings found from {start_iso} to {end_iso}."
        values = [float(r["glucose_mmol_l"]) for r in rows]
        avg = sum(values) / len(values)
        est_a1c = (avg * 18.0 + 46.7) / 28.7
        return (
            f"Estimated A1C ({start_iso} to {end_iso}, {len(values)} readings): "
            f"{est_a1c:.2f}% (mean glucose {avg:.1f} mmol/L)."
        )

    @tool
    def get_glucose_summary_from_db_range(
        start_iso: str, end_iso: str, low: float = 3.9, high: float = 10.0
    ) -> str:
        """Compact summary (avg, min, max, TIR, estimated A1C) for an explicit UTC range."""
        rows, err = _range_values(start_iso, end_iso)
        if err:
            return err
        if not rows:
            return f"No CGM readings found from {start_iso} to {end_iso}."
        values = [float(r["glucose_mmol_l"]) for r in rows]
        avg = sum(values) / len(values)
        in_range = sum(1 for v in values if low <= v <= high)
        tir = 100.0 * in_range / len(values)
        est_a1c = (avg * 18.0 + 46.7) / 28.7
        return (
            f"Glucose summary ({start_iso} to {end_iso}, {len(values)} readings): "
            f"avg {avg:.1f} mmol/L, min {min(values):.1f}, max {max(values):.1f}, "
            f"time in range {low:.1f}–{high:.1f} mmol/L: {tir:.1f}%, "
            f"estimated A1C: {est_a1c:.2f}%."
        )

    @tool
    def get_cgm_analytics_from_db_range(
        start_iso: str,
        end_iso: str,
        low: float = 3.9,
        high: float = 10.0,
        hourly: bool = False,
    ) -> str:
        """Server-side glycemic analytics for an explicit UTC range — computed in code so
        the model does not have to tally raw readings.

        Returns, in one compact block:
          - reading_count, mean, min, max, std dev, CV%
          - estimated A1C (ADAG), time in range (TIR%)
          - time above range: count and TAR% (> high)
          - hypoglycemia: level-1 count (< low) and level-2 count (< 3.0 mmol/L)
          - (when hourly=True) average glucose per hour-of-day (00–23)

        Use this for time-in-range, time-above-range, hypoglycemia-severity, variability,
        and dawn-phenomenon (hourly=True) questions instead of pulling raw readings.
        """
        rows, err = _range_values(start_iso, end_iso)
        if err:
            return err
        if not rows:
            return f"No CGM readings found from {start_iso} to {end_iso}."

        values = [float(r["glucose_mmol_l"]) for r in rows]
        n = len(values)
        avg = sum(values) / n
        std_dev = (sum((v - avg) ** 2 for v in values) / n) ** 0.5
        cv_percent = (std_dev / avg) * 100 if avg else 0.0
        est_a1c = (avg * 18.0 + 46.7) / 28.7
        in_range = sum(1 for v in values if low <= v <= high)
        tir = 100.0 * in_range / n
        above = sum(1 for v in values if v > high)
        tar = 100.0 * above / n
        level1 = sum(1 for v in values if v < low)
        level2 = sum(1 for v in values if v < 3.0)

        out = (
            f"CGM analytics ({start_iso} to {end_iso}, {n} readings): "
            f"mean {avg:.1f} mmol/L, min {min(values):.1f}, max {max(values):.1f}, "
            f"std dev {std_dev:.1f}, CV {cv_percent:.1f}%. "
            f"TIR {low:.1f}–{high:.1f}: {tir:.1f}% | TAR >{high:.1f}: {tar:.1f}% "
            f"({above} readings). "
            f"Hypo: level-1 <{low:.1f}: {level1} readings | level-2 <3.0: {level2} readings. "
            f"estimated A1C: {est_a1c:.2f}%."
        )

        if hourly:
            buckets: dict[int, list[float]] = {}
            for r in rows:
                ts = str(r["factory_timestamp_utc"])
                try:
                    hour = int(ts[11:13])
                except (ValueError, IndexError):
                    continue
                buckets.setdefault(hour, []).append(float(r["glucose_mmol_l"]))
            hourly_parts = [
                f"{h:02d}h:{sum(buckets[h]) / len(buckets[h]):.1f}"
                for h in range(24)
                if h in buckets
            ]
            out += "\n[hourly avg] " + " ".join(hourly_parts)

        return out

    @tool
    def log_conclusion(task_id: str, findings: str, notes: str = "") -> str:
        """Log the final structured findings for this benchmark task.

        findings must be a JSON string with key-value pairs representing
        your structured conclusions. Example:
        '{"glucose_mmol_l": 5.0, "timestamp": "2027-07-04T09:01:45+00:00"}'
        """
        schema = """
        CREATE TABLE IF NOT EXISTS benchmark_conclusions (
            task_id TEXT PRIMARY KEY,
            findings_json TEXT NOT NULL,
            notes TEXT,
            logged_at TEXT NOT NULL
        );
        """
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(schema)
            conn.execute(
                """
                INSERT INTO benchmark_conclusions(task_id, findings_json, notes, logged_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    findings_json=excluded.findings_json,
                    notes=excluded.notes,
                    logged_at=excluded.logged_at
                """,
                (task_id, findings, notes, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            return f"Conclusion logged for task {task_id}: {findings}"
        except Exception as exc:
            return f"Failed to log conclusion: {exc}"
        finally:
            conn.close()

    # No-tools mode: the full dataset is dumped into the context window instead,
    # so the only tool the agent keeps is log_conclusion (its answer channel).
    if no_tools:
        return [log_conclusion]

    return [
        get_cgm_readings,
        get_nearest_cgm_reading,
        get_cgm_summary_from_db_range,
        get_cgm_spikes_from_db_range,
        # ISO-range, DB-backed aggregate tools (offline twins of the live days-based
        # tools, which can't run here — they pull from the live LibreLink sensor).
        get_time_in_range_from_db_range,
        get_average_glucose_from_db_range,
        get_estimated_a1c_from_db_range,
        get_glucose_summary_from_db_range,
        # Rich single-call analytics (TAR, hypo severity, hourly) for the analysis tasks.
        get_cgm_analytics_from_db_range,
        get_meals,
        get_bolus_doses,
        get_basal_rates,
        get_patient_info,
        log_conclusion,
        # Stateful production tools — they write to the patched kb_store singleton
        log_meal,
        log_insulin,
        log_glucose,
        search_logged_events,
        get_icr_summary,
        # Cached Spoonacular food/nutrition tools (read-through JSON cache)
        *CACHED_SPOONACULAR_TOOLS,
    ]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _agent_node(
    state: MessagesState,
    llm_with_tools: Any,
    data_context: str | None = None,
    skill_text: str | None = None,
):
    messages = state["messages"]
    if not any(getattr(m, "type", None) == "system" for m in messages):
        system_text = BENCHMARK_SYSTEM_PROMPT
        # No-tools mode: append the full dataset to the system message so the
        # model has everything it needs to answer without querying tools.
        if data_context:
            system_text = f"{system_text}\n\n{data_context}"
        # Oracle skill: append the skill instructions to the system prompt.
        if skill_text:
            system_text = f"{system_text}\n\n{skill_text}"
        messages = [SystemMessage(content=system_text)] + messages

    response = llm_with_tools.invoke(messages)

    # If the model hit the token cap (runaway reasoning / overlong answer), the
    # response is truncated. Surface it; the graph still proceeds — a truncated
    # turn usually has no (or an incomplete) tool call, so it routes to END and the
    # task is scored on whatever it managed to log (often nothing -> low score).
    finish_reason = (getattr(response, "response_metadata", {}) or {}).get(
        "finish_reason"
    )
    if finish_reason == "length":
        print(
            "[benchmark] WARNING: agent response hit the max_tokens cap (truncated); "
            "continuing with the partial output."
        )

    return {"messages": [response]}


def _should_continue(state: MessagesState):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_benchmark_graph(
    db_path: str,
    model_name: str | None = None,
    provider: str | None = None,
    no_tools: bool = False,
    agent_timeout: int | None = None,
    agent_max_tokens: int | None = None,
    skill: dict[str, Any] | None = None,
):
    tools = _build_benchmark_tools(db_path, no_tools=no_tools)
    # Oracle skill: expose only the tools declared by the skill (+ log_conclusion).
    if skill is not None:
        allowed = set(skill["tools"]) | {"log_conclusion"}
        available = {t.name for t in tools}
        unknown = set(skill["tools"]) - available
        if unknown:
            raise ValueError(
                f"Skill '{skill['name']}' declares unknown tools: {sorted(unknown)}"
            )
        tools = [t for t in tools if t.name in allowed]
    # No-tools mode injects the full dataset, so prefill is large and slow. Give the
    # agent a generous timeout and disable retries — a retry just re-sends the whole
    # prompt and restarts prompt processing from scratch (the "restart" loop). The
    # max_tokens cap bounds runaway reasoning so a single task can't generate forever.
    llm_with_tools = get_llm(
        model_name=model_name,
        provider=provider,
        timeout=agent_timeout or BENCHMARK_AGENT_TIMEOUT,
        max_retries=0,
        max_tokens=agent_max_tokens or BENCHMARK_AGENT_MAX_TOKENS,
    ).bind_tools(tools)
    tool_node = ToolNode(tools)

    # In no-tools mode, serialize the whole dataset once and inject it into the
    # system message on every agent step.
    data_context = serialize_patient_data(db_path) if no_tools else None

    def agent_node(state: MessagesState):
        return _agent_node(
            state,
            llm_with_tools,
            data_context=data_context,
            skill_text=skill["body"] if skill else None,
        )

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _should_continue, ["tools", END])
    builder.add_edge("tools", "agent")
    return builder.compile()


# ---------------------------------------------------------------------------
# Trace capture helpers
# ---------------------------------------------------------------------------


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _usage_of(message: Any) -> dict[str, int]:
    """Token counts reported by the endpoint for one model call (0 if missing)."""
    usage = getattr(message, "usage_metadata", None) or {}
    details = usage.get("output_token_details") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "reasoning_tokens": int(details.get("reasoning") or 0),
    }


def _token_stats(ai_messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-call token usage for one task."""
    calls = [m["usage"] for m in ai_messages if m.get("type") == "ai" and "usage" in m]
    inputs = [u["input_tokens"] for u in calls]
    return {
        "llm_calls": len(calls),
        "input_tokens_total": sum(inputs),
        "output_tokens_total": sum(u["output_tokens"] for u in calls),
        "reasoning_tokens_total": sum(u["reasoning_tokens"] for u in calls),
        "first_call_input_tokens": inputs[0] if inputs else 0,
        "max_input_tokens": max(inputs) if inputs else 0,
    }


def _capture_messages(
    messages: list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], str]:
    """Extract ai_messages, tool_calls, tool_results, final_response from LangGraph messages."""
    ai_messages: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: dict[str, str] = {}
    final_response = ""

    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                call_id = str(call.get("id", ""))
                tool_calls.append(
                    {
                        "id": call_id,
                        "name": call.get("name"),
                        "args": call.get("args", {}),
                    }
                )

            text = _content_to_text(message.content).strip()
            ai_messages.append(
                {
                    "type": "ai",
                    "content": text,
                    "usage": _usage_of(message),
                    "tool_calls": [
                        {
                            "id": str(c.get("id", "")),
                            "name": c.get("name"),
                            "args": c.get("args", {}),
                        }
                        for c in (message.tool_calls or [])
                    ],
                }
            )
            if text:
                final_response = text

        if isinstance(message, ToolMessage):
            tool_results[str(message.tool_call_id or "")] = _content_to_text(
                message.content
            )
            ai_messages.append(
                {
                    "type": "tool",
                    "tool_call_id": str(message.tool_call_id or ""),
                    "content": _content_to_text(message.content)[:2000],
                }
            )

    return ai_messages, tool_calls, tool_results, final_response


# ---------------------------------------------------------------------------
# Single-turn runner
# ---------------------------------------------------------------------------


def run_single_task(
    graph: Any,
    task: dict[str, Any],
) -> dict[str, Any]:
    prompt_text = f"\n[Task ID: {task['task_id']}] {task['prompt']}"
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    state = graph.invoke(
        {"messages": [HumanMessage(content=prompt_text)]},
        config={"recursion_limit": 24},
    )
    messages = state.get("messages", []) if isinstance(state, dict) else []

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    ai_messages, tool_calls, tool_results, final_response = _capture_messages(messages)

    return {
        "task_id": task["task_id"],
        "prompt": prompt_text,
        "started_at": started_at,
        "latency_ms": latency_ms,
        "messages": ai_messages,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "final_response": final_response,
        "turn_count": len([m for m in ai_messages if m["type"] == "ai"]),
        "tool_call_count": len(tool_calls),
        **_token_stats(ai_messages),
    }


# ---------------------------------------------------------------------------
# Multi-turn runner
# ---------------------------------------------------------------------------


def _apply_injections(
    task: dict[str, Any],
    turn_number: int,
    db_path: str,
    kb_store: Any | None = None,
) -> None:
    """Apply pre-turn state injections (cgm_db_insert, kb_insert, delay)."""
    script = task.get("user_simulator", {}).get("script", [])
    turn_spec = next((s for s in script if s.get("turn") == turn_number), None)
    if not turn_spec:
        return

    inject = turn_spec.get("inject_events")
    if not inject:
        return

    itype = inject.get("type")

    if itype == "cgm_db_insert":
        conn = sqlite3.connect(db_path)
        try:
            for row in inject.get("rows", []):
                conn.execute(
                    """
                    INSERT INTO cgm_readings(
                        patient_id, factory_timestamp_utc, device_timestamp_local,
                        glucose_mmol_l, trend, source, raw_json, inserted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(patient_id, factory_timestamp_utc, glucose_mmol_l)
                    DO UPDATE SET glucose_mmol_l=excluded.glucose_mmol_l
                    """,
                    (
                        row.get("patient_id", "540"),
                        row.get("factory_timestamp_utc"),
                        row.get("device_timestamp_local"),
                        row.get("glucose_mmol_l"),
                        row.get("trend"),
                        row.get("source", "benchmark_inject"),
                        row.get("raw_json", "{}"),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    elif itype == "kb_insert" and kb_store is not None:
        for event in inject.get("events", []):
            etype = event.get("event_type")
            if etype == "meal":
                kb_store.add_meal_event(
                    timestamp=event["timestamp"],
                    carbs_g=event["carbs_g"],
                    meal_type=event["meal_type"],
                    notes=event.get("notes", ""),
                )
            elif etype == "insulin":
                kb_store.add_insulin_event(
                    timestamp=event["timestamp"],
                    units=event["units"],
                    insulin_type=event["insulin_type"],
                    timing_tag=event["timing_tag"],
                    notes=event.get("notes", ""),
                )
            elif etype == "glucose":
                kb_store.add_glucose_event(
                    timestamp=event["timestamp"],
                    glucose_mmol_l=event["glucose_mmol_l"],
                    source=event.get("source", "manual"),
                )

    # Handle delay
    delay = inject.get("delay_seconds")
    if delay:
        time.sleep(float(delay))


def _should_stop(
    task: dict[str, Any], turn_number: int, turn_trace: dict[str, Any]
) -> bool:
    """Check whether the multi-turn task should stop early after this turn."""
    script = task.get("user_simulator", {}).get("script", [])
    turn_spec = next((s for s in script if s.get("turn") == turn_number), None)
    if not turn_spec:
        return False

    stop_condition = turn_spec.get("stop_condition", "any")
    if stop_condition == "any":
        return False

    if stop_condition == "tool_called":
        called = {c.get("name") for c in turn_trace.get("tool_calls", [])}
        expected = set(turn_spec.get("expected_tools", []))
        if expected and called & expected:
            return True

    if stop_condition == "response_contains":
        criteria = turn_spec.get("stop_criteria", {})
        response = turn_trace.get("final_response", "").lower()
        expected_substrings = [
            s.lower() for s in criteria.get("expected_substrings", [])
        ]
        if expected_substrings and all(s in response for s in expected_substrings):
            return True

    return False


def run_multi_turn_task(
    graph: Any,
    task: dict[str, Any],
    db_path: str,
    kb_store: Any | None = None,
    simulator_provider: str | None = None,
    simulator_model: str | None = None,
) -> dict[str, Any]:
    """Run a multi-turn task using a user simulator.

    Supports three simulator types:
      - scripted: deterministic turn-by-turn script
      - llm: LLM-based user that generates the next message
      - evaluator (tau-bench style): LLM that plays the user AND evaluates the
        agent in real-time, with the ability to REJECT and terminate early

    Returns a trace with:
      - 'turns': list of per-turn trace fragments
      - 'messages': flat list (backward compatible)
      - 'tool_calls': cumulative tool calls across all turns
      - 'final_response': the last agent response
      - 'evaluator_rejection': rejection details if evaluator terminated early
    """
    simulator = build_simulator(
        task, provider=simulator_provider, model_override=simulator_model
    )
    is_evaluator = isinstance(simulator, EvaluatorSimulator)
    max_turns = task.get(
        "max_turns",
        simulator.max_turns if hasattr(simulator, "max_turns") else 5,
    )

    messages: list[Any] = []
    turns: list[dict[str, Any]] = []
    all_tool_calls: list[dict[str, Any]] = []
    all_tool_results: dict[str, str] = {}
    evaluator_rejection: dict[str, Any] | None = None
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    for turn_number in range(1, max_turns + 1):
        # Get the next user message — evaluator simulators use next_action()
        if is_evaluator:
            evaluator_action = simulator.next_action(turns)
            if evaluator_action is None:
                break
            if isinstance(evaluator_action, RejectAction):
                # Evaluator rejected the agent — record and stop
                evaluator_rejection = simulator.reject_reason
                break
            user_message = evaluator_action.message
        else:
            user_message = simulator.next_message(turns)
            if user_message is None:
                break

        # Apply any pre-turn state injections
        _apply_injections(task, turn_number, db_path, kb_store)

        # Run the graph with accumulated history
        # Prepend /no_think on the first turn only to suppress Qwen3 extended reasoning
        prefixed_message = f"\n{user_message}" if turn_number == 1 else user_message
        state = graph.invoke(
            {"messages": messages + [HumanMessage(content=prefixed_message)]},
            config={"recursion_limit": 24},
        )
        new_messages = state.get("messages", []) if isinstance(state, dict) else []
        turn_messages_raw = new_messages[len(messages) :]
        messages = new_messages

        # Capture turn-level trace
        ai_msgs, turn_tool_calls, turn_tool_results, final_response = _capture_messages(
            turn_messages_raw
        )

        turn_trace = {
            "turn_number": turn_number,
            "user_message": user_message,
            "messages": ai_msgs,
            "tool_calls": turn_tool_calls,
            "tool_results": turn_tool_results,
            "final_response": final_response,
            "tool_call_count": len(turn_tool_calls),
        }
        turns.append(turn_trace)
        all_tool_calls.extend(turn_tool_calls)
        all_tool_results.update(turn_tool_results)

        # Check stop condition (scripted task stop conditions)
        if not is_evaluator and _should_stop(task, turn_number, turn_trace):
            break

        # For evaluator simulators, also allow scripted stop conditions
        if is_evaluator and _should_stop(task, turn_number, turn_trace):
            break

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Build flat messages from turns for backward compatibility
    flat_messages: list[dict[str, Any]] = []
    for turn in turns:
        flat_messages.extend(turn["messages"])

    return {
        "task_id": task["task_id"],
        "prompt": task.get("description", task.get("prompt", "")),
        "started_at": started_at,
        "latency_ms": latency_ms,
        "messages": flat_messages,
        "turns": turns,
        "tool_calls": all_tool_calls,
        "tool_results": all_tool_results,
        "final_response": turns[-1]["final_response"] if turns else "",
        "turn_count": len(turns),
        "tool_call_count": len(all_tool_calls),
        **_token_stats(flat_messages),
        **(
            {"evaluator_rejection": evaluator_rejection}
            if evaluator_rejection is not None
            else {}
        ),
    }


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------


def load_tasks(tasks_path: str = "benchmark_tasks.json") -> list[dict[str, Any]]:
    path = Path(tasks_path)
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")
    tasks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("benchmark_tasks.json must be a JSON array of task objects")
    return tasks


def select_tasks(
    tasks: list[dict[str, Any]], task_ids: list[str] | None
) -> list[dict[str, Any]]:
    if not task_ids:
        return tasks
    id_set = set(task_ids)
    selected = [t for t in tasks if t["task_id"] in id_set]
    missing = id_set - {t["task_id"] for t in selected}
    if missing:
        raise ValueError(f"Task IDs not found: {sorted(missing)}")
    return selected


def filter_tasks_by_mode(
    tasks: list[dict[str, Any]], mode: str
) -> list[dict[str, Any]]:
    """Filter tasks according to the selected benchmark mode."""
    if mode == "all":
        return tasks
    if mode == "stateful":
        return [t for t in tasks if t.get("category") == "logging"]
    if mode == "evaluator":
        sim_config = tasks[0].get("user_simulator", {}) if tasks else {}
        return [
            t for t in tasks if t.get("user_simulator", {}).get("type") == "evaluator"
        ]
    if mode == "multi_turn":
        return [
            t
            for t in tasks
            if t.get("mode") == "multi_turn" or t.get("multi_turn") is True
        ]
    # read-only (default)
    return [
        t
        for t in tasks
        if t.get("category") != "logging"
        and t.get("mode") != "multi_turn"
        and t.get("multi_turn") is not True
    ]


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------


def run_benchmark(
    *,
    tasks: list[dict[str, Any]],
    db_path: str,
    model_name: str | None = None,
    kb_store: Any | None = None,
    provider: str | None = None,
    simulator_provider: str | None = None,
    simulator_model: str | None = None,
    no_tools: bool = False,
    agent_timeout: int | None = None,
    agent_max_tokens: int | None = None,
    skills_mode: str = "none",
) -> list[dict[str, Any]]:
    # Baseline agent: all tools, no skill.
    graph = build_benchmark_graph(
        db_path=db_path,
        model_name=model_name,
        provider=provider,
        no_tools=no_tools,
        agent_timeout=agent_timeout,
        agent_max_tokens=agent_max_tokens,
    )
    traces: list[dict[str, Any]] = []
    # One agent per skill: built the first time the skill is needed, then reused.
    skill_graphs: dict[str, Any] = {}
    for task in tasks:
        # Default: run the task with the baseline agent.
        task_graph, skill = graph, None
        # Oracle mode: pick the skill from the task category, if one is mapped.
        if skills_mode == "oracle" and not no_tools:
            skill = load_oracle_skill(task.get("category", ""))
            if skill is not None:
                if skill["name"] not in skill_graphs:
                    skill_graphs[skill["name"]] = build_benchmark_graph(
                        db_path=db_path,
                        model_name=model_name,
                        provider=provider,
                        no_tools=no_tools,
                        agent_timeout=agent_timeout,
                        agent_max_tokens=agent_max_tokens,
                        skill=skill,
                    )
                task_graph = skill_graphs[skill["name"]]
                print(
                    f"[benchmark] Oracle skill '{skill['name']}' for {task['task_id']}"
                )
        print(
            f"[benchmark] Running {task['task_id']}: {task.get('prompt', '')[:60]}..."
        )
        try:
            if task.get("mode") == "multi_turn" or task.get("multi_turn") is True:
                trace = run_multi_turn_task(
                    task_graph,
                    task,
                    db_path=db_path,
                    kb_store=kb_store,
                    simulator_provider=simulator_provider,
                    simulator_model=simulator_model,
                )
            else:
                trace = run_single_task(task_graph, task)
        except Exception as exc:
            print(f"[benchmark] TASK ERROR {task['task_id']}: {exc}")
            trace = {
                "task_id": task["task_id"],
                "error": str(exc),
                "messages": [],
                "tool_calls": [],
                "final_response": "",
            }
        # Record which skill (if any) was used, so results can be analysed later.
        trace["skill"] = skill["name"] if skill else None
        traces.append(trace)
    return traces
