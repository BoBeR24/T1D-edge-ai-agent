# agent/tools.py
# LangChain tools for the DiabetesAgent.
# Add new tools here and append them to TOOLS — no other files need changing.
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pylibrelinkup import PyLibreLinkUp

from agent.data.cgm_store import cgm_store, reading_to_json
from agent.kb.store import coerce_iso, kb_store, parse_iso
from agent.spoonacular import SPOONACULAR_TOOLS
from agent.tavily_search import TAVILY_TOOLS
from config import (
    CGM_LOG_ENRICHMENT_WINDOW_MINUTES,
    INSULIN_RATIOS_PATH,
    LIBRE_EMAIL,
    LIBRE_PASSWORD,
)

MG_DL_TO_MMOL_L = 18.0

VALID_INSULIN_TYPES = {"rapid", "basal", "correction"}

VALID_MEAL_TYPES = {"breakfast", "lunch", "dinner", "snack"}

MEDICAL_DISCLAIMER = (
    "\u26a0\ufe0f This is an informational calculation \u2014 not medical advice. "
    "Always consult your healthcare provider before making dosing decisions."
)


def _load_ratios(path: str | None = None) -> dict[str, float]:
    """Load insulin ratios from the JSON file. Returns empty dict if file missing."""
    file_path = Path(path or INSULIN_RATIOS_PATH)
    if not file_path.exists():
        return {}
    try:
        data = json.loads(file_path.read_text())
        return {
            k: float(v)
            for k, v in data.items()
            if isinstance(v, (int, float)) and v > 0
        }
    except Exception:
        return {}


def _save_ratios(ratios: dict[str, float], path: str | None = None) -> None:
    """Persist insulin ratios to the JSON file."""
    file_path = Path(path or INSULIN_RATIOS_PATH)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(ratios, indent=2))


_init_error = ""
if not LIBRE_EMAIL or not LIBRE_PASSWORD:
    # No live-CGM credentials configured. Skip the network round-trip entirely so
    # importing this module stays fast offline (benchmark and OhioT1DM workflows
    # never need LibreLinkUp); the live-CGM tools report this when called.
    _client = None
    _patients = []
    _init_error = "LIBRE_EMAIL / LIBRE_PASSWORD are not set (see .env.example)."
else:
    try:
        _client = PyLibreLinkUp(email=LIBRE_EMAIL, password=LIBRE_PASSWORD)
        _client.authenticate()
        _patients = _client.get_patients()
    except Exception as exc:
        _client = None
        _patients = []
        _init_error = str(exc)


def _get_patient_identifier():
    if _client is None or not _patients:
        detail = f" Details: {_init_error}" if _init_error else ""
        return None, (
            "LibreLink sensor client is unavailable. "
            "Check credentials and network connectivity."
            f"{detail}"
        )
    return _patients[0], ""


def _patient_id_text(patient_identifier) -> str:
    for attr in ("patient_id", "id"):
        value = getattr(patient_identifier, attr, None)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        if text.startswith("<MagicMock"):
            continue
        return text
    return str(patient_identifier)


def _filtered_readings(days: int):
    patient_identifier, message = _get_patient_identifier()
    if patient_identifier is None:
        return None, message

    client = _client
    if client is None:
        return None, "LibreLink sensor client is unavailable."

    if days <= 0:
        return None, "`days` must be greater than 0."

    try:
        logbook = client.logbook(patient_identifier=patient_identifier)
    except Exception as exc:
        return None, f"Failed to fetch sensor logbook: {exc}"

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    try:
        filtered = [r for r in logbook if r.factory_timestamp >= cutoff]
        filtered.sort(key=lambda r: r.factory_timestamp)
    except (AttributeError, TypeError):
        return None, "Could not filter logbook data: readings are missing timestamps."

    if not filtered:
        return None, f"No glucose readings found in the past {days} days."

    return filtered, ""


def _reading_values(readings):
    try:
        values = [float(r.value_in_mg_per_dl) for r in readings]
    except (AttributeError, TypeError, ValueError):
        return None, "Could not parse glucose values from sensor readings."

    if not values:
        return None, "No glucose values available to analyze."

    return values, ""


def _reading_timestamp_iso(reading) -> str | None:
    ts = getattr(reading, "factory_timestamp", None)
    if ts is None:
        ts = getattr(reading, "timestamp", None)
    if ts is None:
        return None
    try:
        return coerce_iso(ts)
    except Exception:
        return None


def _reading_device_timestamp_iso(reading) -> str | None:
    ts = getattr(reading, "timestamp", None)
    if ts is None:
        return None
    try:
        return coerce_iso(ts)
    except Exception:
        return None


def _active_patient_id() -> str | None:
    stored = cgm_store.get_sync_state("active_patient_id")
    if isinstance(stored, str) and stored.strip():
        return stored.strip()

    patient_identifier, _ = _get_patient_identifier()
    if patient_identifier is not None:
        patient_id = _patient_id_text(patient_identifier).strip()
        if patient_id:
            return patient_id
    return None


def _cgm_context_note(
    timestamp_iso: str, *, window_minutes: int = CGM_LOG_ENRICHMENT_WINDOW_MINUTES
) -> tuple[str, bool]:
    context = cgm_store.get_post_log_context(
        timestamp_iso=timestamp_iso,
        window_minutes=window_minutes,
        patient_id=_active_patient_id(),
    )
    if not context:
        return (
            f"CGM context pending (0–{window_minutes} min post-log window).",
            False,
        )

    note = (
        f"CGM context (0–{window_minutes} min post-log): "
        f"{context['reading_count']} readings, "
        f"start {context['start_glucose_mmol_l']:.1f} mmol/L @ {context['start_timestamp']}, "
        f"end {context['end_glucose_mmol_l']:.1f} mmol/L @ {context['end_timestamp']}, "
        f"min {context['min_glucose_mmol_l']:.1f}, max {context['max_glucose_mmol_l']:.1f}, "
        f"delta {context['delta_mmol_l']:+.1f} mmol/L, "
        f"peak rise {context['peak_rise_mmol_l']:+.1f} mmol/L, "
        f"peak drop {context['min_glucose_mmol_l'] - context['start_glucose_mmol_l']:+.1f} mmol/L."
    )
    return note, True


def _compose_notes_with_context(base_notes: str, context_note: str) -> str:
    notes_clean = base_notes.strip()
    if notes_clean:
        return f"{notes_clean}\n\n{context_note}"
    return context_note


def _enqueue_deferred_log_enrichment(
    *, event_type: str, point_id: str, timestamp_iso: str, base_notes: str
) -> int:
    return cgm_store.enqueue_log_enrichment(
        event_type=event_type,
        point_id=point_id,
        timestamp_iso=timestamp_iso,
        base_notes=base_notes,
        window_minutes=CGM_LOG_ENRICHMENT_WINDOW_MINUTES,
    )


def _process_deferred_log_enrichment(
    limit: int = 25, max_attempts: int = 24
) -> dict[str, int]:
    jobs = cgm_store.get_due_enrichment_jobs(limit=limit, max_attempts=max_attempts)
    processed = 0
    updated = 0
    retried = 0
    failed = 0

    for job in jobs:
        processed += 1
        try:
            event_type = str(job.get("event_type", ""))
            point_id = str(job.get("point_id", ""))
            timestamp_iso = str(job.get("timestamp_iso", ""))
            base_notes = str(job.get("base_notes", ""))
            job_id = int(job.get("id", 0))
            window_minutes = int(
                job.get("window_minutes", CGM_LOG_ENRICHMENT_WINDOW_MINUTES)
            )

            context_note, available = _cgm_context_note(
                timestamp_iso, window_minutes=window_minutes
            )
            if not available:
                cgm_store.mark_enrichment_retry(
                    job_id=job_id, error="post-log window not populated yet"
                )
                retried += 1
                continue

            merged_notes = _compose_notes_with_context(base_notes, context_note)
            ok = kb_store.update_log_event_notes(
                event_type=event_type, point_id=point_id, notes=merged_notes
            )
            if not ok:
                cgm_store.mark_enrichment_failed(
                    job_id=job_id, error="target log event not found in KB"
                )
                failed += 1
                continue

            cgm_store.mark_enrichment_done(job_id=job_id)
            updated += 1
        except Exception as exc:
            cgm_store.mark_enrichment_retry(
                job_id=int(job.get("id", 0)), error=str(exc)
            )
            retried += 1

    return {
        "processed": processed,
        "updated": updated,
        "retried": retried,
        "failed": failed,
    }


@tool
def get_latest_glucose() -> str:
    """Fetch the most recent blood glucose reading from the LibreLink CGM sensor."""
    patient_identifier, message = _get_patient_identifier()
    if patient_identifier is None:
        return message

    client = _client
    if client is None:
        return "LibreLink sensor client is unavailable."

    reading = client.latest(patient_identifier=patient_identifier)
    return str(reading)


@tool
def get_glucose_logbook() -> str:
    """Fetch the full recent glucose history (logbook) from the LibreLink CGM sensor."""
    patient_identifier, message = _get_patient_identifier()
    if patient_identifier is None:
        return message

    client = _client
    if client is None:
        return "LibreLink sensor client is unavailable."

    logbook = client.logbook(patient_identifier=patient_identifier)
    return "\n".join(str(reading) for reading in logbook)


@tool
def get_glucose_history(days: int = 7) -> str:
    """Fetch glucose readings from the past N days from the LibreLink CGM sensor."""
    readings, message = _filtered_readings(days)
    if readings is not None:
        return f"{len(readings)} readings over the past {days} days:\n" + "\n".join(
            f"- {reading}" for reading in readings
        )

    if "missing timestamps" in message:
        patient_identifier, patient_message = _get_patient_identifier()
        if patient_identifier is None:
            return patient_message

        client = _client
        if client is None:
            return "LibreLink sensor client is unavailable."

        logbook = client.logbook(patient_identifier=patient_identifier)
        # Fall back to returning all logbook data if filtering fails
        return "\n".join(str(reading) for reading in logbook)

    return message.replace("glucose readings", "readings")


@tool
def sync_cgm_to_db(days: int = 14) -> str:
    """Sync CGM glucose readings from LibreLinkUp into the SQLite CGM time-series store."""
    cgm_store.set_sync_state(
        "last_sync_started_at", datetime.now(timezone.utc).isoformat()
    )

    patient_identifier, patient_message = _get_patient_identifier()
    if patient_identifier is None:
        cgm_store.set_sync_state("last_sync_error", patient_message)
        return patient_message

    patient_id = _patient_id_text(patient_identifier)
    cgm_store.set_sync_state("active_patient_id", patient_id)

    readings, message = _filtered_readings(days)
    if readings is None:
        cgm_store.set_sync_state("last_sync_error", message)
        return message

    inserted = 0
    updated = 0
    skipped = 0

    for reading in readings:
        factory_iso = _reading_timestamp_iso(reading)
        local_iso = _reading_device_timestamp_iso(reading)
        value = getattr(reading, "value_in_mg_per_dl", None)
        trend_raw = getattr(reading, "trend", None)

        if factory_iso is None or value is None:
            skipped += 1
            continue

        trend: int | None
        try:
            trend = int(trend_raw) if trend_raw is not None else None
        except Exception:
            trend = None

        try:
            was_insert = cgm_store.upsert_cgm_reading(
                patient_id=patient_id,
                factory_timestamp_utc=factory_iso,
                device_timestamp_local=local_iso,
                glucose_mmol_l=float(value) / MG_DL_TO_MMOL_L,
                trend=trend,
                source="librelinkup",
                raw_json=reading_to_json(reading),
            )
            if was_insert:
                inserted += 1
            else:
                updated += 1
        except Exception:
            skipped += 1

    result = (
        f"Synced CGM readings to SQLite: {inserted} inserted, {updated} updated, {skipped} skipped "
        f"(window: past {days} days)."
    )
    cgm_store.set_sync_state(
        "last_sync_completed_at", datetime.now(timezone.utc).isoformat()
    )
    cgm_store.set_sync_state("last_sync_result", result)
    cgm_store.set_sync_state("last_sync_error", "")
    return result


@tool
def sync_latest_cgm_to_db() -> str:
    """Fetch latest CGM reading from LibreLinkUp and upsert it into SQLite."""
    cgm_store.set_sync_state(
        "last_sync_started_at", datetime.now(timezone.utc).isoformat()
    )

    patient_identifier, message = _get_patient_identifier()
    if patient_identifier is None:
        cgm_store.set_sync_state("last_sync_error", message)
        return message

    client = _client
    if client is None:
        error = "LibreLink sensor client is unavailable."
        cgm_store.set_sync_state("last_sync_error", error)
        return error

    try:
        reading = client.latest(patient_identifier=patient_identifier)
    except Exception as exc:
        error = f"Failed to fetch latest CGM reading: {exc}"
        cgm_store.set_sync_state("last_sync_error", error)
        return error

    factory_iso = _reading_timestamp_iso(reading)
    local_iso = _reading_device_timestamp_iso(reading)
    value = getattr(reading, "value_in_mg_per_dl", None)
    trend_raw = getattr(reading, "trend", None)
    if factory_iso is None or value is None:
        error = "Latest CGM reading is missing timestamp/value and could not be stored."
        cgm_store.set_sync_state("last_sync_error", error)
        return error

    trend: int | None
    try:
        trend = int(trend_raw) if trend_raw is not None else None
    except Exception:
        trend = None

    patient_id = _patient_id_text(patient_identifier)
    cgm_store.set_sync_state("active_patient_id", patient_id)
    try:
        was_insert = cgm_store.upsert_cgm_reading(
            patient_id=patient_id,
            factory_timestamp_utc=factory_iso,
            device_timestamp_local=local_iso,
            glucose_mmol_l=float(value) / MG_DL_TO_MMOL_L,
            trend=trend,
            source="librelinkup-latest",
            raw_json=reading_to_json(reading),
        )
    except Exception as exc:
        error = f"Failed to store latest CGM reading in SQLite: {exc}"
        cgm_store.set_sync_state("last_sync_error", error)
        return error

    action = "inserted" if was_insert else "updated"
    result = (
        "Synced latest CGM reading to SQLite: "
        f"{action} {float(value):.1f} mmol/L at {factory_iso}."
    )
    cgm_store.set_sync_state(
        "last_sync_completed_at", datetime.now(timezone.utc).isoformat()
    )
    cgm_store.set_sync_state("last_sync_result", result)
    cgm_store.set_sync_state("last_sync_error", "")
    return result


@tool
def get_cgm_readings(start_iso: str, end_iso: str, limit: int = 500) -> str:
    """Read CGM glucose readings from SQLite for a UTC time range."""
    if limit <= 0:
        return "`limit` must be greater than 0."

    try:
        start = coerce_iso(start_iso)
        end = coerce_iso(end_iso)
    except Exception as exc:
        return f"Invalid timestamp input: {exc}"

    rows = cgm_store.get_readings(start_iso=start, end_iso=end, limit=limit)
    if not rows:
        return "No CGM readings found in that range."

    lines = [
        f"{row['factory_timestamp_utc']}: {float(row['glucose_mmol_l']):.1f} mmol/L"
        for row in rows
    ]
    return (
        f"CGM readings from {start} to {end} ({len(rows)} rows, limit {limit}):\n"
        + "\n".join(f"- {line}" for line in lines)
    )


@tool
def get_recent_cgm_readings(days: int = 2, limit: int = 500) -> str:
    """Read recent CGM readings from SQLite for the past N days."""
    if days <= 0:
        return "`days` must be greater than 0."
    if limit <= 0:
        return "`limit` must be greater than 0."

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    rows = cgm_store.get_readings(
        start_iso=start_dt.isoformat(), end_iso=end_dt.isoformat(), limit=limit
    )
    if not rows:
        return f"No CGM readings found in the past {days} days."

    lines = [
        f"{row['factory_timestamp_utc']}: {float(row['glucose_mmol_l']):.1f} mmol/L"
        for row in rows
    ]
    return (
        f"Recent CGM readings (past {days} days, {len(rows)} rows, limit {limit}):\n"
        + "\n".join(f"- {line}" for line in lines)
    )


@tool
def get_cgm_summary_from_db(days: int = 7, low: float = 3.9, high: float = 10.0) -> str:
    """Return glucose summary stats from the SQLite CGM store."""
    if days <= 0:
        return "`days` must be greater than 0."
    if low >= high:
        return "`low` must be less than `high`."

    summary = cgm_store.summarize(days=days, low=low, high=high)
    if summary is None:
        return f"No CGM readings found in SQLite for the past {days} days."

    return (
        f"SQLite CGM summary (past {days} days, {summary['count']} readings): "
        f"avg {summary['avg']:.1f} mmol/L, min {summary['min']:.1f}, max {summary['max']:.1f}, "
        f"time in range {low}–{high} mmol/L: {summary['tir']:.1f}%, "
        f"estimated A1C: {summary['estimated_a1c']:.2f}%."
    )


@tool
def get_cgm_spikes_from_db(
    days: int = 7,
    min_rise_mmol_l: float = 2.2,
    min_drop_mmol_l: float = 2.2,
    window_minutes: int = 120,
) -> str:
    """Detect notable glucose spikes and dips from SQLite CGM readings."""
    if days <= 0:
        return "`days` must be greater than 0."
    if min_rise_mmol_l <= 0:
        return "`min_rise_mmol_l` must be greater than 0."
    if min_drop_mmol_l <= 0:
        return "`min_drop_mmol_l` must be greater than 0."
    if window_minutes <= 0:
        return "`window_minutes` must be greater than 0."

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    rows = cgm_store.get_readings(
        start_iso=start_dt.isoformat(), end_iso=end_dt.isoformat(), limit=5000
    )
    if len(rows) < 2:
        return f"Not enough CGM readings in SQLite over the past {days} days to detect excursions."

    spikes: list[tuple[str, str, float, float, float]] = []
    dips: list[tuple[str, str, float, float, float]] = []

    for i, start in enumerate(rows):
        start_ts = parse_iso(str(start["factory_timestamp_utc"]))
        start_val = float(start["glucose_mmol_l"])
        found_spike = False
        found_dip = False

        for end in rows[i + 1 :]:
            end_ts = parse_iso(str(end["factory_timestamp_utc"]))
            delta_minutes = (end_ts - start_ts).total_seconds() / 60.0
            if delta_minutes < 0:
                continue
            if delta_minutes > window_minutes:
                break

            end_val = float(end["glucose_mmol_l"])
            delta = end_val - start_val

            if (not found_spike) and delta >= min_rise_mmol_l:
                spikes.append(
                    (
                        str(start["factory_timestamp_utc"]),
                        str(end["factory_timestamp_utc"]),
                        start_val,
                        end_val,
                        delta,
                    )
                )
                found_spike = True

            if (not found_dip) and delta <= -min_drop_mmol_l:
                dips.append(
                    (
                        str(start["factory_timestamp_utc"]),
                        str(end["factory_timestamp_utc"]),
                        start_val,
                        end_val,
                        delta,
                    )
                )
                found_dip = True

            if found_spike and found_dip:
                break

    if not spikes and not dips:
        return (
            f"No spikes/dips found in SQLite over the past {days} days "
            f"with rise >= {min_rise_mmol_l:.1f} mmol/L or drop >= {min_drop_mmol_l:.1f} mmol/L "
            f"in <= {window_minutes} minutes."
        )

    spike_lines = [
        f"{start_ts} -> {end_ts}: {start_val:.1f} → {end_val:.1f} mmol/L (rise {delta:+.1f})"
        for start_ts, end_ts, start_val, end_val, delta in spikes[:10]
    ]
    dip_lines = [
        f"{start_ts} -> {end_ts}: {start_val:.1f} → {end_val:.1f} mmol/L (drop {delta:+.1f})"
        for start_ts, end_ts, start_val, end_val, delta in dips[:10]
    ]

    blocks = [
        (
            f"Spikes: {len(spikes)} detected "
            f"(threshold +{min_rise_mmol_l:.1f} mmol/L within {window_minutes} min)."
            + (
                "\n" + "\n".join(f"- {line}" for line in spike_lines)
                if spike_lines
                else ""
            )
        ),
        (
            f"Dips: {len(dips)} detected "
            f"(threshold -{min_drop_mmol_l:.1f} mmol/L within {window_minutes} min)."
            + ("\n" + "\n".join(f"- {line}" for line in dip_lines) if dip_lines else "")
        ),
    ]

    return f"SQLite CGM excursions over past {days} days:\n" + "\n\n".join(blocks)


@tool
def get_nearest_cgm_reading(timestamp: str, window_minutes: int = 90) -> str:
    """Find the nearest CGM reading in SQLite to a target timestamp."""
    if window_minutes <= 0:
        return "`window_minutes` must be greater than 0."

    try:
        target = coerce_iso(timestamp)
    except Exception as exc:
        return f"Invalid timestamp input: {exc}"

    row = cgm_store.get_nearest_reading(
        timestamp_iso=target,
        window_minutes=window_minutes,
        direction="any",
    )
    if row is None:
        return f"No CGM reading found within ±{window_minutes} minutes of {target}."

    return (
        f"Nearest CGM reading to {target}: {row['factory_timestamp_utc']} "
        f"{float(row['glucose_mmol_l']):.1f} mmol/L "
        f"(delta {float(row['delta_minutes']):+.1f} min)."
    )


@tool
def run_deferred_log_enrichment(limit: int = 25, max_attempts: int = 24) -> str:
    """Process pending deferred meal/insulin log enrichment jobs from the SQLite queue."""
    if limit <= 0:
        return "`limit` must be greater than 0."
    if max_attempts <= 0:
        return "`max_attempts` must be greater than 0."

    result = _process_deferred_log_enrichment(limit=limit, max_attempts=max_attempts)
    return (
        "Deferred log enrichment run: "
        f"processed {result['processed']}, updated {result['updated']}, "
        f"retried {result['retried']}, failed {result['failed']}."
    )


@tool
def sync_glucose_to_kb(days: int = 7) -> str:
    """Sync CGM glucose readings from the past N days into the vector knowledge base."""
    readings, message = _filtered_readings(days)
    if readings is None:
        return message

    inserted = 0
    skipped = 0
    for reading in readings:
        timestamp = _reading_timestamp_iso(reading)
        value = getattr(reading, "value_in_mg_per_dl", None)
        if timestamp is None or value is None:
            skipped += 1
            continue
        try:
            kb_store.add_glucose_event(
                timestamp=timestamp,
                glucose_mmol_l=float(value) / MG_DL_TO_MMOL_L,
                source="cgm",
            )
            inserted += 1
        except Exception:
            skipped += 1

    return (
        f"Synced glucose readings to KB: {inserted} upserted, {skipped} skipped "
        f"(window: past {days} days)."
    )


@tool
def get_time_in_range(days: int = 7, low: float = 3.9, high: float = 10.0) -> str:
    """Calculate the percentage of time glucose levels were within a target range
    over the past N days. Default range is 3.9–10.0 mmol/L (standard clinical target)."""
    import pandas as pd
    from glucostats.extract_statistics import ExtractGlucoStats

    if low >= high:
        return "`low` must be less than `high`."

    filtered, message = _filtered_readings(days)
    if filtered is None:
        return message

    values, parse_message = _reading_values(filtered)
    if values is None:
        return parse_message

    values_mmol = [v / MG_DL_TO_MMOL_L for v in values]
    df = pd.DataFrame(
        {
            "timestamp": [
                getattr(r, "factory_timestamp", getattr(r, "timestamp", None))
                for r in filtered
            ],
            "glucose_mmol_l": values_mmol,
        }
    )
    df.index = ["patient"] * len(df)

    try:
        extractor = ExtractGlucoStats(["pt_ir"])
        extractor.configuration(in_range_interval=[low, high], time_units="h")
        stats = extractor.transform(df)
        tir = stats["pt_ir"].iloc[0]
    except Exception as exc:
        values_mmol = [v / MG_DL_TO_MMOL_L for v in values]
        in_range_count = sum(1 for v in values_mmol if low <= v <= high)
        tir = (in_range_count / len(values_mmol)) * 100
        return (
            "Time in range computed with fallback method "
            f"({low:.1f}-{high:.1f} mmol/L) over the past {days} days: {tir:.1f}% "
            f"(based on {len(values_mmol)} readings). "
            f"Reason glucostats failed: {exc}"
        )

    return (
        f"Time in range ({low}–{high} mmol/L) over the past {days} days: "
        f"{tir:.1f}% (based on {len(filtered)} readings)."
    )


@tool
def get_average_glucose(days: int = 7) -> str:
    """Calculate average glucose for the past N days."""
    readings, message = _filtered_readings(days)
    if readings is None:
        return message

    values, parse_message = _reading_values(readings)
    if values is None:
        return parse_message

    avg = sum(values) / len(values) / MG_DL_TO_MMOL_L
    return (
        f"Average glucose over the past {days} days: "
        f"{avg:.1f} mmol/L (based on {len(values)} readings)."
    )


@tool
def get_estimated_a1c(days: int = 14) -> str:
    """Estimate A1C from CGM glucose values over the past N days.

    Uses the ADAG conversion: A1C = (average glucose in mg/dL + 46.7) / 28.7.
    """
    readings, message = _filtered_readings(days)
    if readings is None:
        return message

    values, parse_message = _reading_values(readings)
    if values is None:
        return parse_message

    avg_mg_dl = sum(values) / len(values)
    avg_mmol_l = avg_mg_dl / MG_DL_TO_MMOL_L
    est_a1c = (avg_mg_dl + 46.7) / 28.7
    return (
        f"Estimated A1C over the past {days} days: {est_a1c:.2f}% "
        f"(average glucose {avg_mmol_l:.1f} mmol/L; {len(values)} readings)."
    )


@tool
def get_glucose_summary(days: int = 7, low: float = 3.9, high: float = 10.0) -> str:
    """Return a compact glucose summary for the past N days.

    Includes average, min, max, estimated A1C, and target-range coverage.
    """
    readings, message = _filtered_readings(days)
    if readings is None:
        return message

    values, parse_message = _reading_values(readings)
    if values is None:
        return parse_message

    if low >= high:
        return "`low` must be less than `high`."

    values_mmol = [v / MG_DL_TO_MMOL_L for v in values]
    avg = sum(values_mmol) / len(values_mmol)
    min_val = min(values_mmol)
    max_val = max(values_mmol)
    in_range_count = sum(1 for v in values_mmol if low <= v <= high)
    tir = (in_range_count / len(values_mmol)) * 100
    avg_mg_dl = sum(values) / len(values)
    est_a1c = (avg_mg_dl + 46.7) / 28.7

    return (
        f"Glucose summary (past {days} days, {len(values)} readings): "
        f"avg {avg:.1f} mmol/L, min {min_val:.1f}, max {max_val:.1f}, "
        f"time in range {low:.1f}–{high:.1f} mmol/L: {tir:.1f}%, "
        f"estimated A1C: {est_a1c:.2f}%."
    )


@tool
def log_glucose(value: float, timestamp: str | None = None) -> str:
    """Log a blood glucose reading in mmol/L. Optionally include a timestamp (ISO format)."""
    iso = coerce_iso(timestamp)
    # `value` is already in mmol/L (per the tool's contract), so store it as-is.
    # It must NOT be divided by MG_DL_TO_MMOL_L — that conversion only applies to raw
    # mg/dL source data (e.g. LibreLinkUp readings), not user-supplied mmol/L input.
    kb_store.add_glucose_event(
        timestamp=iso, glucose_mmol_l=float(value), source="manual"
    )
    if timestamp:
        return f"Logged glucose: {value} mmol/L at {iso}"
    return f"Logged glucose: {value} mmol/L"


@tool
def log_meal(
    carbs_g: float, meal_type: str, timestamp: str | None = None, notes: str = ""
) -> str:
    """Log a meal event with carbohydrate grams for future trend and I:C tracking."""
    iso = coerce_iso(timestamp)
    context_note, context_available = _cgm_context_note(iso)
    enriched_notes = _compose_notes_with_context(notes, context_note)
    point_id = kb_store.add_meal_event(
        timestamp=iso,
        carbs_g=float(carbs_g),
        meal_type=meal_type,
        notes=enriched_notes,
    )

    if not context_available:
        _enqueue_deferred_log_enrichment(
            event_type="meal",
            point_id=point_id,
            timestamp_iso=iso,
            base_notes=notes,
        )

    return f"Logged meal: {meal_type}, {carbs_g:.1f}g carbs at {iso}. {context_note}"


@tool
def log_insulin(
    units: float,
    insulin_type: str,
    timing_tag: str,
    timestamp: str | None = None,
    notes: str = "",
) -> str:
    """Log an insulin event. insulin_type must be one of: rapid, basal, correction."""
    insulin_type = insulin_type.strip().lower()
    if insulin_type not in VALID_INSULIN_TYPES:
        valid = ", ".join(sorted(VALID_INSULIN_TYPES))
        return f"Invalid insulin_type '{insulin_type}'. Use one of: {valid}."

    iso = coerce_iso(timestamp)
    context_note, context_available = _cgm_context_note(iso)
    enriched_notes = _compose_notes_with_context(notes, context_note)
    point_id = kb_store.add_insulin_event(
        timestamp=iso,
        units=float(units),
        insulin_type=insulin_type,
        timing_tag=timing_tag,
        notes=enriched_notes,
    )

    if not context_available:
        _enqueue_deferred_log_enrichment(
            event_type="insulin",
            point_id=point_id,
            timestamp_iso=iso,
            base_notes=notes,
        )

    return (
        f"Logged insulin: {units:.2f}U {insulin_type} ({timing_tag}) at {iso}. "
        f"{context_note}"
    )


# TODO: For now we consider notifications as just another type of calendar event, this should be changed to a proper notification implementation, or at least a separate database
@tool
def add_notification_events_to_kb(notifications: list[dict[str, Any]]) -> str:
    """Add a list of scheduled notification events to the knowledge base.

    Each notification in the list should be a dictionary containing:
    - title (str)
    - start_time (str)
    - end_time (str, optional, defaults to start_time)
    - description (str, optional)
    """
    if not isinstance(notifications, list):
        return "Error: 'notifications' must be a list of event dictionaries."

    added_count = 0
    errors = []

    for item in notifications:
        try:
            event_id = str(uuid.uuid4())
            title = item.get("title", "Untitled Notification")
            start_time = item.get("start_time")
            end_time = item.get("end_time", start_time)
            event_group = "notification"
            description = item.get("description", "")

            if not start_time or not end_time:
                errors.append(f"Missing start_time or end_time for notification '{title}'")
                continue

            kb_store.add_calendar_event(
                event_id=event_id,
                title=title,
                start_time=start_time,
                end_time=end_time,
                event_group=event_group,
                description=description,
            )
            added_count += 1
        except Exception as exc:
            errors.append(f"Failed to add '{item.get('title', 'Unknown')}': {exc}")

    result = f"Successfully added {added_count} notification event(s) to the knowledge base."
    if errors:
        result += f" Errors encountered ({len(errors)}): " + "; ".join(errors)
    return result


@tool
def add_calendar_event_to_kb(
        title: str,
        start_time: str,
        end_time: str,
        event_group: str = "general",
        description: str = "",
) -> str:
    """Add a calendar event to the knowledge base.

    event_group should indicate the type of planned event.
    Allowed event_groups values:
    'planned_activity', 'planned_meal', 'notification'
    """
    try:
        point_id = kb_store.add_calendar_event(
            event_id=str(uuid.uuid4()),
            title=title,
            start_time=start_time,
            end_time=end_time,
            event_group=event_group,
            description=description,
        )
        return f"Successfully stored calendar event '{title}' (group: {event_group}) with ID {point_id}."
    except Exception as exc:
        return f"Failed to store calendar event: {exc}"


@tool
def get_calendar_events_from_kb(
        since_days: int = 0,
        until_days: int = 1,
        event_groups: list[str] | None = None,
) -> str:
    """Retrieve calendar events from the knowledge base within a time range. Time range is defined starting from the current time,
    since_days is time delta into the past, until_days is time delta into the future.

    Example:
    - since_days == 0 -- from current time, since_days == 1 -- from the day before.
    - until_days == 0 -- until current time, until_days == 1 -- until the day after.

    Parameters:
    - since_days (int, defaults to 0)
    - until_days (int, defaults to 1)
    - event_groups (list[str], optional, defaults to None) must be a list of groups to filter by. Leave as default to retrieve all events
    Allowed event_groups values:
        'planned_activity', 'planned_meal', 'notification'
    """
    # if event_groups is None:
    #     event_groups = []
    # groups_filter = [g.strip() for g in event_groups.split(",") if g.strip()] if event_groups else None

    try:
        events = kb_store.get_calendar_events(
            since_days=since_days,
            until_days=until_days,
            event_groups=event_groups,
        )
        if not events:
            return f"No calendar events found in the specified range (past {since_days} to future {until_days} days)."

        lines = [
            f"- [{e.get('event_group', 'unknown')}] {e.get('title', 'Untitled')} @ {e.get('start_time', '?')} to {e.get('end_time', '?')}"
            f" (Location: {e.get('location', 'N/A')})"
            for e in events
        ]
        return f"Found {len(events)} calendar events:\n" + "\n".join(lines)
    except Exception as exc:
        return f"Failed to retrieve calendar events: {exc}"


@tool
def search_logged_events(
    query: str, since_days: int = 30, event_types: str = "meal,insulin"
) -> str:
    """Semantic search across manual meal/insulin logs in the knowledge base."""
    parsed_types = [part.strip() for part in event_types.split(",") if part.strip()]
    allowed = {"meal", "insulin"}
    selected = [t for t in parsed_types if t in allowed] or None

    rows = kb_store.search_logs(
        query, since_days=since_days, event_types=selected, top_k=10
    )
    if not rows:
        return "No matching logged events found."

    lines = []
    for row in rows[:5]:
        event_type = row.get("event_type", "unknown")
        timestamp = row.get("timestamp", "unknown time")
        if event_type == "meal":
            lines.append(
                f"- meal @ {timestamp}: {row.get('meal_type', 'meal')}, {row.get('carbs_g', '?')}g carbs"
            )
        elif event_type == "insulin":
            lines.append(
                f"- insulin @ {timestamp}: {row.get('units', '?')}U {row.get('insulin_type', '?')} ({row.get('timing_tag', '')})"
            )
        else:
            lines.append(f"- {event_type} @ {timestamp}")

    return "Matched logged events:\n" + "\n".join(lines)


@tool
def rag_search_reports(query: str, since_days: int = 180, top_k: int = 5) -> str:
    """RAG-style semantic search over historical report content in the vector knowledge base."""
    rows = kb_store.search_reports(query=query, since_days=since_days, top_k=top_k)
    if not rows:
        return "No report context found for that query."

    lines = []
    for row in rows:
        file_path = row.get("file_path", "unknown")
        score = row.get("score", 0.0)
        content = str(row.get("content", "")).replace("\n", " ").strip()
        excerpt = content[:180]
        lines.append(f"- ({score:.3f}) {file_path}: {excerpt}")
    return "Top report matches:\n" + "\n".join(lines)


@tool
def rag_search_glucose_patterns(
    query: str, since_days: int = 30, top_k: int = 10
) -> str:
    """RAG-style semantic search over stored glucose events in the vector knowledge base."""
    rows = kb_store.search_glucose(query=query, since_days=since_days, top_k=top_k)
    if not rows:
        return "No glucose events found for that query."

    lines = []
    for row in rows[:8]:
        lines.append(
            f"- ({row.get('score', 0.0):.3f}) {row.get('timestamp', 'unknown')}: "
            f"{row.get('glucose_mmol_l', '?')} mmol/L [{row.get('source', 'unknown')}]"
        )
    return "Top glucose matches:\n" + "\n".join(lines)


@tool
def get_icr_summary(days: int = 14, pairing_window_minutes: int = 60) -> str:
    """Estimate observed insulin-to-carb patterns from logged meals and insulin events."""
    if days <= 0:
        return "`days` must be greater than 0."
    if pairing_window_minutes <= 0:
        return "`pairing_window_minutes` must be greater than 0."

    meals = kb_store.get_meal_events(since_days=days)
    insulin = [
        event
        for event in kb_store.get_insulin_events(since_days=days)
        if str(event.get("insulin_type", "")).lower() == "rapid"
    ]

    if not meals:
        return f"No meal logs found in the past {days} days."
    if not insulin:
        return f"No rapid insulin logs found in the past {days} days."

    window_seconds = pairing_window_minutes * 60
    ratios = []
    for meal in meals:
        meal_ts = meal.get("timestamp")
        carbs = meal.get("carbs_g")
        if not isinstance(meal_ts, str) or carbs is None:
            continue
        try:
            meal_dt = parse_iso(meal_ts)
            carbs_val = float(carbs)
        except Exception:
            continue

        nearest = None
        nearest_delta = None
        for dose in insulin:
            dose_ts = dose.get("timestamp")
            units = dose.get("units")
            if not isinstance(dose_ts, str) or units is None:
                continue
            try:
                dose_dt = parse_iso(dose_ts)
                units_val = float(units)
            except Exception:
                continue
            if units_val <= 0:
                continue

            delta = abs((dose_dt - meal_dt).total_seconds())
            if delta <= window_seconds and (
                nearest_delta is None or delta < nearest_delta
            ):
                nearest = units_val
                nearest_delta = delta

        if nearest is not None and nearest > 0:
            ratios.append(carbs_val / nearest)

    if not ratios:
        return (
            "No meal/rapid-insulin pairs found within the configured window "
            f"({pairing_window_minutes} min) over the past {days} days."
        )

    ratios.sort()
    median = ratios[len(ratios) // 2]
    avg = sum(ratios) / len(ratios)
    return (
        f"Observed I:C summary (past {days} days): median {median:.1f} g/U, "
        f"mean {avg:.1f} g/U, based on {len(ratios)} matched meal-insulin pairs "
        f"(window {pairing_window_minutes} min)."
    )


@tool
def set_insulin_ratio(meal_type: str, ratio: float) -> str:
    """Save (or overwrite) the insulin-to-carb ratio for a meal time.

    meal_type must be one of: breakfast, lunch, dinner, snack.
    ratio is the number of grams of carbs covered by 1 unit of rapid insulin
    (e.g., ratio=10 means 1 unit covers 10g of carbs).
    """
    meal_type = meal_type.strip().lower()
    if meal_type not in VALID_MEAL_TYPES:
        valid = ", ".join(sorted(VALID_MEAL_TYPES))
        return f"Invalid meal_type '{meal_type}'. Use one of: {valid}."
    if ratio <= 0:
        return "Ratio must be greater than 0."

    ratios = _load_ratios()
    previous = ratios.get(meal_type)
    ratios[meal_type] = float(ratio)
    _save_ratios(ratios)

    ratio_str = f"1:{ratio:.1f}"
    if previous is not None:
        return (
            f"{meal_type.capitalize()} insulin ratio updated to {ratio_str} "
            f"(1 unit per {ratio:.1f}g carbs). Previous ratio was 1:{previous:.1f}."
        )
    return (
        f"{meal_type.capitalize()} insulin ratio set to {ratio_str} "
        f"(1 unit per {ratio:.1f}g carbs)."
    )


@tool
def get_insulin_ratio(meal_type: str) -> str:
    """Get the saved insulin-to-carb ratio for a specific meal time.

    meal_type must be one of: breakfast, lunch, dinner, snack.
    """
    meal_type = meal_type.strip().lower()
    if meal_type not in VALID_MEAL_TYPES:
        valid = ", ".join(sorted(VALID_MEAL_TYPES))
        return f"Invalid meal_type '{meal_type}'. Use one of: {valid}."

    ratios = _load_ratios()
    value = ratios.get(meal_type)
    if value is None:
        return (
            f"No insulin ratio saved for {meal_type}. Use set_insulin_ratio to set one."
        )
    return (
        f"{meal_type.capitalize()} insulin ratio: 1:{value:.1f} "
        f"(1 unit of rapid insulin covers {value:.1f}g of carbs)."
    )


@tool
def list_insulin_ratios() -> str:
    """List all saved insulin-to-carb ratios by meal type."""
    ratios = _load_ratios()
    if not ratios:
        return (
            "No insulin ratios saved yet. "
            "Use set_insulin_ratio to set ratios for breakfast, lunch, dinner, or snack."
        )

    lines = ["Saved insulin-to-carb ratios:"]
    for meal in sorted(VALID_MEAL_TYPES):
        if meal in ratios:
            lines.append(f"  - {meal.capitalize()}: 1:{ratios[meal]:.1f}")
    for meal in sorted(ratios):
        if meal not in VALID_MEAL_TYPES:
            lines.append(f"  - {meal}: 1:{ratios[meal]:.1f}")
    return "\n".join(lines)


@tool
def calculate_insulin_dosage(
    carbs_g: float, meal_type: str, custom_ratio: float | None = None
) -> str:
    """Calculate the recommended rapid insulin dosage for a meal.

    If custom_ratio is provided, use it directly. Otherwise, look up the
    saved ratio for the given meal_type. Dosage = carbs_g / ratio.

    meal_type must be one of: breakfast, lunch, dinner, snack.
    This is an informational calculation ONLY \u2014 not medical advice.
    """
    meal_type = meal_type.strip().lower()
    if meal_type not in VALID_MEAL_TYPES:
        valid = ", ".join(sorted(VALID_MEAL_TYPES))
        return f"Invalid meal_type '{meal_type}'. Use one of: {valid}."
    if carbs_g <= 0:
        return "Carbs must be greater than 0 grams."

    if custom_ratio is not None:
        if custom_ratio <= 0:
            return "Custom ratio must be greater than 0."
        ratio = float(custom_ratio)
        source = "custom"
    else:
        ratios = _load_ratios()
        ratio = ratios.get(meal_type)
        source = meal_type
        if ratio is None:
            return (
                f"No insulin ratio saved for {meal_type}. "
                f"Use set_insulin_ratio to set one, or pass a custom_ratio."
            )

    units = round(carbs_g / ratio, 1)
    return (
        f"For {carbs_g:.0f}g carbs ({source} ratio 1:{ratio:.1f}): "
        f"{units} units of rapid insulin.\n{MEDICAL_DISCLAIMER}"
    )

CORE_TOOLS = [
    get_latest_glucose,
    get_glucose_logbook,
    get_glucose_history,
    sync_cgm_to_db,
    sync_latest_cgm_to_db,
    get_cgm_readings,
    get_recent_cgm_readings,
    get_cgm_summary_from_db,
    get_cgm_spikes_from_db,
    get_nearest_cgm_reading,
    run_deferred_log_enrichment,
    sync_glucose_to_kb,
    get_time_in_range,
    get_average_glucose,
    get_estimated_a1c,
    get_glucose_summary,
    rag_search_reports,
    rag_search_glucose_patterns,
    log_glucose,
    log_meal,
    log_insulin,
    add_calendar_event_to_kb,
    get_calendar_events_from_kb,
    search_logged_events,
    get_icr_summary,
    set_insulin_ratio,
    get_insulin_ratio,
    list_insulin_ratios,
    calculate_insulin_dosage,
    *SPOONACULAR_TOOLS,
    *TAVILY_TOOLS,
]

ALL_TOOLS = [
    *CORE_TOOLS,
    add_notification_events_to_kb
]

# TODO: add more verbose tool descriptions