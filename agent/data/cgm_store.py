from __future__ import annotations

import atexit
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import CGM_DB_PATH

MG_DL_TO_MMOL_L = 18.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def coerce_iso(value: str | datetime | None) -> str:
    if value is None:
        return utc_now_iso()
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class CGMStore:
    def __init__(self, db_path: str = CGM_DB_PATH):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._ensure_schema()
        self._migrate_schema()

    def _ensure_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS cgm_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            factory_timestamp_utc TEXT NOT NULL,
            device_timestamp_local TEXT,
            glucose_mmol_l REAL NOT NULL,
            trend INTEGER,
            source TEXT NOT NULL DEFAULT 'librelinkup',
            raw_json TEXT,
            inserted_at TEXT NOT NULL,
            UNIQUE(patient_id, factory_timestamp_utc, glucose_mmol_l)
        );

        CREATE INDEX IF NOT EXISTS idx_cgm_patient_ts
            ON cgm_readings(patient_id, factory_timestamp_utc);

        CREATE INDEX IF NOT EXISTS idx_cgm_ts
            ON cgm_readings(factory_timestamp_utc);

        CREATE TABLE IF NOT EXISTS cgm_sync_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cgm_enrichment_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            point_id TEXT NOT NULL,
            timestamp_iso TEXT NOT NULL,
            base_notes TEXT NOT NULL DEFAULT '',
            window_minutes INTEGER NOT NULL,
            ready_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(event_type, point_id)
        );

        CREATE INDEX IF NOT EXISTS idx_cgm_enrichment_ready
            ON cgm_enrichment_queue(status, ready_at);
        """
        with self._lock:
            self._conn.executescript(schema)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def clear_all(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cgm_readings")
            self._conn.execute("DELETE FROM cgm_sync_state")
            self._conn.execute("DELETE FROM cgm_enrichment_queue")
            self._conn.commit()

    def set_sync_state(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO cgm_sync_state(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, value),
            )
            self._conn.commit()

    def get_sync_state(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM cgm_sync_state WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return str(row["value"])

    def get_sync_status(self) -> dict[str, Any]:
        queue = self.get_enrichment_queue_counts()
        return {
            "last_sync_started_at": self.get_sync_state("last_sync_started_at"),
            "last_sync_completed_at": self.get_sync_state("last_sync_completed_at"),
            "last_sync_result": self.get_sync_state("last_sync_result"),
            "last_sync_error": self.get_sync_state("last_sync_error"),
            "reading_count": self.count_readings(),
            "enrichment_queue": queue,
        }

    def count_readings(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM cgm_readings"
            ).fetchone()
        return int(row["c"] if row else 0)

    def get_enrichment_queue_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT status, COUNT(*) AS c
                FROM cgm_enrichment_queue
                GROUP BY status
                """
            ).fetchall()
        counts = {"pending": 0, "done": 0, "failed": 0}
        for row in rows:
            status = str(row["status"])
            counts[status] = int(row["c"])
        return counts

    def enqueue_log_enrichment(
        self,
        *,
        event_type: str,
        point_id: str,
        timestamp_iso: str,
        base_notes: str,
        window_minutes: int,
    ) -> int:
        ts = coerce_iso(timestamp_iso)
        ready_at = (parse_iso(ts) + timedelta(minutes=int(window_minutes))).isoformat()
        now_iso = utc_now_iso()

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO cgm_enrichment_queue(
                    event_type,
                    point_id,
                    timestamp_iso,
                    base_notes,
                    window_minutes,
                    ready_at,
                    status,
                    attempts,
                    last_error,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, '', ?, ?)
                ON CONFLICT(event_type, point_id)
                DO UPDATE SET
                    timestamp_iso=excluded.timestamp_iso,
                    base_notes=excluded.base_notes,
                    window_minutes=excluded.window_minutes,
                    ready_at=excluded.ready_at,
                    status='pending',
                    last_error='',
                    updated_at=excluded.updated_at
                """,
                (
                    event_type,
                    point_id,
                    ts,
                    base_notes,
                    int(window_minutes),
                    ready_at,
                    now_iso,
                    now_iso,
                ),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT id FROM cgm_enrichment_queue WHERE event_type = ? AND point_id = ?",
                (event_type, point_id),
            ).fetchone()

        return int(row["id"] if row else 0)

    def get_due_enrichment_jobs(
        self, *, limit: int = 50, max_attempts: int = 24
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        now_iso = utc_now_iso()

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, event_type, point_id, timestamp_iso, base_notes,
                       window_minutes, ready_at, attempts, last_error
                FROM cgm_enrichment_queue
                WHERE status = 'pending'
                  AND ready_at <= ?
                  AND attempts < ?
                ORDER BY ready_at ASC, id ASC
                LIMIT ?
                """,
                (now_iso, int(max_attempts), safe_limit),
            ).fetchall()

        return [dict(row) for row in rows]

    def mark_enrichment_done(self, *, job_id: int) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE cgm_enrichment_queue
                SET status = 'done', updated_at = ?, last_error = ''
                WHERE id = ?
                """,
                (utc_now_iso(), int(job_id)),
            )
            self._conn.commit()

    def mark_enrichment_retry(self, *, job_id: int, error: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE cgm_enrichment_queue
                SET attempts = attempts + 1,
                    last_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (str(error), utc_now_iso(), int(job_id)),
            )
            self._conn.commit()

    def mark_enrichment_failed(self, *, job_id: int, error: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE cgm_enrichment_queue
                SET status = 'failed',
                    attempts = attempts + 1,
                    last_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (str(error), utc_now_iso(), int(job_id)),
            )
            self._conn.commit()

    def _migrate_schema(self) -> None:
        with self._lock:
            cursor = self._conn.execute("PRAGMA table_info(cgm_readings)")
            columns = [row["name"] for row in cursor.fetchall()]
            if "glucose_mg_dl" in columns and "glucose_mmol_l" not in columns:
                self._conn.execute(
                    "ALTER TABLE cgm_readings RENAME COLUMN glucose_mg_dl TO glucose_mmol_l"
                )
                self._conn.execute(
                    "UPDATE cgm_readings SET glucose_mmol_l = glucose_mmol_l / ?",
                    (MG_DL_TO_MMOL_L,),
                )
                self._conn.commit()

    def upsert_cgm_reading(
        self,
        *,
        patient_id: str,
        factory_timestamp_utc: str | datetime,
        device_timestamp_local: str | datetime | None,
        glucose_mmol_l: float,
        trend: int | None = None,
        source: str = "librelinkup",
        raw_json: str | None = None,
    ) -> bool:
        patient = str(patient_id or "unknown")
        factory_iso = coerce_iso(factory_timestamp_utc)
        local_iso = (
            coerce_iso(device_timestamp_local)
            if device_timestamp_local is not None
            else None
        )

        with self._lock:
            existing = self._conn.execute(
                """
                SELECT 1 FROM cgm_readings
                WHERE patient_id = ? AND factory_timestamp_utc = ? AND glucose_mmol_l = ?
                LIMIT 1
                """,
                (patient, factory_iso, float(glucose_mmol_l)),
            ).fetchone()

            self._conn.execute(
                """
                INSERT INTO cgm_readings(
                    patient_id,
                    factory_timestamp_utc,
                    device_timestamp_local,
                    glucose_mmol_l,
                    trend,
                    source,
                    raw_json,
                    inserted_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(patient_id, factory_timestamp_utc, glucose_mmol_l)
                DO UPDATE SET
                    device_timestamp_local=excluded.device_timestamp_local,
                    trend=excluded.trend,
                    source=excluded.source,
                    raw_json=excluded.raw_json,
                    inserted_at=excluded.inserted_at
                """,
                (
                    patient,
                    factory_iso,
                    local_iso,
                    float(glucose_mmol_l),
                    trend,
                    source,
                    raw_json,
                    utc_now_iso(),
                ),
            )
            self._conn.commit()

        return existing is None

    def get_readings(
        self,
        *,
        start_iso: str,
        end_iso: str,
        limit: int = 500,
        patient_id: str | None = None,
    ) -> list[dict[str, Any]]:
        start = coerce_iso(start_iso)
        end = coerce_iso(end_iso)
        safe_limit = max(1, min(int(limit), 5000))

        args: list[Any] = [start, end]
        where = "factory_timestamp_utc >= ? AND factory_timestamp_utc <= ?"
        if patient_id:
            where += " AND patient_id = ?"
            args.append(str(patient_id))

        args.append(safe_limit)

        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT patient_id, factory_timestamp_utc, device_timestamp_local,
                       glucose_mmol_l, trend, source
                FROM cgm_readings
                WHERE {where}
                ORDER BY factory_timestamp_utc ASC
                LIMIT ?
                """,
                args,
            ).fetchall()

        return [dict(row) for row in rows]

    def get_glucose_series(
        self, *, start_iso: str, end_iso: str, patient_id: str | None = None
    ) -> list[tuple[str, float]]:
        """All (timestamp, glucose_mmol_l) pairs in a range, oldest first.

        Unlike get_readings there is no row cap, so aggregate metrics cover the
        whole period even for sensors that report every minute.
        """
        start = coerce_iso(start_iso)
        end = coerce_iso(end_iso)
        args: list[Any] = [start, end]
        where = "factory_timestamp_utc >= ? AND factory_timestamp_utc <= ?"
        if patient_id:
            where += " AND patient_id = ?"
            args.append(str(patient_id))

        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT factory_timestamp_utc, glucose_mmol_l
                FROM cgm_readings
                WHERE {where}
                ORDER BY factory_timestamp_utc ASC
                """,
                args,
            ).fetchall()

        return [(str(r[0]), float(r[1])) for r in rows]

    def get_nearest_reading(
        self,
        *,
        timestamp_iso: str,
        window_minutes: int = 90,
        direction: str = "any",
        patient_id: str | None = None,
    ) -> dict[str, Any] | None:
        target_dt = parse_iso(coerce_iso(timestamp_iso))
        start_dt = target_dt - timedelta(minutes=window_minutes)
        end_dt = target_dt + timedelta(minutes=window_minutes)
        rows = self.get_readings(
            start_iso=start_dt.isoformat(),
            end_iso=end_dt.isoformat(),
            limit=5000,
            patient_id=patient_id,
        )
        if not rows:
            return None

        best: dict[str, Any] | None = None
        best_delta: float | None = None
        for row in rows:
            ts = row.get("factory_timestamp_utc")
            if not isinstance(ts, str):
                continue
            ts_dt = parse_iso(ts)
            delta_sec = (ts_dt - target_dt).total_seconds()
            if direction == "forward" and delta_sec < 0:
                continue
            if direction == "backward" and delta_sec > 0:
                continue
            abs_delta = abs(delta_sec)
            if best_delta is None or abs_delta < best_delta:
                best = dict(row)
                best_delta = abs_delta

        if best is None:
            return None

        best["delta_minutes"] = round(
            (parse_iso(best["factory_timestamp_utc"]) - target_dt).total_seconds()
            / 60.0,
            1,
        )
        return best

    def get_post_log_context(
        self,
        *,
        timestamp_iso: str,
        window_minutes: int = 90,
        patient_id: str | None = None,
    ) -> dict[str, Any] | None:
        start_dt = parse_iso(coerce_iso(timestamp_iso))
        end_dt = start_dt + timedelta(minutes=window_minutes)
        rows = self.get_readings(
            start_iso=start_dt.isoformat(),
            end_iso=end_dt.isoformat(),
            limit=5000,
            patient_id=patient_id,
        )
        if not rows:
            return None

        values = [float(row["glucose_mmol_l"]) for row in rows]
        first = rows[0]
        last = rows[-1]
        start_val = float(first["glucose_mmol_l"])
        end_val = float(last["glucose_mmol_l"])
        peak_val = max(values)
        trough_val = min(values)
        return {
            "reading_count": len(rows),
            "start_timestamp": first["factory_timestamp_utc"],
            "end_timestamp": last["factory_timestamp_utc"],
            "start_glucose_mmol_l": start_val,
            "end_glucose_mmol_l": end_val,
            "max_glucose_mmol_l": peak_val,
            "min_glucose_mmol_l": trough_val,
            "delta_mmol_l": end_val - start_val,
            "peak_rise_mmol_l": peak_val - start_val,
            "window_minutes": int(window_minutes),
            "readings": rows,
        }

    def summarize(
        self, *, days: int = 7, low: float = 3.9, high: float = 10.0
    ) -> dict[str, Any] | None:
        if days <= 0:
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        rows = self.get_readings(
            start_iso=cutoff.isoformat(),
            end_iso=datetime.now(timezone.utc).isoformat(),
            limit=5000,
        )
        if not rows:
            return None

        values = [float(row["glucose_mmol_l"]) for row in rows]
        avg = sum(values) / len(values)
        in_range = sum(1 for v in values if low <= v <= high)
        avg_mg_dl = avg * MG_DL_TO_MMOL_L
        return {
            "days": days,
            "count": len(values),
            "avg": avg,
            "min": min(values),
            "max": max(values),
            "tir": (in_range / len(values)) * 100,
            "estimated_a1c": (avg_mg_dl + 46.7) / 28.7,
        }


def _json_default(value: Any) -> str:
    return str(value)


def reading_to_json(reading: Any) -> str:
    model_dump = getattr(reading, "model_dump", None)
    if callable(model_dump):
        try:
            return json.dumps(model_dump(mode="json"), default=_json_default)
        except Exception:
            pass
    try:
        return json.dumps(reading, default=_json_default)
    except Exception:
        return str(reading)


cgm_store = CGMStore()
atexit.register(cgm_store.close)
