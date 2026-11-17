from __future__ import annotations

import hashlib
import importlib
import math
import uuid
import atexit
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import KB_EMBEDDING_MODEL, KB_QDRANT_PATH


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


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


class Embedder:
    def __init__(self, model_name: str):
        self._model = None
        self._size = 384
        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
            SentenceTransformer = getattr(sentence_transformers, "SentenceTransformer")
            self._model = SentenceTransformer(model_name)
            self._size = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            self._model = None

    @property
    def size(self) -> int:
        return self._size

    def embed(self, text: str) -> list[float]:
        if self._model is not None:
            vector = self._model.encode(text)
            return [float(x) for x in vector]

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [digest[i % len(digest)] / 255.0 for i in range(self._size)]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]


class InMemoryVectorBackend:
    def __init__(self):
        self._collections: dict[str, dict[str, tuple[list[float], dict[str, Any]]]] = {}

    def ensure_collection(self, name: str) -> None:
        self._collections.setdefault(name, {})

    def clear(self, name: str) -> None:
        self._collections[name] = {}

    def upsert(self, collection: str, point_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self._collections.setdefault(collection, {})[point_id] = (vector, payload)

    def search(
        self,
        collection: str,
        query_vector: list[float],
        *,
        limit: int,
        since_cutoff_iso: str | None,
        until_cutoff_iso: str | None,
        since_field: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for vector, payload in self._collections.get(collection, {}).values():
            ts = payload.get(since_field)
            if isinstance(ts, str):
                if since_cutoff_iso and ts < since_cutoff_iso:
                    continue
                if until_cutoff_iso and ts > until_cutoff_iso:
                    continue
            row = dict(payload)
            row["score"] = cosine_similarity(vector, query_vector)
            rows.append(row)
        rows.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return rows[:limit]

    def scroll(
        self,
        collection: str,
        *,
        since_cutoff_iso: str | None,
        until_cutoff_iso: str | None,
        since_field: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for _, payload in self._collections.get(collection, {}).values():
            ts = payload.get(since_field)
            if isinstance(ts, str):
                if since_cutoff_iso and ts < since_cutoff_iso:
                    continue
                if until_cutoff_iso and ts > until_cutoff_iso:
                    continue
            rows.append(dict(payload))
        rows.sort(key=lambda r: r.get("timestamp", ""))
        return rows

    def close(self) -> None:
        return None


class QdrantBackend:
    def __init__(self, path: str, vector_size: int):
        qdrant_client = importlib.import_module("qdrant_client")
        qdrant_models = importlib.import_module("qdrant_client.models")
        QdrantClient = getattr(qdrant_client, "QdrantClient")
        Distance = getattr(qdrant_models, "Distance")
        VectorParams = getattr(qdrant_models, "VectorParams")

        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=path)
        self._vector_size = vector_size
        self._distance = Distance.COSINE
        self._vector_params = VectorParams

    def ensure_collection(self, name: str) -> None:
        try:
            self._client.get_collection(name)
        except Exception:
            self._client.create_collection(
                collection_name=name,
                vectors_config=self._vector_params(size=self._vector_size, distance=self._distance),
            )

    def clear(self, name: str) -> None:
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        self.ensure_collection(name)

    def upsert(self, collection: str, point_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        qdrant_models = importlib.import_module("qdrant_client.models")
        PointStruct = getattr(qdrant_models, "PointStruct")

        self._client.upsert(
            collection_name=collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    def search(
        self,
        collection: str,
        query_vector: list[float],
        *,
        limit: int,
        since_cutoff_iso: str | None,
        until_cutoff_iso: str | None,
        since_field: str,
    ) -> list[dict[str, Any]]:
        query_filter = None
        conditions = []
        if since_cutoff_iso is not None or until_cutoff_iso is not None:
            qdrant_models = importlib.import_module("qdrant_client.models")
            DatetimeRange = getattr(qdrant_models, "DatetimeRange")
            FieldCondition = getattr(qdrant_models, "FieldCondition")
            Filter = getattr(qdrant_models, "Filter")

            conditions.append(FieldCondition(key=since_field, range=DatetimeRange(gte=since_cutoff_iso, lte=until_cutoff_iso)))
            query_filter = Filter(must=conditions)

        response = self._client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        rows: list[dict[str, Any]] = []
        points = getattr(response, "points", [])
        for point in points:
            row = dict(point.payload or {})
            row["score"] = float(point.score)
            rows.append(row)
        return rows

    def scroll(
        self,
        collection: str,
        *,
        since_cutoff_iso: str | None,
        until_cutoff_iso: str | None,
        since_field: str,
    ) -> list[dict[str, Any]]:
        scroll_filter = None
        conditions = []
        if since_cutoff_iso is not None or until_cutoff_iso is not None:
            qdrant_models = importlib.import_module("qdrant_client.models")
            DatetimeRange = getattr(qdrant_models, "DatetimeRange")
            FieldCondition = getattr(qdrant_models, "FieldCondition")
            Filter = getattr(qdrant_models, "Filter")

            conditions.append(FieldCondition(key=since_field, range=DatetimeRange(gte=since_cutoff_iso, lte=until_cutoff_iso)))
            scroll_filter = Filter(must=conditions)

        points, _ = self._client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        rows = [dict(point.payload or {}) for point in points]
        rows.sort(key=lambda r: r.get("timestamp", ""))
        return rows

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


class KnowledgeBaseStore:
    COLLECTIONS = {
        "report_chunks": "report_chunks",
        "glucose_events": "glucose_events",
        "meal_events": "meal_events",
        "insulin_events": "insulin_events",
        "calendar_events": "calendar_events",
    }

    def __init__(self, path: str = KB_QDRANT_PATH, embedding_model: str = KB_EMBEDDING_MODEL):
        self._embedder = Embedder(embedding_model)
        self._backend = self._create_backend(path)
        for collection in self.COLLECTIONS.values():
            self._backend.ensure_collection(collection)

    def _create_backend(self, path: str):
        try:
            return QdrantBackend(path=path, vector_size=self._embedder.size)
        except Exception:
            return InMemoryVectorBackend()

    def clear_all(self) -> None:
        for collection in self.COLLECTIONS.values():
            self._backend.clear(collection)

    def close(self) -> None:
        close_fn = getattr(self._backend, "close", None)
        if callable(close_fn):
            close_fn()

    def _point_id(self, seed: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

    def _search(
        self,
        collection: str,
        query: str,
        *,
        top_k: int,
        since_days: int | None,
        until_days: int | None,
        since_field: str,
    ) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        since_cutoff_iso = (now - timedelta(days=since_days)).isoformat() if since_days is not None else None
        until_cutoff_iso = (now + timedelta(days=until_days)).isoformat() if until_days is not None else None
        return self._backend.search(
            collection,
            self._embedder.embed(query),
            limit=top_k,
            since_cutoff_iso=since_cutoff_iso,
            until_cutoff_iso=until_cutoff_iso,
            since_field=since_field,
        )

    def add_report_document(self, *, report_id: str, file_path: str, content: str, period_days: int) -> int:
        chunks = self.chunk_text(content)
        for index, chunk in enumerate(chunks):
            point_id = self._point_id(f"report:{report_id}:{index}")
            payload = {
                "event_type": "report_chunk",
                "report_id": report_id,
                "file_path": file_path,
                "period_days": int(period_days),
                "chunk_index": index,
                "content": chunk,
                "created_at": utc_now_iso(),
            }
            self._backend.upsert(self.COLLECTIONS["report_chunks"], point_id, self._embedder.embed(chunk), payload)
        return len(chunks)

    def search_reports(self, query: str, *, since_days: int | None = None, until_days: int | None = None, top_k: int = 5) -> list[dict[str, Any]]:
        return self._search(
            self.COLLECTIONS["report_chunks"],
            query,
            top_k=top_k,
            since_days=since_days,
            until_days=until_days,
            since_field="created_at",
        )

    def add_glucose_event(self, *, timestamp: str | datetime, glucose_mmol_l: float, source: str = "cgm") -> str:
        iso = coerce_iso(timestamp)
        point_id = self._point_id(f"glucose:{iso}:{glucose_mmol_l}:{source}")
        payload = {
            "point_id": point_id,
            "event_type": "glucose",
            "timestamp": iso,
            "glucose_mmol_l": float(glucose_mmol_l),
            "source": source,
            "created_at": utc_now_iso(),
        }
        text = f"glucose {glucose_mmol_l:.1f} mmol/L at {iso} from {source}"
        self._backend.upsert(self.COLLECTIONS["glucose_events"], point_id, self._embedder.embed(text), payload)
        return point_id

    def search_glucose(self, query: str, *, since_days: int | None = None, until_days: int | None = None, top_k: int = 10) -> list[dict[str, Any]]:
        return self._search(
            self.COLLECTIONS["glucose_events"],
            query,
            top_k=top_k,
            since_days=since_days,
            until_days=until_days,
            since_field="timestamp",
        )

    def add_meal_event(self, *, timestamp: str | datetime, carbs_g: float, meal_type: str, notes: str = "") -> str:
        iso = coerce_iso(timestamp)
        point_id = self._point_id(f"meal:{iso}:{carbs_g}:{meal_type}:{notes}")
        payload = {
            "point_id": point_id,
            "event_type": "meal",
            "timestamp": iso,
            "carbs_g": float(carbs_g),
            "meal_type": meal_type,
            "notes": notes,
            "created_at": utc_now_iso(),
        }
        text = f"meal {meal_type} carbs {carbs_g:.1f}g at {iso}. {notes}".strip()
        self._backend.upsert(self.COLLECTIONS["meal_events"], point_id, self._embedder.embed(text), payload)
        return point_id

    def add_insulin_event(
        self,
        *,
        timestamp: str | datetime,
        units: float,
        insulin_type: str,
        timing_tag: str,
        notes: str = "",
    ) -> str:
        iso = coerce_iso(timestamp)
        point_id = self._point_id(f"insulin:{iso}:{units}:{insulin_type}:{timing_tag}:{notes}")
        payload = {
            "point_id": point_id,
            "event_type": "insulin",
            "timestamp": iso,
            "units": float(units),
            "insulin_type": insulin_type,
            "timing_tag": timing_tag,
            "notes": notes,
            "created_at": utc_now_iso(),
        }
        text = f"insulin {insulin_type} {units:.2f}U {timing_tag} at {iso}. {notes}".strip()
        self._backend.upsert(self.COLLECTIONS["insulin_events"], point_id, self._embedder.embed(text), payload)
        return point_id

    def add_calendar_event(
        self,
        *,
        event_id: str,
        title: str,
        event_group: str,
        start_time: str | datetime,
        end_time: str | datetime,
        description: str = "",
    ) -> str:

        start_iso = coerce_iso(start_time)
        end_iso = coerce_iso(end_time)
        point_id = self._point_id(f"calendar:{event_id}:{start_iso}:{title}")
        payload = {
            "point_id": point_id,
            "event_type": "calendar",
            "event_id": event_id,
            "title": title,
            "event_group": event_group,
            "timestamp": start_iso,
            "start_time": start_iso,
            "end_time": end_iso,
            "description": description,
            "created_at": utc_now_iso(),
        }
        text = (
            f"calendar event '{title}' from {start_iso} to {end_iso}. "
            f"Description: {description}."
        ).strip()
        self._backend.upsert(
            self.COLLECTIONS["calendar_events"],
            point_id,
            self._embedder.embed(text),
            payload,
        )
        return point_id

    def search_calendar_events(
        self, query: str, *, since_days: int | None = 0, until_days: int | None = 7, top_k: int = 10
    ) -> list[dict[str, Any]]:
        return self._search(
            self.COLLECTIONS["calendar_events"],
            query,
            top_k=top_k,
            since_days=since_days,
            until_days=until_days,
            since_field="timestamp",
        )

    def search_logs(
        self,
        query: str,
        *,
        since_days: int | None = None,
        until_days: int | None = None,
        event_types: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        include_meal = event_types is None or "meal" in event_types
        include_insulin = event_types is None or "insulin" in event_types
        if include_meal:
            rows.extend(
                self._search(
                    self.COLLECTIONS["meal_events"],
                    query,
                    top_k=top_k,
                    since_days=since_days,
                    until_days=until_days,
                    since_field="timestamp",
                )
            )
        if include_insulin:
            rows.extend(
                self._search(
                    self.COLLECTIONS["insulin_events"],
                    query,
                    top_k=top_k,
                    since_days=since_days,
                    until_days=until_days,
                    since_field="timestamp",
                )
            )
        rows.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return rows[:top_k]

    def get_meal_events(self, *, since_days: int = 14, until_days: int | None = None) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        cutoff_start = (now - timedelta(days=since_days)).isoformat() if since_days is not None else None
        cutoff_end = (now + timedelta(days=until_days)).isoformat() if until_days is not None else None
        return self._backend.scroll(
            self.COLLECTIONS["meal_events"],
            since_cutoff_iso=cutoff_start,
            until_cutoff_iso=cutoff_end,
            since_field="timestamp",
        )

    def get_insulin_events(self, *, since_days: int = 14, until_days: int | None = None) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        cutoff_start = (now - timedelta(days=since_days)).isoformat() if since_days is not None else None
        cutoff_end = (now + timedelta(days=until_days)).isoformat() if until_days is not None else None
        return self._backend.scroll(
            self.COLLECTIONS["insulin_events"],
            since_cutoff_iso=cutoff_start,
            until_cutoff_iso=cutoff_end,
            since_field="timestamp",
        )

    def get_calendar_events(
            self,
            *,
            since_days: int = 0,
            until_days: int = 1,
            event_groups: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        cutoff_start = (now - timedelta(days=since_days)).isoformat() if since_days is not None else None
        cutoff_end = (now + timedelta(days=until_days)).isoformat() if until_days is not None else None

        rows = self._backend.scroll(
            self.COLLECTIONS["calendar_events"],
            since_cutoff_iso=cutoff_start,
            until_cutoff_iso=cutoff_end,
            since_field="timestamp",
        )

        if event_groups is not None:
            rows = [r for r in rows if str(r.get("event_group", "")) in event_groups]

        return rows

    # TODO: Add methods for updating/deleting calendar events

    def update_log_event_notes(self, *, event_type: str, point_id: str, notes: str) -> bool:
        if event_type not in {"meal", "insulin"}:
            return False

        collection = self.COLLECTIONS["meal_events"] if event_type == "meal" else self.COLLECTIONS["insulin_events"]
        rows = self._backend.scroll(collection, since_cutoff_iso=None, until_cutoff_iso=None, since_field="timestamp")
        target = next((row for row in rows if str(row.get("point_id", "")) == str(point_id)), None)
        if target is None:
            return False

        payload = dict(target)
        payload["notes"] = notes

        if event_type == "meal":
            text = (
                f"meal {payload.get('meal_type', 'meal')} carbs {float(payload.get('carbs_g', 0.0)):.1f}g "
                f"at {payload.get('timestamp', '')}. {notes}"
            ).strip()
        else:
            text = (
                f"insulin {payload.get('insulin_type', '')} {float(payload.get('units', 0.0)):.2f}U "
                f"{payload.get('timing_tag', '')} at {payload.get('timestamp', '')}. {notes}"
            ).strip()

        self._backend.upsert(collection, str(point_id), self._embedder.embed(text), payload)
        return True

    @staticmethod
    def chunk_text(text: str, *, chunk_size: int = 700, overlap: int = 120) -> list[str]:
        normalized = (text or "").strip()
        if not normalized:
            return [""]
        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            chunks.append(normalized[start:end])
            if end == len(normalized):
                break
            start = max(0, end - overlap)
        return chunks


kb_store = KnowledgeBaseStore()
atexit.register(kb_store.close)