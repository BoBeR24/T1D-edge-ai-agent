# agent/data/cgm_analytics.py
# Pure glucose analytics shared by the agent tools and the benchmark tools.
# Everything here works on plain (timestamp, glucose_mmol_l) pairs, oldest
# first, so it can be tested without a sensor, a database or an LLM.
from __future__ import annotations

from datetime import datetime
from typing import Any

from agent.data.cgm_store import MG_DL_TO_MMOL_L, parse_iso

# International consensus on time in ranges (Battelino et al., Diabetes Care 2019).
# Thresholds are in mg/dL, the unit the consensus defines them in, so that a
# reading of exactly 70 or 180 mg/dL lands in range instead of being pushed out
# by mmol/L rounding (70 / 18 = 3.889 < 3.9).
TBR2_MG_DL = 54.0  # level 2 hypoglycaemia: < 54 mg/dL (< 3.0 mmol/L)
TBR1_MG_DL = 70.0  # below range: < 70 mg/dL (< 3.9 mmol/L)
TAR1_MG_DL = 180.0  # above range: > 180 mg/dL (> 10.0 mmol/L)
TAR2_MG_DL = 250.0  # level 2 hyperglycaemia: > 250 mg/dL (> 13.9 mmol/L)
CV_TARGET_PERCENT = 36.0
MIN_DAYS_OF_DATA = 14
MIN_SENSOR_COVERAGE_PERCENT = 70.0


def _typical_interval_minutes(timestamps: list[datetime]) -> float | None:
    """Median gap between consecutive readings, i.e. the sensor's sampling rate."""
    gaps = sorted(
        (b - a).total_seconds() / 60.0
        for a, b in zip(timestamps, timestamps[1:])
        if b > a
    )
    if not gaps:
        return None
    return gaps[len(gaps) // 2]


def compute_clinical_cgm_metrics(
    series: list[tuple[str, float]], period_days: int
) -> dict[str, Any] | None:
    """Consensus CGM metrics from (timestamp, glucose_mmol_l) pairs.

    Percentages are shares of readings, which equal shares of time when the
    sensor samples at a fixed rate.
    """
    if not series:
        return None

    timestamps = [parse_iso(ts) for ts, _ in series]
    mg_dl = [round(v * MG_DL_TO_MMOL_L, 1) for _, v in series]
    n = len(mg_dl)

    def pct(count: int) -> float:
        return 100.0 * count / n

    mean_mg_dl = sum(mg_dl) / n
    sd_mg_dl = (sum((v - mean_mg_dl) ** 2 for v in mg_dl) / n) ** 0.5

    interval = _typical_interval_minutes(timestamps)
    if interval:
        coverage = min(100.0, 100.0 * n * interval / (period_days * 24 * 60))
    else:
        coverage = None

    return {
        "count": n,
        "mean_mmol_l": mean_mg_dl / MG_DL_TO_MMOL_L,
        # Glucose management indicator (Bergenstal et al., Diabetes Care 2018).
        "gmi_percent": 3.31 + 0.02392 * mean_mg_dl,
        "cv_percent": 100.0 * sd_mg_dl / mean_mg_dl if mean_mg_dl else 0.0,
        "tbr_level2": pct(sum(1 for v in mg_dl if v < TBR2_MG_DL)),
        "tbr_level1": pct(sum(1 for v in mg_dl if TBR2_MG_DL <= v < TBR1_MG_DL)),
        "tir": pct(sum(1 for v in mg_dl if TBR1_MG_DL <= v <= TAR1_MG_DL)),
        "tar_level1": pct(sum(1 for v in mg_dl if TAR1_MG_DL < v <= TAR2_MG_DL)),
        "tar_level2": pct(sum(1 for v in mg_dl if v > TAR2_MG_DL)),
        "interval_minutes": interval,
        "coverage_percent": coverage,
    }


# (start_timestamp, end_timestamp, start_value, end_value, delta)
Excursion = tuple[str, str, float, float, float]


def _detect_rises(
    times: list[datetime],
    values: list[float],
    threshold: float,
    window_minutes: float,
) -> list[tuple[int, int]]:
    """Index pairs (start, peak) of distinct rises of at least `threshold`
    reached within `window_minutes`. Each rise is reported once."""
    window_seconds = window_minutes * 60.0
    events: list[tuple[int, int]] = []
    n = len(values)
    i = 0
    while i < n - 1:
        hit = None
        j = i + 1
        while j < n and (times[j] - times[i]).total_seconds() <= window_seconds:
            if values[j] - values[i] >= threshold:
                hit = j
                break
            j += 1
        if hit is None:
            i += 1
            continue

        # The rise starts at the last lowest reading before the threshold is hit.
        start = min(range(i, hit), key=lambda k: (values[k], -k))

        # Follow the rise to its peak; a full reverse move ends it.
        peak = hit
        k = hit + 1
        while k < n and (times[k] - times[start]).total_seconds() <= window_seconds:
            if values[k] <= values[peak] - threshold:
                break
            if values[k] > values[peak]:
                peak = k
            k += 1

        if events and events[-1][1] == start:
            # Still the same climb, just longer than one window.
            events[-1] = (events[-1][0], peak)
        else:
            events.append((start, peak))
        i = peak
    return events


def detect_glucose_excursions(
    series: list[tuple[str, float]],
    *,
    min_rise_mmol_l: float,
    min_drop_mmol_l: float,
    window_minutes: float,
) -> tuple[list[Excursion], list[Excursion]]:
    """Find distinct glucose spikes and dips.

    A spike is a rise of at least `min_rise_mmol_l` reached within
    `window_minutes`; a dip is the same for a drop of `min_drop_mmol_l`.
    One continuous rise (or fall) counts as one event, from its lowest
    (highest) reading to its peak (trough), however many readings it spans.
    """
    stamps = [ts for ts, _ in series]
    times = [parse_iso(ts) for ts in stamps]
    values = [float(v) for _, v in series]

    def build(pairs: list[tuple[int, int]]) -> list[Excursion]:
        return [
            (stamps[a], stamps[b], values[a], values[b], values[b] - values[a])
            for a, b in pairs
        ]

    spikes = build(_detect_rises(times, values, min_rise_mmol_l, window_minutes))
    dips = build(
        _detect_rises(times, [-v for v in values], min_drop_mmol_l, window_minutes)
    )
    return spikes, dips


def format_excursions(
    spikes: list[Excursion],
    dips: list[Excursion],
    *,
    min_rise_mmol_l: float,
    min_drop_mmol_l: float,
    window_minutes: float,
    max_listed: int = 10,
) -> str:
    """Two text blocks, spikes then dips, listing at most `max_listed` each."""

    def block(label: str, word: str, sign: str, threshold: float, events: list[Excursion]) -> str:
        head = (
            f"{label}: {len(events)} detected "
            f"(threshold {sign}{threshold:.1f} mmol/L within {window_minutes:g} min)"
        )
        if len(events) > max_listed:
            head += f", first {max_listed} listed"
        lines = [
            f"- {s} -> {e}: {sv:.1f} → {ev:.1f} mmol/L ({word} {d:+.1f})"
            for s, e, sv, ev, d in events[:max_listed]
        ]
        return "\n".join([head + ".", *lines])

    return "\n\n".join(
        [
            block("Spikes", "rise", "+", min_rise_mmol_l, spikes),
            block("Dips", "drop", "-", min_drop_mmol_l, dips),
        ]
    )
