# agent/data/cgm_analytics.py
# Pure glucose analytics shared by the agent tools and the benchmark tools.
# Everything here works on plain (timestamp, glucose_mmol_l) pairs, oldest
# first, so it can be tested without a sensor, a database or an LLM.
from __future__ import annotations

from datetime import datetime
from agent.data.cgm_store import parse_iso

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
