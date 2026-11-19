# tests/test_excursions.py
# Spike / dip detection: one rise is one spike, and only within the time window.
from datetime import datetime, timedelta, timezone

from agent.data.cgm_analytics import detect_glucose_excursions
from agent.data.cgm_store import cgm_store
from agent.tools import get_cgm_spikes_from_db

START = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _series(values, interval_minutes=5):
    return [
        ((START + timedelta(minutes=i * interval_minutes)).isoformat(), float(v))
        for i, v in enumerate(values)
    ]


def _detect(values, rise=2.2, drop=2.2, window=120, interval_minutes=5):
    return detect_glucose_excursions(
        _series(values, interval_minutes),
        min_rise_mmol_l=rise,
        min_drop_mmol_l=drop,
        window_minutes=window,
    )


def _one_meal_day():
    """A flat day at 6.0 with one rise to 10.0 at noon and back down."""
    values = []
    for k in range(288):
        minute = k * 5
        if 720 <= minute <= 780:
            values.append(6.0 + 4.0 * (minute - 720) / 60)
        elif 780 < minute <= 840:
            values.append(10.0 - 4.0 * (minute - 780) / 60)
        else:
            values.append(6.0)
    return values


class TestDetectGlucoseExcursions:
    def test_one_rise_is_one_spike_and_one_dip(self):
        spikes, dips = _detect(_one_meal_day())

        assert len(spikes) == 1
        start_ts, end_ts, start_val, end_val, delta = spikes[0]
        assert start_ts == (START + timedelta(minutes=720)).isoformat()
        assert end_ts == (START + timedelta(minutes=780)).isoformat()
        assert (start_val, end_val, delta) == (6.0, 10.0, 4.0)

        assert len(dips) == 1
        assert dips[0][0] == (START + timedelta(minutes=780)).isoformat()
        assert dips[0][1] == (START + timedelta(minutes=840)).isoformat()
        assert dips[0][4] == -4.0

    def test_slow_drift_outside_the_window_is_not_a_spike(self):
        # +3.0 mmol/L spread evenly over 14 hours: never +2.2 within 2 hours.
        values = [6.0 + 3.0 * k / 168 for k in range(169)]
        spikes, dips = _detect(values)
        assert spikes == []
        assert dips == []

    def test_window_minutes_changes_the_result(self):
        # +3.0 over 90 minutes: a spike for a 120-minute window, not for 30.
        values = [6.0] * 6 + [6.0 + 3.0 * k / 18 for k in range(1, 19)] + [9.0] * 6
        assert len(_detect(values, window=120)[0]) == 1
        assert _detect(values, window=30)[0] == []

    def test_two_separate_meals_are_two_spikes(self):
        day = _one_meal_day()
        evening = [10.0 if 216 <= k <= 222 else v for k, v in enumerate(day)]
        spikes, _ = _detect(evening)
        assert len(spikes) == 2

    def test_long_climb_is_reported_once(self):
        # 6 -> 14 over four hours: longer than one window, still a single rise.
        values = [6.0] * 6 + [6.0 + 8.0 * k / 48 for k in range(1, 49)] + [14.0] * 6
        spikes, _ = _detect(values)
        assert len(spikes) == 1
        assert spikes[0][2] == 6.0
        assert spikes[0][3] == 14.0

    def test_flat_and_tiny_series(self):
        assert _detect([6.0] * 50) == ([], [])
        assert _detect([6.0]) == ([], [])
        assert _detect([]) == ([], [])


class TestSpikeTool:
    def test_tool_counts_one_meal_once(self):
        newest = datetime.now(timezone.utc) - timedelta(seconds=30)
        day = _one_meal_day()
        for i, value in enumerate(day):
            ts = newest - timedelta(minutes=(len(day) - 1 - i) * 5)
            cgm_store.upsert_cgm_reading(
                patient_id="test-patient",
                factory_timestamp_utc=ts.isoformat(),
                device_timestamp_local=ts.isoformat(),
                glucose_mmol_l=value,
                trend=3,
            )

        out = get_cgm_spikes_from_db.invoke({"days": 2})
        assert "Spikes: 1 detected" in out
        assert "Dips: 1 detected" in out
        assert "6.0 → 10.0 mmol/L (rise +4.0)" in out
