# tests/test_clinical_metrics.py
# Consensus CGM metrics (time below / in / above range, CV, GMI, data sufficiency).
from datetime import datetime, timedelta, timezone

import pytest

from agent.data.cgm_analytics import compute_clinical_cgm_metrics
from agent.data.cgm_store import cgm_store
from agent.tools import get_cgm_clinical_metrics


def _series(values_mg_dl, interval_minutes=5, start=None):
    start = start or datetime(2030, 1, 1, tzinfo=timezone.utc)
    return [
        ((start + timedelta(minutes=i * interval_minutes)).isoformat(), v / 18.0)
        for i, v in enumerate(values_mg_dl)
    ]


def _store(values_mg_dl, interval_minutes=5):
    # Newest reading 30 s ago, so the oldest still falls inside a window that is
    # exactly n * interval long when the tool computes "now" a moment later.
    # Bulk insert: one upsert per row takes about a minute for two weeks of data.
    newest = datetime.now(timezone.utc) - timedelta(seconds=30)
    n = len(values_mg_dl)
    rows = [
        (
            "test-patient",
            (newest - timedelta(minutes=(n - 1 - i) * interval_minutes)).isoformat(),
            v / 18.0,
            newest.isoformat(),
        )
        for i, v in enumerate(values_mg_dl)
    ]
    with cgm_store._lock:
        cgm_store._conn.executemany(
            "INSERT INTO cgm_readings(patient_id, factory_timestamp_utc, "
            "glucose_mmol_l, inserted_at) VALUES (?, ?, ?, ?)",
            rows,
        )
        cgm_store._conn.commit()


class TestComputeClinicalMetrics:
    def test_range_shares_add_up_and_match_known_split(self):
        # 100 readings: 1 level-2 low, 3 level-1 low, 70 in range,
        # 20 level-1 high, 6 level-2 high.
        values = [50] + [65] * 3 + [120] * 70 + [200] * 20 + [300] * 6
        m = compute_clinical_cgm_metrics(_series(values), period_days=1)

        assert m["tbr_level2"] == pytest.approx(1.0)
        assert m["tbr_level1"] == pytest.approx(3.0)
        assert m["tir"] == pytest.approx(70.0)
        assert m["tar_level1"] == pytest.approx(20.0)
        assert m["tar_level2"] == pytest.approx(6.0)
        total = (
            m["tbr_level2"] + m["tbr_level1"] + m["tir"]
            + m["tar_level1"] + m["tar_level2"]
        )
        assert total == pytest.approx(100.0)

    def test_consensus_boundaries_are_in_range(self):
        # 70 and 180 mg/dL are inside the 70–180 target range; 54 is level-1 low
        # and 250 is level-1 high. 70/18 rounds below 3.9 mmol/L, which is the
        # trap the mg/dL thresholds avoid.
        m = compute_clinical_cgm_metrics(_series([70, 180, 54, 250]), period_days=1)
        assert m["tir"] == pytest.approx(50.0)
        assert m["tbr_level1"] == pytest.approx(25.0)
        assert m["tbr_level2"] == pytest.approx(0.0)
        assert m["tar_level1"] == pytest.approx(25.0)
        assert m["tar_level2"] == pytest.approx(0.0)

    def test_cv_and_gmi(self):
        # Mean 150 mg/dL, population SD 50 -> CV 33.3%; GMI = 3.31 + 0.02392*150.
        m = compute_clinical_cgm_metrics(_series([100, 200] * 10), period_days=1)
        assert m["mean_mmol_l"] == pytest.approx(150 / 18.0)
        assert m["cv_percent"] == pytest.approx(100 * 50 / 150)
        assert m["gmi_percent"] == pytest.approx(3.31 + 0.02392 * 150)

    def test_coverage_uses_the_sensor_interval(self):
        # Half a day of 5-minute readings over a one-day period -> 50% coverage.
        m = compute_clinical_cgm_metrics(_series([120] * 144), period_days=1)
        assert m["interval_minutes"] == pytest.approx(5.0)
        assert m["coverage_percent"] == pytest.approx(50.0)

    def test_empty_series(self):
        assert compute_clinical_cgm_metrics([], period_days=14) is None


class TestGetCgmClinicalMetricsTool:
    def test_reports_metrics_with_targets(self):
        _store([120] * 4032)  # 14 days of 5-minute readings, all in range
        out = get_cgm_clinical_metrics.invoke({"days": 14})
        assert "4032 readings" in out
        assert "Time in range 3.9–10.0 mmol/L: 100.0% (target >70%, meets target)" in out
        assert "Time below 3.0 mmol/L: 0.0%" in out
        assert "Data warning" not in out

    def test_flags_too_much_time_below_range(self):
        _store([60] * 10 + [120] * 90)
        out = get_cgm_clinical_metrics.invoke({"days": 14})
        assert "Time below 3.9 mmol/L (incl. <3.0): 10.0% (target <4%, ABOVE target)" in out

    def test_warns_when_data_is_insufficient(self):
        _store([120] * 288)  # one day of data in a 14-day window
        out = get_cgm_clinical_metrics.invoke({"days": 14})
        assert "Data warning" in out
        assert "less than 70%" in out

    def test_warns_when_period_is_short(self):
        _store([120] * 288)
        out = get_cgm_clinical_metrics.invoke({"days": 1})
        assert "shorter than the recommended 14 days" in out

    def test_uses_every_reading_beyond_5000(self):
        # 4 days of 1-minute readings = 5,760 rows, above get_readings' 5000 cap.
        _store([120] * 2880 + [60] * 2880, interval_minutes=1)
        out = get_cgm_clinical_metrics.invoke({"days": 4})
        assert "5760 readings" in out
        assert "Time below 3.9 mmol/L (incl. <3.0): 50.0%" in out

    def test_no_data(self):
        out = get_cgm_clinical_metrics.invoke({"days": 14})
        assert "No CGM readings found" in out

    def test_rejects_non_positive_days(self):
        assert "greater than 0" in get_cgm_clinical_metrics.invoke({"days": 0})
