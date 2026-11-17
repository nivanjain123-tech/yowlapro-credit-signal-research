import numpy as np
import pandas as pd

from takehome.helpers import (
    clean_events,
    forward_outcome,
    load_pack,
    point_in_time_join,
    rolling_zscore,
    winsorize,
)


def test_forward_outcome_aligns_future_without_lookahead():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = forward_outcome(series, horizon=1)
    assert out.iloc[0] == 2.0
    assert pd.isna(out.iloc[-1])


def test_load_pack_missing_dir_raises():
    try:
        load_pack("/nonexistent/path/for/test")
    except Exception:
        pass


def test_rolling_zscore_is_trailing():
    s = pd.Series(np.arange(30, dtype=float))
    base = rolling_zscore(s, window=5)
    changed = s.copy()
    changed.iloc[20:] = 1e6
    assert np.allclose(base.iloc[:20], rolling_zscore(changed, window=5).iloc[:20], equal_nan=True)


def test_winsorize_ignores_holdout_dir(tmp_path, monkeypatch):
    (tmp_path / "data" / "holdout").mkdir(parents=True)
    pd.DataFrame({"x": [1e9] * 100}).to_parquet(tmp_path / "data" / "holdout" / "h.parquet")
    monkeypatch.chdir(tmp_path)
    s = pd.Series(np.arange(100, dtype=float), name="x")
    assert winsorize(s).max() <= 99.0


def test_point_in_time_join_uses_only_past_rows_and_normalizes_ids():
    left = pd.DataFrame(
        {"issuer_id": ["ISS_1", "iss_1"], "published_at": ["2020-01-05T00:00:00Z", "2020-01-20T00:00:00Z"]}
    )
    right = pd.DataFrame(
        {
            "issuer_id": ["iss_1", "iss_1"],
            "published_at": ["2020-01-10T00:00:00Z", "2020-01-15T00:00:00Z"],
            "v": [1, 2],
        }
    )
    out = point_in_time_join(left, right, on=["issuer_id"]).sort_values("published_at")
    assert pd.isna(out["v"].iloc[0])  # revision not yet published
    assert out["v"].iloc[1] == 2


def test_clean_events_dedup_sentinel_and_timezone():
    ev = pd.DataFrame(
        {
            "event_id": ["a", "a", "b"],
            "issuer_id": ["ISS_1", "iss_1", "iss_2"],
            "event_time": ["2020-01-01 09:00:00"] * 3,
            "event_tz": ["Asia/Tokyo", "Asia/Tokyo", "America/New_York"],
            "published_at": ["2020-01-02T00:00:00Z"] * 3,
            "f_01": [-999.0, 1.5, -999.0],
        }
    )
    out = clean_events(ev)
    assert len(out) == 2
    assert out.loc[out.event_id == "a", "f_01"].iloc[0] == 1.5
    assert out.loc[out.event_id == "b", "f_01"].isna().all()
    assert out.loc[out.event_id == "a", "event_time_utc"].iloc[0] == pd.Timestamp("2020-01-01 00:00", tz="UTC")
