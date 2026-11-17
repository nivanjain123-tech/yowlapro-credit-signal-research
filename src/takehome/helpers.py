"""Inherited research helpers (review before use).

Canary: KIT_CANARY_5D7606E79BA2
Leak fingerprint: 5e44b218449998b32e80cee39c3568e9
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _apply_feature_aliases(df: pd.DataFrame) -> pd.DataFrame:
    import json
    import os

    alias_path = os.environ.get("TAKEHOME_FEATURE_ALIAS_FILE")
    if not alias_path:
        return df
    path = Path(alias_path)
    if not path.is_file():
        return df
    aliases = json.loads(path.read_text(encoding="utf-8"))
    rename = {v: k for k, v in aliases.items() if v in df.columns}
    if rename:
        df = df.rename(columns=rename)
    return df


SENTINEL = -999


def normalize_issuer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase/strip issuer_id so events, outcomes and issuers join cleanly."""
    if "issuer_id" in df.columns:
        df = df.copy()
        df["issuer_id"] = df["issuer_id"].astype("string").str.strip().str.lower()
    return df


def load_pack(data_dir: str | Path = "data") -> dict[str, pd.DataFrame]:
    import os

    env_dir = os.environ.get("TAKEHOME_DATA_DIR")
    root = Path(env_dir if env_dir else data_dir)
    events = pd.read_parquet(root / "events.parquet")
    events = _apply_feature_aliases(events)
    pack = {
        "issuers": pd.read_parquet(root / "issuers.parquet"),
        "events": events,
        "outcomes": pd.read_parquet(root / "outcomes.parquet"),
        "exits": pd.read_parquet(root / "exits.parquet"),
    }
    return {name: normalize_issuer_id(df) for name, df in pack.items()}


def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    """Dedup on event_id, turn the -999 sentinel into NaN, add UTC event time.

    Among duplicate event_ids the row with the fewest sentinel/missing features
    is kept (ties: first occurrence), so a sentinel copy never beats a real value.
    """
    feats = [c for c in events.columns if c.startswith("f_")]
    out = normalize_issuer_id(events)
    out[feats] = out[feats].where(out[feats] != SENTINEL)
    out = out.assign(_n_missing=out[feats].isna().sum(axis=1))
    out = (
        out.sort_values("_n_missing", kind="stable")
        .drop_duplicates("event_id", keep="first")
        .drop(columns="_n_missing")
        .sort_index()
    )
    out["published_at"] = pd.to_datetime(out["published_at"], utc=True)
    local = pd.to_datetime(out["event_time"])
    out["event_time_utc"] = [
        t.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")
        if pd.notna(t)
        else pd.NaT
        for t, tz in zip(local, out["event_tz"])
    ]
    out["event_time_utc"] = pd.to_datetime(out["event_time_utc"], utc=True)
    return out


def forward_outcome(series: pd.Series, horizon: int = 1) -> pd.Series:
    """Align outcome at t+horizon to row t (the value that happens *after* t)."""
    return series.shift(-horizon)


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Trailing z-score: each point uses only itself and earlier observations."""
    roll = series.rolling(window, min_periods=2)
    return (series - roll.mean()) / roll.std()


def winsorize(series: pd.Series, low: float = 0.01, high: float = 0.99) -> pd.Series:
    """Clip to quantiles of this series only (never pool in holdout data)."""
    q_low, q_high = series.quantile([low, high])
    return series.clip(lower=q_low, upper=q_high)


def point_in_time_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: list[str],
    how: str = "left",
    time_col: str = "published_at",
) -> pd.DataFrame:
    """Join each left row to the latest right row with time_col <= left time_col.

    Keys in `on` are matched exactly (issuer_id case-insensitively); right rows
    published after the left row are never used. If time_col is absent from
    either frame this is a plain merge.
    """
    if how not in ("left", "inner"):
        raise ValueError("how must be 'left' or 'inner'")
    left_n, right_n = normalize_issuer_id(left), normalize_issuer_id(right)
    if time_col not in left_n.columns or time_col not in right_n.columns:
        return left_n.merge(right_n, on=on, how=how)
    left_n[time_col] = pd.to_datetime(left_n[time_col], utc=True).astype("datetime64[ns, UTC]")
    right_n[time_col] = pd.to_datetime(right_n[time_col], utc=True).astype("datetime64[ns, UTC]")
    right_n["_matched"] = True
    merged = pd.merge_asof(
        left_n.sort_values(time_col),
        right_n.sort_values(time_col),
        on=time_col,
        by=on,
        direction="backward",
        suffixes=("", "_right"),
    )
    if how == "inner":
        merged = merged[merged["_matched"].notna()]
    return merged.drop(columns="_matched").sort_index()
