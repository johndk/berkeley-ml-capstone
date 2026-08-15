"""Causal same-airline departure-backlog features.

This module applies the established departure-backlog timing rules separately
within each reporting airline.  For a sample departure at cutoff ``T``, only
earlier-scheduled flights operated by the same airline can enter its trailing
window.  Delay outcomes contribute only after the corresponding flight has
actually pushed back.

The functions perform deterministic feature construction and validation only.
Learned imputation, scaling, encoding, and selection remain model-pipeline
responsibilities.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from feature_engineering_backlog import (
    DEFAULT_BACKLOG_WINDOW_MINUTES,
    add_departure_backlog_features,
    backlog_feature_names,
    validate_departure_backlog_features,
)


DEFAULT_AIRLINE_COLUMN = "Reporting_Airline"


def _require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise KeyError(f"{context} is missing required columns: {missing}")


def airline_backlog_feature_names(
    window_minutes: int = DEFAULT_BACKLOG_WINDOW_MINUTES,
) -> list[str]:
    """Return ordered feature names for a same-airline trailing window."""

    base_names = backlog_feature_names(window_minutes)
    renamed = [name.replace("BACKLOG_", "AIRLINE_BACKLOG_", 1) for name in base_names]
    renamed.append(f"AIRLINE_BACKLOG_W{int(window_minutes)}_PENDING_SHARE")
    return renamed


DEFAULT_AIRLINE_BACKLOG_FEATURES = airline_backlog_feature_names()


def _normalized_airlines(source: pd.DataFrame, airline_column: str) -> pd.Series:
    _require_columns(source, [airline_column], "same-airline backlog input")
    airlines = source[airline_column].astype("string").str.strip().str.upper()
    if airlines.isna().any() or airlines.eq("").any():
        raise ValueError(f"{airline_column} must contain complete airline codes")
    return airlines


def add_airline_departure_backlog_features(
    source: pd.DataFrame,
    *,
    window_minutes: int = DEFAULT_BACKLOG_WINDOW_MINUTES,
    airline_column: str = DEFAULT_AIRLINE_COLUMN,
) -> pd.DataFrame:
    """Append causal backlog summaries calculated within each airline.

    The existing airport-wide helper is applied independently to the row
    positions belonging to each normalized airline code.  This preserves its
    strict ``[T - window, T)`` scheduled cohort, strict gate-out completion
    test, and missing-value semantics while preventing other airlines from
    contributing to a sample's operational state.

    ``PENDING_SHARE`` equals pending divided by scheduled cohort count.  It is
    missing when the same-airline scheduled cohort is empty.
    """

    airlines = _normalized_airlines(source, airline_column)
    base_names = backlog_feature_names(window_minutes)
    output_names = airline_backlog_feature_names(window_minutes)
    renamed_names = output_names[: len(base_names)]
    pending_share_name = output_names[-1]

    if set(output_names) & set(source.columns):
        raise ValueError("Input already contains same-airline backlog features")

    blocks: list[pd.DataFrame] = []
    position_series = pd.Series(np.arange(len(source), dtype=np.int64))
    for airline in pd.unique(airlines):
        positions = position_series.loc[airlines.eq(airline).to_numpy()].to_numpy()
        airline_source = source.iloc[positions].reset_index(drop=True)
        airline_features = add_departure_backlog_features(
            airline_source,
            window_minutes=window_minutes,
        )
        block = airline_features[base_names].rename(
            columns=dict(zip(base_names, renamed_names, strict=True))
        )
        block["_SOURCE_POSITION"] = positions
        blocks.append(block)

    ordered = (
        pd.concat(blocks, ignore_index=True)
        .sort_values("_SOURCE_POSITION", kind="stable")
        .reset_index(drop=True)
    )
    expected_positions = np.arange(len(source), dtype=np.int64)
    if not np.array_equal(ordered["_SOURCE_POSITION"].to_numpy(), expected_positions):
        raise ValueError("Same-airline backlog construction changed row identity")

    scheduled = pd.to_numeric(ordered[renamed_names[0]], errors="raise")
    pending = pd.to_numeric(ordered[renamed_names[2]], errors="raise")
    pending_share = np.full(len(source), np.nan, dtype=float)
    has_scheduled_history = scheduled.gt(0).to_numpy()
    pending_share[has_scheduled_history] = (
        pending[has_scheduled_history] / scheduled[has_scheduled_history]
    )

    result = source.copy()
    for name in renamed_names:
        result[name] = pd.Series(ordered[name].array, index=result.index)
    result[pending_share_name] = pending_share
    return result


def validate_airline_departure_backlog_features(
    df: pd.DataFrame,
    *,
    window_minutes: int = DEFAULT_BACKLOG_WINDOW_MINUTES,
    airline_column: str = DEFAULT_AIRLINE_COLUMN,
) -> pd.DataFrame:
    """Validate same-airline identities, ranges, and missing semantics."""

    _normalized_airlines(df, airline_column)
    base_names = backlog_feature_names(window_minutes)
    output_names = airline_backlog_feature_names(window_minutes)
    renamed_names = output_names[: len(base_names)]
    pending_share_name = output_names[-1]
    _require_columns(df, output_names, "same-airline backlog output")

    base_view = df[renamed_names].rename(
        columns=dict(zip(renamed_names, base_names, strict=True))
    )
    base_validation = validate_departure_backlog_features(
        base_view,
        window_minutes=window_minutes,
    )
    base_validation.index = renamed_names

    scheduled = pd.to_numeric(df[renamed_names[0]], errors="coerce")
    pending = pd.to_numeric(df[renamed_names[2]], errors="coerce")
    pending_share = pd.to_numeric(df[pending_share_name], errors="coerce")
    has_scheduled_history = scheduled.gt(0)
    if pending_share.loc[has_scheduled_history].isna().any():
        raise ValueError("Airline pending share is missing despite scheduled history")
    if pending_share.loc[~has_scheduled_history].notna().any():
        raise ValueError("Airline pending share must be missing without scheduled history")
    if not pending_share.loc[has_scheduled_history].between(0, 1).all():
        raise ValueError("Airline pending share must be between zero and one")
    expected = pending.loc[has_scheduled_history] / scheduled.loc[has_scheduled_history]
    if not np.allclose(
        pending_share.loc[has_scheduled_history].to_numpy(),
        expected.to_numpy(),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Airline pending share does not equal pending / scheduled")

    share_validation = pd.DataFrame(
        {
            "dtype": [str(df[pending_share_name].dtype)],
            "missing_count": [int(df[pending_share_name].isna().sum())],
            "missing_percent": [float(df[pending_share_name].isna().mean() * 100)],
            "minimum": [pending_share.min()],
            "maximum": [pending_share.max()],
        },
        index=[pending_share_name],
    )
    return pd.concat([base_validation, share_validation])
