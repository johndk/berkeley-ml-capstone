"""Causal departure-backlog features for the capstone flight datasets.

The functions in this module reconstruct the information that would have been
available immediately before a flight's scheduled departure timestamp.  They
do not perform learned preprocessing; imputation, scaling, encoding, and
feature selection remain responsibilities of the model pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_BACKLOG_WINDOW_MINUTES = 30


def _validate_window_minutes(window_minutes: int) -> int:
    """Return a validated positive, whole-number window length."""

    if isinstance(window_minutes, bool):
        raise TypeError("window_minutes must be a positive integer")
    try:
        numeric = float(window_minutes)
    except (TypeError, ValueError) as exc:
        raise TypeError("window_minutes must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        raise ValueError("window_minutes must be a positive integer")
    return int(numeric)


def backlog_feature_names(
    window_minutes: int = DEFAULT_BACKLOG_WINDOW_MINUTES,
) -> list[str]:
    """Return the ordered feature names for a trailing backlog window."""

    window_minutes = _validate_window_minutes(window_minutes)
    prefix = f"BACKLOG_W{window_minutes}"
    return [
        f"{prefix}_SCHEDULED_COUNT",
        f"{prefix}_COMPLETED_COUNT",
        f"{prefix}_PENDING_COUNT",
        f"{prefix}_DELAYED_DEPARTURE_COUNT",
        f"{prefix}_DELAY_RATE",
        f"{prefix}_MEAN_DEP_DELAY",
        f"{prefix}_MEAN_DEP_DELAY_MINUTES",
        f"{prefix}_TOTAL_DEP_DELAY_MINUTES",
    ]


DEFAULT_BACKLOG_FEATURES = backlog_feature_names()


def _require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise KeyError(f"{context} is missing required columns: {missing}")


def _numeric_complete(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        raise ValueError(
            f"{column} must be complete before departure-backlog features are built"
        )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{column} contains a non-finite value")
    return values


def actual_departure_timestamps(source: pd.DataFrame) -> pd.Series:
    """Reconstruct gate-out timestamps from schedule time and signed delay.

    BTS ``DepDelay`` is actual gate departure minus scheduled gate departure in
    minutes.  Adding it to ``DATE`` handles midnight rollovers without trying
    to infer a date from the HHMM-formatted ``DepTime`` column.
    """

    _require_columns(source, ["DATE", "DepDelay"], "departure backlog input")
    scheduled = pd.to_datetime(source["DATE"], errors="coerce")
    if scheduled.isna().any():
        raise ValueError("DATE must contain valid, complete scheduled timestamps")
    departure_delay = _numeric_complete(source, "DepDelay")
    return scheduled + pd.to_timedelta(departure_delay, unit="m")


def add_departure_backlog_features(
    source: pd.DataFrame,
    *,
    window_minutes: int = DEFAULT_BACKLOG_WINDOW_MINUTES,
) -> pd.DataFrame:
    """Add causal recent-performance and pending-backlog features.

    For a sample flight with scheduled cutoff ``T``, the cohort contains other
    flights scheduled in ``[T - window, T)``.  Flights scheduled at ``T`` are
    deliberately excluded, so simultaneous samples see the same pre-cutoff
    state and a sample can never contribute its own outcome.

    ``COMPLETED_COUNT`` and the delay aggregates use only cohort flights whose
    reconstructed gate-out timestamp is strictly before ``T``.  The remaining
    cohort flights form ``PENDING_COUNT``.  This reconstructs information an
    operational gate-out feed could have supplied at ``T`` while never using a
    completed flight's delay before that flight actually pushed back.
    """

    window_minutes = _validate_window_minutes(window_minutes)
    required = ["DATE", "DepDelay", "DepDelayMinutes", "DepDel15"]
    _require_columns(source, required, "departure backlog input")

    scheduled = pd.to_datetime(source["DATE"], errors="coerce")
    if scheduled.isna().any():
        raise ValueError("DATE must contain valid, complete scheduled timestamps")

    departure_delay = _numeric_complete(source, "DepDelay")
    delay_minutes = _numeric_complete(source, "DepDelayMinutes")
    delayed = _numeric_complete(source, "DepDel15")
    if (delay_minutes < 0).any():
        raise ValueError("DepDelayMinutes must be nonnegative")
    if not delayed.isin([0, 1]).all():
        raise ValueError("DepDel15 must be binary")

    actual_departure = scheduled + pd.to_timedelta(departure_delay, unit="m")
    scheduled_ns = scheduled.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    actual_ns = actual_departure.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    delay_values = departure_delay.to_numpy(dtype=float)
    delay_minute_values = delay_minutes.to_numpy(dtype=float)
    delayed_values = delayed.to_numpy(dtype=np.int8)

    # Work in scheduled-time order while preserving the source row order in the
    # returned dataframe.  Stable ordering makes duplicate schedule timestamps
    # deterministic, although the strict upper boundary excludes all of them.
    order = np.argsort(scheduled_ns, kind="stable")
    ordered_scheduled = scheduled_ns[order]
    ordered_actual = actual_ns[order]
    ordered_delay = delay_values[order]
    ordered_delay_minutes = delay_minute_values[order]
    ordered_delayed = delayed_values[order]

    window_ns = pd.Timedelta(minutes=window_minutes).value
    left_edges = np.searchsorted(
        ordered_scheduled, scheduled_ns - window_ns, side="left"
    )
    right_edges = np.searchsorted(ordered_scheduled, scheduled_ns, side="left")

    row_count = len(source)
    scheduled_count = right_edges - left_edges
    completed_count = np.zeros(row_count, dtype=np.int32)
    delayed_count = np.zeros(row_count, dtype=np.int32)
    sum_delay = np.zeros(row_count, dtype=float)
    sum_delay_minutes = np.zeros(row_count, dtype=float)

    for row_number, (left, right, cutoff_ns) in enumerate(
        zip(left_edges, right_edges, scheduled_ns, strict=True)
    ):
        if left == right:
            continue
        completed = ordered_actual[left:right] < cutoff_ns
        if not completed.any():
            continue
        completed_count[row_number] = int(completed.sum())
        delayed_count[row_number] = int(ordered_delayed[left:right][completed].sum())
        sum_delay[row_number] = ordered_delay[left:right][completed].sum()
        sum_delay_minutes[row_number] = ordered_delay_minutes[left:right][
            completed
        ].sum()

    pending_count = scheduled_count - completed_count
    has_completed_history = completed_count > 0
    delay_rate = np.full(row_count, np.nan, dtype=float)
    mean_delay = np.full(row_count, np.nan, dtype=float)
    mean_delay_minutes = np.full(row_count, np.nan, dtype=float)
    delay_rate[has_completed_history] = (
        delayed_count[has_completed_history] / completed_count[has_completed_history]
    )
    mean_delay[has_completed_history] = (
        sum_delay[has_completed_history] / completed_count[has_completed_history]
    )
    mean_delay_minutes[has_completed_history] = (
        sum_delay_minutes[has_completed_history]
        / completed_count[has_completed_history]
    )

    names = backlog_feature_names(window_minutes)
    result = source.copy()
    result[names[0]] = pd.Series(scheduled_count, index=result.index, dtype="Int32")
    result[names[1]] = pd.Series(completed_count, index=result.index, dtype="Int32")
    result[names[2]] = pd.Series(pending_count, index=result.index, dtype="Int32")
    result[names[3]] = pd.Series(delayed_count, index=result.index, dtype="Int32")
    result[names[4]] = delay_rate
    result[names[5]] = mean_delay
    result[names[6]] = mean_delay_minutes
    result[names[7]] = sum_delay_minutes
    return result


def validate_departure_backlog_features(
    df: pd.DataFrame,
    *,
    window_minutes: int = DEFAULT_BACKLOG_WINDOW_MINUTES,
) -> pd.DataFrame:
    """Validate backlog identities, ranges, and missing-value semantics."""

    names = backlog_feature_names(window_minutes)
    _require_columns(df, names, "departure backlog output")
    values = df[names].apply(pd.to_numeric, errors="coerce")

    scheduled, completed, pending, delayed_count = (
        values[name] for name in names[:4]
    )
    if values[names[:4]].isna().any().any():
        raise ValueError("Backlog count features must be complete")
    if (values[names[:4]] < 0).any().any():
        raise ValueError("Backlog count features must be nonnegative")
    if not scheduled.eq(completed + pending).all():
        raise ValueError("Backlog scheduled count must equal completed plus pending")
    if (delayed_count > completed).any():
        raise ValueError("Backlog delayed count cannot exceed completed count")

    has_history = completed.gt(0)
    aggregate_names = names[4:7]
    if values.loc[has_history, aggregate_names].isna().any().any():
        raise ValueError("Backlog rates and means are missing despite completed history")
    if values.loc[~has_history, aggregate_names].notna().any().any():
        raise ValueError("Backlog rates and means must be missing without completed history")
    if not values.loc[has_history, names[4]].between(0, 1).all():
        raise ValueError("Backlog delay rate must be between zero and one")
    if (values[names[6]].dropna() < 0).any() or (values[names[7]] < 0).any():
        raise ValueError("Nonnegative backlog delay measures contain a negative value")
    numeric_array = values.to_numpy(dtype=float, na_value=np.nan)
    finite_values = numeric_array[~np.isnan(numeric_array)]
    if not np.isfinite(finite_values).all():
        raise ValueError("Backlog features contain a non-finite value")

    return pd.DataFrame(
        {
            "dtype": df[names].dtypes.astype(str),
            "missing_count": df[names].isna().sum(),
            "missing_percent": df[names].isna().mean().mul(100),
            "minimum": values.min(),
            "maximum": values.max(),
        }
    )
