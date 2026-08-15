"""Causal aircraft-rotation features for JFK-style departure models.

The target departure table supplies the flights to predict.  A separate BTS
table of flights arriving at the target airport supplies candidate preceding
legs.  Matching uses only scheduled flight-leg order and aircraft tail number.
An inbound leg's realized arrival delay is exposed only when its arrival had
already occurred by the target flight's scheduled-departure cutoff.

This module performs deterministic feature construction only.  Learned
imputation, encoding, scaling, feature selection, and model fitting belong in
the experiment pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


# Static mappings keep annual reconstruction deterministic and avoid a runtime
# geocoding dependency.  The mapping covers the project's current airport set.
AIRPORT_TIME_ZONES = {
    "ABQ": "America/Denver",
    "ACK": "America/New_York",
    "AGS": "America/New_York",
    "ANC": "America/Anchorage",
    "ATL": "America/New_York",
    "AUS": "America/Chicago",
    "BNA": "America/Chicago",
    "BOS": "America/New_York",
    "BGR": "America/New_York",
    "BQN": "America/Puerto_Rico",
    "BUF": "America/New_York",
    "BTV": "America/New_York",
    "BUR": "America/Los_Angeles",
    "BWI": "America/New_York",
    "BZN": "America/Denver",
    "CHS": "America/New_York",
    "CLE": "America/New_York",
    "CLT": "America/New_York",
    "CMH": "America/New_York",
    "CVG": "America/New_York",
    "DCA": "America/New_York",
    "DAB": "America/New_York",
    "DEN": "America/Denver",
    "DFW": "America/Chicago",
    "DTW": "America/Detroit",
    "EGE": "America/Denver",
    "FLL": "America/New_York",
    "HNL": "Pacific/Honolulu",
    "HOU": "America/Chicago",
    "HYA": "America/New_York",
    "IAD": "America/New_York",
    "IAH": "America/Chicago",
    "IND": "America/Indiana/Indianapolis",
    "ITH": "America/New_York",
    "JAC": "America/Denver",
    "JAX": "America/New_York",
    "JFK": "America/New_York",
    "LAS": "America/Los_Angeles",
    "LAX": "America/Los_Angeles",
    "LGA": "America/New_York",
    "LGB": "America/Los_Angeles",
    "MCI": "America/Chicago",
    "MCO": "America/New_York",
    "MIA": "America/New_York",
    "MKE": "America/Chicago",
    "MSP": "America/Chicago",
    "MSY": "America/Chicago",
    "MVY": "America/New_York",
    "OAK": "America/Los_Angeles",
    "ONT": "America/Los_Angeles",
    "ORD": "America/Chicago",
    "ORF": "America/New_York",
    "ORH": "America/New_York",
    "PBI": "America/New_York",
    "PDX": "America/Los_Angeles",
    "PHL": "America/New_York",
    "PHX": "America/Phoenix",
    "PIT": "America/New_York",
    "PSP": "America/Los_Angeles",
    "PSE": "America/Puerto_Rico",
    "PWM": "America/New_York",
    "RDU": "America/New_York",
    "RIC": "America/New_York",
    "RNO": "America/Los_Angeles",
    "ROC": "America/New_York",
    "RSW": "America/New_York",
    "SAN": "America/Los_Angeles",
    "SAT": "America/Chicago",
    "SAV": "America/New_York",
    "SDF": "America/New_York",
    "SEA": "America/Los_Angeles",
    "SFO": "America/Los_Angeles",
    "SJC": "America/Los_Angeles",
    "SJU": "America/Puerto_Rico",
    "SLC": "America/Denver",
    "SMF": "America/Los_Angeles",
    "SNA": "America/Los_Angeles",
    "SRQ": "America/New_York",
    "STT": "America/St_Thomas",
    "SYR": "America/New_York",
    "TPA": "America/New_York",
}


ROTATION_FEATURES = [
    "ROTATION_STATUS",
    "ROTATION_MATCH_FOUND",
    "ROTATION_INBOUND_ORIGIN",
    "ROTATION_SCHEDULED_TURN_MINUTES",
    "ROTATION_INBOUND_ARRIVED_BY_CUTOFF",
    "ROTATION_INBOUND_NOT_ARRIVED_BY_CUTOFF",
    "ROTATION_INBOUND_OVERDUE_MINUTES",
    "ROTATION_LOG_INBOUND_OVERDUE_MINUTES",
    "ROTATION_ACTUAL_TURN_MINUTES",
    "ROTATION_LOG_ACTUAL_TURN_MINUTES",
    "ROTATION_INBOUND_ARR_DELAY",
    "ROTATION_INBOUND_DELAYED_15",
    "ROTATION_LOG_SCHEDULED_TURN_MINUTES",
]

ROTATION_AUDIT_COLUMNS = [
    "ROTATION_TARGET_CUTOFF_UTC",
    "ROTATION_PRIOR_REPORTING_AIRLINE",
    "ROTATION_PRIOR_FLIGHT_NUMBER",
    "ROTATION_PRIOR_FLIGHT_DATE",
    "ROTATION_PRIOR_SCHEDULED_DEPARTURE_UTC",
    "ROTATION_PRIOR_SCHEDULED_ARRIVAL_UTC",
    "ROTATION_PRIOR_ACTUAL_ARRIVAL_UTC",
    "ROTATION_SCHEDULE_RECONSTRUCTION_ERROR_MINUTES",
]

ROTATION_OUTPUT_COLUMNS = ROTATION_FEATURES + ROTATION_AUDIT_COLUMNS


def _require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise KeyError(f"{context} is missing required columns: {missing}")


def _normalize_airport(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.upper()


def _normalize_tail(values: pd.Series) -> pd.Series:
    tails = values.astype("string").str.strip().str.upper()
    return tails.mask(tails.eq(""))


def _complete_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{column} must be complete for rotation reconstruction")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{column} contains a non-finite value")
    return values


def _localize_by_airport(
    local_timestamps: pd.Series,
    airports: pd.Series,
    *,
    context: str,
) -> pd.Series:
    """Convert airport-local naive timestamps to naive UTC timestamps."""

    local = pd.to_datetime(local_timestamps, errors="coerce")
    airport = _normalize_airport(airports)
    if local.isna().any():
        raise ValueError(f"{context} contains an invalid local timestamp")
    missing_zones = sorted(set(airport.dropna().unique()) - set(AIRPORT_TIME_ZONES))
    if missing_zones:
        raise KeyError(f"No time-zone mapping for {context} airports: {missing_zones}")

    result = pd.Series(pd.NaT, index=local.index, dtype="datetime64[ns]")
    for airport_code, row_index in airport.groupby(airport, sort=False).groups.items():
        localized = (
            pd.DatetimeIndex(local.loc[row_index])
            .tz_localize(
                AIRPORT_TIME_ZONES[str(airport_code)],
                ambiguous=True,
                nonexistent="shift_forward",
            )
            .tz_convert("UTC")
            .tz_localize(None)
        )
        result.loc[row_index] = localized.to_numpy(dtype="datetime64[ns]")
    if result.isna().any():
        raise ValueError(f"{context} could not be converted completely to UTC")
    return result


def _hhmm_parts(values: pd.Series, column: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{column} must be complete and HHMM-formatted")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{column} contains a non-integer clock value")
    whole = numeric.astype("int64")
    is_2400 = whole.eq(2400)
    hour = whole // 100
    minute = whole % 100
    valid = ((hour.between(0, 23)) & minute.between(0, 59)) | is_2400
    if not valid.all():
        bad = whole.loc[~valid].head().tolist()
        raise ValueError(f"{column} contains invalid HHMM values: {bad}")
    return hour.mask(is_2400, 0), minute, is_2400.astype("int8")


def scheduled_departure_timestamps(source: pd.DataFrame) -> pd.Series:
    """Construct airport-local scheduled-departure timestamps from raw BTS fields."""

    _require_columns(source, ["FlightDate", "CRSDepTime"], "BTS schedule input")
    flight_date = pd.to_datetime(source["FlightDate"], errors="coerce").dt.normalize()
    if flight_date.isna().any():
        raise ValueError("FlightDate must be complete for schedule reconstruction")
    hour, minute, is_2400 = _hhmm_parts(source["CRSDepTime"], "CRSDepTime")
    return (
        flight_date
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
        + pd.to_timedelta(is_2400, unit="D")
    )


def reconstruct_inbound_schedule(
    inbound: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct UTC scheduled departure and arrival timestamps.

    BTS ``FlightDate`` is the local date at the origin, while ``CRSArrTime`` is
    expressed in destination-local time.  Candidate destination dates from one
    day before through two days after ``FlightDate`` are evaluated, and the one
    whose UTC duration is closest to ``CRSElapsedTime`` is retained.  A small
    residual can occur on daylight-saving transition schedules or in anomalous
    source rows, so the residual is returned for auditing.
    """

    required = [
        "FlightDate",
        "Origin",
        "Dest",
        "CRSArrTime",
        "CRSElapsedTime",
        "DATE",
    ]
    _require_columns(inbound, required, "inbound rotation input")

    origin = _normalize_airport(inbound["Origin"])
    destination = _normalize_airport(inbound["Dest"])
    scheduled_departure_utc = _localize_by_airport(
        inbound["DATE"], origin, context="inbound scheduled departures"
    )

    flight_date = pd.to_datetime(inbound["FlightDate"], errors="coerce").dt.normalize()
    if flight_date.isna().any():
        raise ValueError("FlightDate must be complete for inbound reconstruction")
    hour, minute, is_2400 = _hhmm_parts(inbound["CRSArrTime"], "CRSArrTime")
    base_local_arrival = (
        flight_date
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
        + pd.to_timedelta(is_2400, unit="D")
    )
    scheduled_elapsed = _complete_numeric(inbound, "CRSElapsedTime")
    if (scheduled_elapsed <= 0).any():
        raise ValueError("CRSElapsedTime must be positive")

    candidate_offsets = np.array([-1, 0, 1, 2], dtype=np.int8)
    candidate_utc: list[pd.Series] = []
    candidate_errors: list[np.ndarray] = []
    for day_offset in candidate_offsets:
        utc = _localize_by_airport(
            base_local_arrival + pd.Timedelta(days=int(day_offset)),
            destination,
            context="inbound scheduled arrivals",
        )
        elapsed = (utc - scheduled_departure_utc).dt.total_seconds().div(60)
        candidate_utc.append(utc)
        candidate_errors.append(
            np.abs(elapsed.to_numpy(dtype=float) - scheduled_elapsed.to_numpy(dtype=float))
        )

    error_matrix = np.column_stack(candidate_errors)
    best_candidate = error_matrix.argmin(axis=1)
    row_number = np.arange(len(inbound))
    scheduled_arrival_values = np.column_stack(
        [candidate.to_numpy(dtype="datetime64[ns]") for candidate in candidate_utc]
    )[row_number, best_candidate]

    return pd.DataFrame(
        {
            "scheduled_departure_utc": scheduled_departure_utc.to_numpy(),
            "scheduled_arrival_utc": scheduled_arrival_values,
            "schedule_error_minutes": error_matrix[row_number, best_candidate],
            "arrival_day_offset": candidate_offsets[best_candidate],
        },
        index=inbound.index,
    )


def _preceding_leg_matches(
    target_tails: pd.Series,
    target_cutoff_utc: pd.Series,
    inbound_tails: pd.Series,
    inbound_arrival_utc: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the immediately preceding known scheduled leg for each target.

    JFK events are ordered by scheduled arrival or departure in UTC.  A
    preceding inbound leg is a rotation match.  A preceding JFK departure blocks an
    older inbound match, which prevents a stale arrival from being assigned
    after the aircraft has already left JFK once.
    """

    target_count = len(target_tails)
    prior_type = np.full(target_count, "NO_PRIOR_EVENT", dtype=object)
    prior_inbound_position = np.full(target_count, -1, dtype=np.int64)

    inbound_events = pd.DataFrame(
        {
            "tail": inbound_tails.to_numpy(),
            "event_time": inbound_arrival_utc.to_numpy(),
            "event_type": "INBOUND",
            "position": np.arange(len(inbound_tails), dtype=np.int64),
        }
    ).dropna(subset=["tail", "event_time"])
    target_events = pd.DataFrame(
        {
            "tail": target_tails.to_numpy(),
            "event_time": target_cutoff_utc.to_numpy(),
            "event_type": "DEPARTURE",
            "position": np.arange(target_count, dtype=np.int64),
        }
    ).dropna(subset=["tail", "event_time"])
    events = pd.concat([inbound_events, target_events], ignore_index=True)
    events["priority"] = events["event_type"].map({"INBOUND": 0, "DEPARTURE": 1})
    events.sort_values(
        ["tail", "event_time", "priority", "position"],
        kind="stable",
        inplace=True,
    )

    # Collapse simultaneous events to the state seen by later timestamps.  A
    # departure sorts last and therefore blocks an inbound at the same instant.
    history = events.drop_duplicates(["tail", "event_time"], keep="last")
    target_groups = target_events.groupby("tail", sort=False).groups
    for tail, history_rows in history.groupby("tail", sort=False):
        target_rows = target_groups.get(tail)
        if target_rows is None:
            continue
        event_times = history_rows["event_time"].to_numpy(dtype="datetime64[ns]")
        query_times = target_events.loc[target_rows, "event_time"].to_numpy(
            dtype="datetime64[ns]"
        )
        previous = np.searchsorted(event_times, query_times, side="left") - 1
        has_previous = previous >= 0
        if not has_previous.any():
            continue

        target_positions = target_events.loc[target_rows, "position"].to_numpy(
            dtype=np.int64
        )[has_previous]
        preceding = history_rows.iloc[previous[has_previous]]
        preceding_type = preceding["event_type"].to_numpy(dtype=object)
        prior_type[target_positions] = preceding_type
        inbound_previous = preceding_type == "INBOUND"
        if inbound_previous.any():
            prior_inbound_position[target_positions[inbound_previous]] = preceding.loc[
                inbound_previous, "position"
            ].to_numpy(dtype=np.int64)

    return prior_type, prior_inbound_position


def prepare_rotation_history(raw_bts: pd.DataFrame, *, airport: str) -> pd.DataFrame:
    """Return all raw BTS movements relevant to rotation order at an airport.

    This preparation is intended for read-only auditing.  It deliberately does
    not apply the project's working-airport cohort filter because an otherwise
    excluded intermediate leg can change the immediately preceding aircraft
    event at the target airport.
    """

    airport = str(airport).strip().upper()
    required = [
        "Year",
        "FlightDate",
        "Reporting_Airline",
        "Tail_Number",
        "Flight_Number_Reporting_Airline",
        "Origin",
        "Dest",
        "CRSDepTime",
        "CRSArrTime",
        "CRSElapsedTime",
        "ArrDelay",
        "ArrDel15",
        "Cancelled",
        "Diverted",
    ]
    _require_columns(raw_bts, required, "raw BTS rotation history")

    cancelled = pd.to_numeric(raw_bts["Cancelled"], errors="coerce")
    diverted = pd.to_numeric(raw_bts["Diverted"], errors="coerce")
    if cancelled.isna().any() or diverted.isna().any():
        raise ValueError("Cancelled and Diverted must be complete in raw BTS history")
    origin = _normalize_airport(raw_bts["Origin"])
    destination = _normalize_airport(raw_bts["Dest"])
    # A usable inbound event must have completed at the target airport.  An
    # outbound flight blocks an older inbound once it departed the target even
    # if it later diverted, so only cancellation excludes an outbound event.
    usable_inbound = destination.eq(airport) & cancelled.eq(0) & diverted.eq(0)
    usable_outbound = origin.eq(airport) & cancelled.eq(0)
    history = raw_bts.loc[usable_inbound | usable_outbound, required[:-2]].copy()
    history["Origin"] = origin.loc[history.index]
    history["Dest"] = destination.loc[history.index]
    history["Tail_Number"] = _normalize_tail(history["Tail_Number"])
    if history["Tail_Number"].isna().any():
        raise ValueError("Eligible completed rotation history contains a missing tail")
    history["DATE"] = scheduled_departure_timestamps(history)
    history["RAW_SOURCE_ROW"] = history.index.astype("int64")
    history.reset_index(drop=True, inplace=True)

    inbound = history["Dest"].eq(airport)
    _complete_numeric(history.loc[inbound], "ArrDelay")
    inbound_delayed = _complete_numeric(history.loc[inbound], "ArrDel15")
    if not inbound_delayed.isin([0, 1]).all():
        raise ValueError("Eligible inbound ArrDel15 values must be binary")
    return history


def _preceding_history_matches(
    target_tails: pd.Series,
    target_cutoff_utc: pd.Series,
    inbound_tails: pd.Series,
    inbound_arrival_utc: pd.Series,
    departure_tails: pd.Series,
    departure_utc: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Match targets against a separate complete airport-event history."""

    target_count = len(target_tails)
    prior_type = np.full(target_count, "NO_PRIOR_EVENT", dtype=object)
    prior_inbound_position = np.full(target_count, -1, dtype=np.int64)

    inbound_events = pd.DataFrame(
        {
            "tail": inbound_tails.to_numpy(),
            "event_time": inbound_arrival_utc.to_numpy(),
            "event_type": "INBOUND",
            "position": np.arange(len(inbound_tails), dtype=np.int64),
        }
    ).dropna(subset=["tail", "event_time"])
    departure_events = pd.DataFrame(
        {
            "tail": departure_tails.to_numpy(),
            "event_time": departure_utc.to_numpy(),
            "event_type": "DEPARTURE",
            "position": -1,
        }
    ).dropna(subset=["tail", "event_time"])
    events = pd.concat([inbound_events, departure_events], ignore_index=True)
    events["priority"] = events["event_type"].map({"INBOUND": 0, "DEPARTURE": 1})
    events.sort_values(
        ["tail", "event_time", "priority", "position"],
        kind="stable",
        inplace=True,
    )
    history = events.drop_duplicates(["tail", "event_time"], keep="last")

    query = pd.DataFrame(
        {
            "tail": target_tails.to_numpy(),
            "event_time": target_cutoff_utc.to_numpy(),
            "position": np.arange(target_count, dtype=np.int64),
        }
    ).dropna(subset=["tail", "event_time"])
    query_groups = query.groupby("tail", sort=False).groups
    for tail, history_rows in history.groupby("tail", sort=False):
        query_rows = query_groups.get(tail)
        if query_rows is None:
            continue
        event_times = history_rows["event_time"].to_numpy(dtype="datetime64[ns]")
        query_times = query.loc[query_rows, "event_time"].to_numpy(
            dtype="datetime64[ns]"
        )
        previous = np.searchsorted(event_times, query_times, side="left") - 1
        has_previous = previous >= 0
        if not has_previous.any():
            continue
        target_positions = query.loc[query_rows, "position"].to_numpy(
            dtype=np.int64
        )[has_previous]
        preceding = history_rows.iloc[previous[has_previous]]
        preceding_type = preceding["event_type"].to_numpy(dtype=object)
        prior_type[target_positions] = preceding_type
        inbound_previous = preceding_type == "INBOUND"
        if inbound_previous.any():
            prior_inbound_position[target_positions[inbound_previous]] = preceding.loc[
                inbound_previous, "position"
            ].to_numpy(dtype=np.int64)
    return prior_type, prior_inbound_position


def build_full_history_rotation_audit(
    target_departures: pd.DataFrame,
    full_history: pd.DataFrame,
    *,
    airport: str,
) -> pd.DataFrame:
    """Reconstruct target matches from a fuller completed-flight event history."""

    airport = str(airport).strip().upper()
    _require_columns(
        target_departures,
        ["DATE", "Origin", "Tail_Number"],
        "rotation audit targets",
    )
    _require_columns(
        full_history,
        [
            "FlightDate",
            "Reporting_Airline",
            "Tail_Number",
            "Flight_Number_Reporting_Airline",
            "Origin",
            "Dest",
            "CRSArrTime",
            "CRSElapsedTime",
            "ArrDelay",
            "ArrDel15",
            "DATE",
            "RAW_SOURCE_ROW",
        ],
        "prepared full rotation history",
    )

    target_origin = _normalize_airport(target_departures["Origin"])
    if not target_origin.eq(airport).all():
        raise ValueError(f"Rotation audit targets contain origins other than {airport}")
    inbound = full_history.loc[
        _normalize_airport(full_history["Dest"]).eq(airport)
    ].reset_index(drop=True)
    outbound = full_history.loc[
        _normalize_airport(full_history["Origin"]).eq(airport)
    ].reset_index(drop=True)
    if inbound.empty or outbound.empty:
        raise ValueError("Full rotation history must contain inbound and outbound events")

    target_cutoff_utc = _localize_by_airport(
        target_departures["DATE"], target_origin, context="rotation audit cutoffs"
    )
    inbound_schedule = reconstruct_inbound_schedule(inbound)
    outbound_departure_utc = _localize_by_airport(
        outbound["DATE"],
        outbound["Origin"],
        context="full-history outbound departures",
    )
    inbound_delay = _complete_numeric(inbound, "ArrDelay")
    inbound_delayed = _complete_numeric(inbound, "ArrDel15")
    actual_arrival_utc = inbound_schedule["scheduled_arrival_utc"] + pd.to_timedelta(
        inbound_delay, unit="m"
    )

    prior_type, prior_position = _preceding_history_matches(
        _normalize_tail(target_departures["Tail_Number"]),
        target_cutoff_utc,
        _normalize_tail(inbound["Tail_Number"]),
        inbound_schedule["scheduled_arrival_utc"],
        _normalize_tail(outbound["Tail_Number"]),
        outbound_departure_utc,
    )
    row_count = len(target_departures)
    match = prior_position >= 0
    matched_rows = np.flatnonzero(match)
    matched_inbound = prior_position[match]
    arrived = np.zeros(row_count, dtype=bool)
    arrived[match] = (
        actual_arrival_utc.iloc[matched_inbound].to_numpy(dtype="datetime64[ns]")
        <= target_cutoff_utc.iloc[matched_rows].to_numpy(dtype="datetime64[ns]")
    )
    not_arrived = match & ~arrived

    status = np.full(row_count, "NO_PRIOR_EVENT", dtype=object)
    status[prior_type == "DEPARTURE"] = "PREVIOUS_EVENT_DEPARTURE"
    status[match] = "NOT_ARRIVED"
    status[arrived] = "ARRIVED"
    status[_normalize_tail(target_departures["Tail_Number"]).isna().to_numpy()] = (
        "MISSING_TAIL"
    )

    scheduled_turn = np.full(row_count, np.nan, dtype=float)
    actual_turn = np.full(row_count, np.nan, dtype=float)
    observable_delay = np.full(row_count, np.nan, dtype=float)
    observable_delayed = np.full(row_count, np.nan, dtype=float)
    if match.any():
        scheduled_turn[match] = (
            target_cutoff_utc.iloc[matched_rows].to_numpy(dtype="datetime64[ns]")
            - inbound_schedule["scheduled_arrival_utc"].iloc[
                matched_inbound
            ].to_numpy(dtype="datetime64[ns]")
        ).astype("timedelta64[s]").astype(float) / 60
    if arrived.any():
        arrived_rows = np.flatnonzero(arrived)
        arrived_inbound = prior_position[arrived]
        actual_turn[arrived] = (
            target_cutoff_utc.iloc[arrived_rows].to_numpy(dtype="datetime64[ns]")
            - actual_arrival_utc.iloc[arrived_inbound].to_numpy(
                dtype="datetime64[ns]"
            )
        ).astype("timedelta64[s]").astype(float) / 60
        observable_delay[arrived] = inbound_delay.iloc[arrived_inbound].to_numpy(
            dtype=float
        )
        observable_delayed[arrived] = inbound_delayed.iloc[
            arrived_inbound
        ].to_numpy(dtype=float)

    result = pd.DataFrame(index=target_departures.index)
    result["FULL_ROTATION_STATUS"] = pd.Series(
        status, index=result.index, dtype="string"
    )
    result["FULL_ROTATION_MATCH_FOUND"] = pd.Series(
        match.astype(np.int8), index=result.index, dtype="Int8"
    )
    result["FULL_ROTATION_INBOUND_ARRIVED_BY_CUTOFF"] = pd.Series(
        arrived.astype(np.int8), index=result.index, dtype="Int8"
    )
    result["FULL_ROTATION_INBOUND_NOT_ARRIVED_BY_CUTOFF"] = pd.Series(
        not_arrived.astype(np.int8), index=result.index, dtype="Int8"
    )
    result["FULL_ROTATION_SCHEDULED_TURN_MINUTES"] = scheduled_turn
    result["FULL_ROTATION_ACTUAL_TURN_MINUTES"] = actual_turn
    result["FULL_ROTATION_INBOUND_ARR_DELAY"] = observable_delay
    result["FULL_ROTATION_INBOUND_DELAYED_15"] = observable_delayed
    result["FULL_ROTATION_INBOUND_ORIGIN"] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )
    result["FULL_ROTATION_PRIOR_REPORTING_AIRLINE"] = pd.NA
    result["FULL_ROTATION_PRIOR_FLIGHT_NUMBER"] = pd.NA
    result["FULL_ROTATION_PRIOR_FLIGHT_DATE"] = pd.NaT
    result["FULL_ROTATION_PRIOR_SCHEDULED_DEPARTURE_UTC"] = pd.NaT
    result["FULL_ROTATION_PRIOR_SCHEDULED_ARRIVAL_UTC"] = pd.NaT
    result["FULL_ROTATION_PRIOR_ACTUAL_ARRIVAL_UTC"] = pd.NaT
    result["FULL_ROTATION_PRIOR_RAW_SOURCE_ROW"] = pd.NA
    result["FULL_ROTATION_SCHEDULE_RECONSTRUCTION_ERROR_MINUTES"] = np.nan
    if match.any():
        target_index = result.index[matched_rows]
        values = {
            "FULL_ROTATION_INBOUND_ORIGIN": _normalize_airport(inbound["Origin"]),
            "FULL_ROTATION_PRIOR_REPORTING_AIRLINE": inbound["Reporting_Airline"],
            "FULL_ROTATION_PRIOR_FLIGHT_NUMBER": inbound[
                "Flight_Number_Reporting_Airline"
            ],
            "FULL_ROTATION_PRIOR_FLIGHT_DATE": inbound["FlightDate"],
            "FULL_ROTATION_PRIOR_SCHEDULED_DEPARTURE_UTC": inbound_schedule[
                "scheduled_departure_utc"
            ],
            "FULL_ROTATION_PRIOR_SCHEDULED_ARRIVAL_UTC": inbound_schedule[
                "scheduled_arrival_utc"
            ],
            "FULL_ROTATION_PRIOR_RAW_SOURCE_ROW": inbound["RAW_SOURCE_ROW"],
            "FULL_ROTATION_SCHEDULE_RECONSTRUCTION_ERROR_MINUTES": inbound_schedule[
                "schedule_error_minutes"
            ],
        }
        for column, source in values.items():
            result.loc[target_index, column] = source.iloc[matched_inbound].to_numpy()
    if arrived.any():
        arrived_rows = np.flatnonzero(arrived)
        result.loc[
            result.index[arrived_rows], "FULL_ROTATION_PRIOR_ACTUAL_ARRIVAL_UTC"
        ] = actual_arrival_utc.iloc[prior_position[arrived]].to_numpy()
    return result


def compare_rotation_histories(
    current_rotation: pd.DataFrame,
    full_rotation: pd.DataFrame,
) -> pd.DataFrame:
    """Return a row-level comparison of filtered and full-history matches."""

    current_required = [
        "ROTATION_STATUS",
        "ROTATION_MATCH_FOUND",
        "ROTATION_SCHEDULED_TURN_MINUTES",
        "ROTATION_PRIOR_REPORTING_AIRLINE",
        "ROTATION_PRIOR_FLIGHT_NUMBER",
        "ROTATION_PRIOR_FLIGHT_DATE",
        "ROTATION_PRIOR_SCHEDULED_ARRIVAL_UTC",
    ]
    full_required = [
        "FULL_ROTATION_STATUS",
        "FULL_ROTATION_MATCH_FOUND",
        "FULL_ROTATION_SCHEDULED_TURN_MINUTES",
        "FULL_ROTATION_PRIOR_REPORTING_AIRLINE",
        "FULL_ROTATION_PRIOR_FLIGHT_NUMBER",
        "FULL_ROTATION_PRIOR_FLIGHT_DATE",
        "FULL_ROTATION_PRIOR_SCHEDULED_ARRIVAL_UTC",
    ]
    _require_columns(current_rotation, current_required, "current rotation audit")
    _require_columns(full_rotation, full_required, "full rotation audit")
    if len(current_rotation) != len(full_rotation):
        raise ValueError("Rotation history comparison requires identical target rows")

    result = pd.DataFrame(index=current_rotation.index)
    result["CURRENT_STATUS"] = current_rotation["ROTATION_STATUS"].astype("string")
    result["FULL_STATUS"] = full_rotation["FULL_ROTATION_STATUS"].astype("string")
    result["CURRENT_MATCH_FOUND"] = pd.to_numeric(
        current_rotation["ROTATION_MATCH_FOUND"], errors="coerce"
    ).astype("Int8")
    result["FULL_MATCH_FOUND"] = pd.to_numeric(
        full_rotation["FULL_ROTATION_MATCH_FOUND"], errors="coerce"
    ).astype("Int8")
    result["CURRENT_SCHEDULED_TURN_MINUTES"] = pd.to_numeric(
        current_rotation["ROTATION_SCHEDULED_TURN_MINUTES"], errors="coerce"
    )
    result["FULL_SCHEDULED_TURN_MINUTES"] = pd.to_numeric(
        full_rotation["FULL_ROTATION_SCHEDULED_TURN_MINUTES"], errors="coerce"
    )
    result["TURN_CHANGE_MINUTES"] = (
        result["FULL_SCHEDULED_TURN_MINUTES"]
        - result["CURRENT_SCHEDULED_TURN_MINUTES"]
    )

    current_arrival = pd.to_datetime(
        current_rotation["ROTATION_PRIOR_SCHEDULED_ARRIVAL_UTC"], errors="coerce"
    )
    full_arrival = pd.to_datetime(
        full_rotation["FULL_ROTATION_PRIOR_SCHEDULED_ARRIVAL_UTC"], errors="coerce"
    )
    current_match = result["CURRENT_MATCH_FOUND"].eq(1)
    full_match = result["FULL_MATCH_FOUND"].eq(1)
    same_prior = current_match & full_match & current_arrival.eq(full_arrival)
    match_change = np.full(len(result), "BOTH_UNMATCHED", dtype=object)
    match_change[current_match & ~full_match] = "MATCH_REMOVED"
    match_change[~current_match & full_match] = "MATCH_ADDED"
    match_change[current_match & full_match & ~same_prior] = "DIFFERENT_PRIOR_INBOUND"
    match_change[same_prior] = "SAME_PRIOR_INBOUND"
    both_unmatched_changed = (
        ~current_match
        & ~full_match
        & result["CURRENT_STATUS"].ne(result["FULL_STATUS"]).fillna(False)
    )
    match_change[both_unmatched_changed] = "UNMATCHED_STATUS_CHANGED"
    result["MATCH_CHANGE"] = pd.Series(
        match_change, index=result.index, dtype="string"
    )
    result["STATUS_CHANGED"] = result["CURRENT_STATUS"].ne(result["FULL_STATUS"])
    result["CURRENT_PRIOR_SCHEDULED_ARRIVAL_UTC"] = current_arrival
    result["FULL_PRIOR_SCHEDULED_ARRIVAL_UTC"] = full_arrival
    return result


def summarize_long_turns(
    comparison: pd.DataFrame,
    *,
    thresholds_hours: Iterable[int] = (8, 12, 24, 48),
) -> pd.DataFrame:
    """Compare long scheduled-turn counts under two rotation histories."""

    _require_columns(
        comparison,
        ["CURRENT_SCHEDULED_TURN_MINUTES", "FULL_SCHEDULED_TURN_MINUTES"],
        "rotation history comparison",
    )
    rows = []
    for hours in thresholds_hours:
        if hours <= 0:
            raise ValueError("Long-turn thresholds must be positive")
        current = comparison["CURRENT_SCHEDULED_TURN_MINUTES"].gt(hours * 60)
        full = comparison["FULL_SCHEDULED_TURN_MINUTES"].gt(hours * 60)
        rows.append(
            {
                "threshold_hours": hours,
                "current_count": int(current.sum()),
                "full_history_count": int(full.sum()),
                "count_change": int(full.sum() - current.sum()),
                "current_percent": current.mean() * 100,
                "full_history_percent": full.mean() * 100,
                "current_long_resolved": int((current & ~full).sum()),
                "new_full_history_long": int((~current & full).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("threshold_hours")


def add_departure_rotation_features(
    departures: pd.DataFrame,
    inbound: pd.DataFrame,
    *,
    airport: str,
) -> pd.DataFrame:
    """Append causal preceding-aircraft features to a departure feature table."""

    airport = str(airport).strip().upper()
    if airport not in AIRPORT_TIME_ZONES:
        raise KeyError(f"No time-zone mapping for target airport {airport}")

    departure_required = ["DATE", "Origin", "Tail_Number"]
    inbound_required = [
        "FlightDate",
        "Reporting_Airline",
        "Tail_Number",
        "Flight_Number_Reporting_Airline",
        "Origin",
        "Dest",
        "ArrDelay",
        "ArrDel15",
        "CRSArrTime",
        "CRSElapsedTime",
        "DATE",
    ]
    _require_columns(departures, departure_required, "departure rotation input")
    _require_columns(inbound, inbound_required, "inbound rotation input")
    overlap = sorted(set(ROTATION_OUTPUT_COLUMNS) & set(departures.columns))
    if overlap:
        raise ValueError(f"Departure input already contains rotation columns: {overlap}")

    departure_origin = _normalize_airport(departures["Origin"])
    inbound_destination = _normalize_airport(inbound["Dest"])
    if not departure_origin.eq(airport).all():
        raise ValueError(f"Departure input contains origins other than {airport}")
    if not inbound_destination.eq(airport).all():
        raise ValueError(f"Inbound input contains destinations other than {airport}")

    target_cutoff_utc = _localize_by_airport(
        departures["DATE"], departure_origin, context="target departure cutoffs"
    )
    schedule = reconstruct_inbound_schedule(inbound)
    inbound_arrival_delay = _complete_numeric(inbound, "ArrDelay")
    inbound_delayed = _complete_numeric(inbound, "ArrDel15")
    if not inbound_delayed.isin([0, 1]).all():
        raise ValueError("Inbound ArrDel15 must be binary")
    actual_arrival_utc = schedule["scheduled_arrival_utc"] + pd.to_timedelta(
        inbound_arrival_delay, unit="m"
    )

    target_tails = _normalize_tail(departures["Tail_Number"])
    inbound_tails = _normalize_tail(inbound["Tail_Number"])
    prior_type, prior_position = _preceding_leg_matches(
        target_tails,
        target_cutoff_utc,
        inbound_tails,
        schedule["scheduled_arrival_utc"],
    )

    row_count = len(departures)
    match = prior_position >= 0
    matched_rows = np.flatnonzero(match)
    matched_inbound = prior_position[match]
    arrived = np.zeros(row_count, dtype=bool)
    arrived[match] = (
        actual_arrival_utc.iloc[matched_inbound].to_numpy(dtype="datetime64[ns]")
        <= target_cutoff_utc.iloc[matched_rows].to_numpy(dtype="datetime64[ns]")
    )
    not_arrived = match & ~arrived

    status = np.full(row_count, "NO_PRIOR_EVENT", dtype=object)
    status[prior_type == "DEPARTURE"] = "PREVIOUS_EVENT_DEPARTURE"
    status[match] = "NOT_ARRIVED"
    status[arrived] = "ARRIVED"
    status[target_tails.isna().to_numpy()] = "MISSING_TAIL"

    scheduled_turn = np.full(row_count, np.nan, dtype=float)
    actual_turn = np.full(row_count, np.nan, dtype=float)
    inbound_delay_feature = np.full(row_count, np.nan, dtype=float)
    inbound_delayed_feature = np.full(row_count, np.nan, dtype=float)
    overdue = np.full(row_count, np.nan, dtype=float)
    if match.any():
        cutoff_values = target_cutoff_utc.iloc[matched_rows].to_numpy(dtype="datetime64[ns]")
        scheduled_arrival_values = schedule["scheduled_arrival_utc"].iloc[
            matched_inbound
        ].to_numpy(dtype="datetime64[ns]")
        scheduled_turn[match] = (
            cutoff_values - scheduled_arrival_values
        ).astype("timedelta64[s]").astype(float) / 60
        overdue[match] = 0.0
        overdue[not_arrived] = np.maximum(scheduled_turn[not_arrived], 0.0)
    if arrived.any():
        arrived_rows = np.flatnonzero(arrived)
        arrived_inbound = prior_position[arrived]
        actual_turn[arrived] = (
            target_cutoff_utc.iloc[arrived_rows].to_numpy(dtype="datetime64[ns]")
            - actual_arrival_utc.iloc[arrived_inbound].to_numpy(dtype="datetime64[ns]")
        ).astype("timedelta64[s]").astype(float) / 60
        inbound_delay_feature[arrived] = inbound_arrival_delay.iloc[
            arrived_inbound
        ].to_numpy(dtype=float)
        inbound_delayed_feature[arrived] = inbound_delayed.iloc[
            arrived_inbound
        ].to_numpy(dtype=float)

    result = departures.copy()
    result["ROTATION_STATUS"] = pd.Series(status, index=result.index, dtype="string")
    result["ROTATION_MATCH_FOUND"] = pd.Series(
        match.astype(np.int8), index=result.index, dtype="Int8"
    )
    result["ROTATION_INBOUND_ORIGIN"] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )
    result["ROTATION_SCHEDULED_TURN_MINUTES"] = scheduled_turn
    result["ROTATION_INBOUND_ARRIVED_BY_CUTOFF"] = pd.Series(
        arrived.astype(np.int8), index=result.index, dtype="Int8"
    )
    result["ROTATION_INBOUND_NOT_ARRIVED_BY_CUTOFF"] = pd.Series(
        not_arrived.astype(np.int8), index=result.index, dtype="Int8"
    )
    result["ROTATION_INBOUND_OVERDUE_MINUTES"] = overdue
    result["ROTATION_LOG_INBOUND_OVERDUE_MINUTES"] = np.log1p(overdue)
    result["ROTATION_ACTUAL_TURN_MINUTES"] = actual_turn
    result["ROTATION_LOG_ACTUAL_TURN_MINUTES"] = np.log1p(actual_turn)
    result["ROTATION_INBOUND_ARR_DELAY"] = inbound_delay_feature
    result["ROTATION_INBOUND_DELAYED_15"] = inbound_delayed_feature
    result["ROTATION_LOG_SCHEDULED_TURN_MINUTES"] = np.log1p(scheduled_turn)

    result["ROTATION_TARGET_CUTOFF_UTC"] = target_cutoff_utc.to_numpy()
    for column in ROTATION_AUDIT_COLUMNS[1:]:
        result[column] = pd.NA

    if match.any():
        result.loc[result.index[matched_rows], "ROTATION_INBOUND_ORIGIN"] = (
            _normalize_airport(inbound["Origin"])
            .iloc[matched_inbound]
            .to_numpy()
        )
        audit_values = {
            "ROTATION_PRIOR_REPORTING_AIRLINE": inbound["Reporting_Airline"],
            "ROTATION_PRIOR_FLIGHT_NUMBER": inbound[
                "Flight_Number_Reporting_Airline"
            ],
            "ROTATION_PRIOR_FLIGHT_DATE": inbound["FlightDate"],
            "ROTATION_PRIOR_SCHEDULED_DEPARTURE_UTC": schedule[
                "scheduled_departure_utc"
            ],
            "ROTATION_PRIOR_SCHEDULED_ARRIVAL_UTC": schedule[
                "scheduled_arrival_utc"
            ],
            "ROTATION_SCHEDULE_RECONSTRUCTION_ERROR_MINUTES": schedule[
                "schedule_error_minutes"
            ],
        }
        for column, values in audit_values.items():
            result.loc[result.index[matched_rows], column] = values.iloc[
                matched_inbound
            ].to_numpy()
    if arrived.any():
        arrived_rows = np.flatnonzero(arrived)
        result.loc[
            result.index[arrived_rows], "ROTATION_PRIOR_ACTUAL_ARRIVAL_UTC"
        ] = actual_arrival_utc.iloc[prior_position[arrived]].to_numpy()

    return result


def add_departure_rotation_features_full_history(
    departures: pd.DataFrame,
    raw_bts: pd.DataFrame,
    *,
    airport: str,
) -> pd.DataFrame:
    """Append rotation features reconstructed from every raw airport movement.

    The returned schema intentionally matches ``add_departure_rotation_features``
    so a model can compare history scopes without changing its feature manifest.
    Raw BTS is reduced to completed inbound movements and non-cancelled outbound
    blocking events by ``prepare_rotation_history``.
    """

    overlap = sorted(set(ROTATION_OUTPUT_COLUMNS) & set(departures.columns))
    if overlap:
        raise ValueError(f"Departure input already contains rotation columns: {overlap}")

    history = prepare_rotation_history(raw_bts, airport=airport)
    full = build_full_history_rotation_audit(departures, history, airport=airport)
    result = departures.copy()

    direct_map = {
        "ROTATION_STATUS": "FULL_ROTATION_STATUS",
        "ROTATION_MATCH_FOUND": "FULL_ROTATION_MATCH_FOUND",
        "ROTATION_INBOUND_ORIGIN": "FULL_ROTATION_INBOUND_ORIGIN",
        "ROTATION_SCHEDULED_TURN_MINUTES": (
            "FULL_ROTATION_SCHEDULED_TURN_MINUTES"
        ),
        "ROTATION_INBOUND_ARRIVED_BY_CUTOFF": (
            "FULL_ROTATION_INBOUND_ARRIVED_BY_CUTOFF"
        ),
        "ROTATION_INBOUND_NOT_ARRIVED_BY_CUTOFF": (
            "FULL_ROTATION_INBOUND_NOT_ARRIVED_BY_CUTOFF"
        ),
        "ROTATION_ACTUAL_TURN_MINUTES": "FULL_ROTATION_ACTUAL_TURN_MINUTES",
        "ROTATION_INBOUND_ARR_DELAY": "FULL_ROTATION_INBOUND_ARR_DELAY",
        "ROTATION_INBOUND_DELAYED_15": "FULL_ROTATION_INBOUND_DELAYED_15",
    }
    for output_column, audit_column in direct_map.items():
        result[output_column] = full[audit_column].to_numpy()

    matched = pd.to_numeric(result["ROTATION_MATCH_FOUND"], errors="coerce").eq(1)
    not_arrived = pd.to_numeric(
        result["ROTATION_INBOUND_NOT_ARRIVED_BY_CUTOFF"], errors="coerce"
    ).eq(1)
    scheduled_turn = pd.to_numeric(
        result["ROTATION_SCHEDULED_TURN_MINUTES"], errors="coerce"
    )
    actual_turn = pd.to_numeric(
        result["ROTATION_ACTUAL_TURN_MINUTES"], errors="coerce"
    )
    overdue = pd.Series(np.nan, index=result.index, dtype=float)
    overdue.loc[matched] = 0.0
    overdue.loc[not_arrived] = scheduled_turn.loc[not_arrived].clip(lower=0.0)
    result["ROTATION_INBOUND_OVERDUE_MINUTES"] = overdue
    result["ROTATION_LOG_INBOUND_OVERDUE_MINUTES"] = np.log1p(overdue)
    result["ROTATION_LOG_ACTUAL_TURN_MINUTES"] = np.log1p(actual_turn)
    result["ROTATION_LOG_SCHEDULED_TURN_MINUTES"] = np.log1p(scheduled_turn)

    departure_origin = _normalize_airport(departures["Origin"])
    result["ROTATION_TARGET_CUTOFF_UTC"] = _localize_by_airport(
        departures["DATE"], departure_origin, context="target departure cutoffs"
    ).to_numpy()
    audit_map = {
        "ROTATION_PRIOR_REPORTING_AIRLINE": (
            "FULL_ROTATION_PRIOR_REPORTING_AIRLINE"
        ),
        "ROTATION_PRIOR_FLIGHT_NUMBER": "FULL_ROTATION_PRIOR_FLIGHT_NUMBER",
        "ROTATION_PRIOR_FLIGHT_DATE": "FULL_ROTATION_PRIOR_FLIGHT_DATE",
        "ROTATION_PRIOR_SCHEDULED_DEPARTURE_UTC": (
            "FULL_ROTATION_PRIOR_SCHEDULED_DEPARTURE_UTC"
        ),
        "ROTATION_PRIOR_SCHEDULED_ARRIVAL_UTC": (
            "FULL_ROTATION_PRIOR_SCHEDULED_ARRIVAL_UTC"
        ),
        "ROTATION_PRIOR_ACTUAL_ARRIVAL_UTC": (
            "FULL_ROTATION_PRIOR_ACTUAL_ARRIVAL_UTC"
        ),
        "ROTATION_SCHEDULE_RECONSTRUCTION_ERROR_MINUTES": (
            "FULL_ROTATION_SCHEDULE_RECONSTRUCTION_ERROR_MINUTES"
        ),
    }
    for output_column, audit_column in audit_map.items():
        result[output_column] = full[audit_column].to_numpy()

    # Preserve the documented append order and fail before writing if a future
    # refactor accidentally omits or duplicates a rotation field.
    result = result[[*departures.columns, *ROTATION_OUTPUT_COLUMNS]]
    validate_departure_rotation_features(result)
    return result


def validate_departure_rotation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate rotation identities, ranges, and causal masking rules."""

    _require_columns(df, ROTATION_OUTPUT_COLUMNS, "departure rotation output")
    match = pd.to_numeric(df["ROTATION_MATCH_FOUND"], errors="coerce")
    arrived = pd.to_numeric(
        df["ROTATION_INBOUND_ARRIVED_BY_CUTOFF"], errors="coerce"
    )
    not_arrived = pd.to_numeric(
        df["ROTATION_INBOUND_NOT_ARRIVED_BY_CUTOFF"], errors="coerce"
    )
    indicators = pd.concat([match, arrived, not_arrived], axis=1)
    if indicators.isna().any().any() or not indicators.isin([0, 1]).all().all():
        raise ValueError("Rotation indicator columns must be complete and binary")
    if not match.eq(arrived + not_arrived).all():
        raise ValueError("Rotation match must equal arrived plus not-arrived status")

    matched = match.eq(1)
    arrived_mask = arrived.eq(1)
    not_arrived_mask = not_arrived.eq(1)
    status = df["ROTATION_STATUS"].astype("string")
    if not status.loc[arrived_mask].eq("ARRIVED").all():
        raise ValueError("Arrived rotations have an inconsistent status")
    if not status.loc[not_arrived_mask].eq("NOT_ARRIVED").all():
        raise ValueError("Not-arrived rotations have an inconsistent status")

    match_required = [
        "ROTATION_INBOUND_ORIGIN",
        "ROTATION_SCHEDULED_TURN_MINUTES",
        "ROTATION_INBOUND_OVERDUE_MINUTES",
        "ROTATION_LOG_INBOUND_OVERDUE_MINUTES",
        "ROTATION_LOG_SCHEDULED_TURN_MINUTES",
        "ROTATION_PRIOR_REPORTING_AIRLINE",
        "ROTATION_PRIOR_FLIGHT_NUMBER",
        "ROTATION_PRIOR_FLIGHT_DATE",
        "ROTATION_PRIOR_SCHEDULED_DEPARTURE_UTC",
        "ROTATION_PRIOR_SCHEDULED_ARRIVAL_UTC",
        "ROTATION_SCHEDULE_RECONSTRUCTION_ERROR_MINUTES",
    ]
    if df.loc[matched, match_required].isna().any().any():
        raise ValueError("A matched rotation is missing required schedule information")
    if df.loc[~matched, match_required].notna().any().any():
        raise ValueError("An unmatched departure contains inbound rotation information")

    actual_only = [
        "ROTATION_ACTUAL_TURN_MINUTES",
        "ROTATION_LOG_ACTUAL_TURN_MINUTES",
        "ROTATION_INBOUND_ARR_DELAY",
        "ROTATION_INBOUND_DELAYED_15",
        "ROTATION_PRIOR_ACTUAL_ARRIVAL_UTC",
    ]
    if df.loc[arrived_mask, actual_only].isna().any().any():
        raise ValueError("An arrived rotation is missing observable actual information")
    if df.loc[~arrived_mask, actual_only].notna().any().any():
        raise ValueError("Actual inbound outcomes leaked before the aircraft arrived")

    actual_turn = pd.to_numeric(df["ROTATION_ACTUAL_TURN_MINUTES"], errors="coerce")
    overdue = pd.to_numeric(
        df["ROTATION_INBOUND_OVERDUE_MINUTES"], errors="coerce"
    )
    log_overdue = pd.to_numeric(
        df["ROTATION_LOG_INBOUND_OVERDUE_MINUTES"], errors="coerce"
    )
    scheduled_turn = pd.to_numeric(
        df["ROTATION_SCHEDULED_TURN_MINUTES"], errors="coerce"
    )
    log_scheduled_turn = pd.to_numeric(
        df["ROTATION_LOG_SCHEDULED_TURN_MINUTES"], errors="coerce"
    )
    log_actual_turn = pd.to_numeric(
        df["ROTATION_LOG_ACTUAL_TURN_MINUTES"], errors="coerce"
    )
    delayed = pd.to_numeric(df["ROTATION_INBOUND_DELAYED_15"], errors="coerce")
    if (actual_turn.dropna() < 0).any():
        raise ValueError("Actual turn minutes cannot be negative for an arrived aircraft")
    if (scheduled_turn.dropna() <= 0).any():
        raise ValueError("Matched scheduled turn minutes must be positive")
    if (overdue.dropna() < 0).any():
        raise ValueError("Inbound overdue minutes cannot be negative")
    if not overdue.loc[arrived_mask].eq(0).all():
        raise ValueError("Arrived aircraft must have zero inbound overdue minutes")
    if not delayed.dropna().isin([0, 1]).all():
        raise ValueError("Observable inbound delayed indicator must be binary")
    if not np.allclose(log_overdue.dropna(), np.log1p(overdue.dropna())):
        raise ValueError("Log inbound overdue minutes is inconsistent")
    if not np.allclose(
        log_scheduled_turn.dropna(), np.log1p(scheduled_turn.dropna())
    ):
        raise ValueError("Log scheduled turn minutes is inconsistent")
    if not np.allclose(log_actual_turn.dropna(), np.log1p(actual_turn.dropna())):
        raise ValueError("Log actual turn minutes is inconsistent")

    cutoff = pd.to_datetime(df["ROTATION_TARGET_CUTOFF_UTC"], errors="coerce")
    actual_arrival = pd.to_datetime(
        df["ROTATION_PRIOR_ACTUAL_ARRIVAL_UTC"], errors="coerce"
    )
    if cutoff.isna().any():
        raise ValueError("Rotation target cutoff must be complete")
    if not actual_arrival.loc[arrived_mask].le(cutoff.loc[arrived_mask]).all():
        raise ValueError("An observable inbound arrival occurs after the target cutoff")

    numeric_summary_columns = [
        column
        for column in ROTATION_FEATURES
        if column not in {"ROTATION_STATUS", "ROTATION_INBOUND_ORIGIN"}
    ]
    numeric = df[numeric_summary_columns].apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame(
        {
            "dtype": df[ROTATION_FEATURES].dtypes.astype(str),
            "missing_count": df[ROTATION_FEATURES].isna().sum(),
            "missing_percent": df[ROTATION_FEATURES].isna().mean().mul(100),
            "minimum": numeric.min(),
            "maximum": numeric.max(),
        }
    )
