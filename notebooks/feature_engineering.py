"""Deterministic feature engineering for the capstone flight-delay datasets.

This module intentionally contains no learned preprocessing. Imputation, scaling,
categorical encoding, feature selection, and class balancing belong in model
pipelines fitted on training data only.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


WEATHER_CONDITION_COLUMNS = [
    "Rain",
    "Drizzle",
    "Snow",
    "Fog",
    "Mist",
    "Thunderstorm",
    "FreezingPrecip",
    "Showers",
]

PRE_ENGINEERED_FEATURES = [
    "SCHED_DEP_MINUTE_OF_DAY",
    "SCHED_DEP_HOUR",
    "SCHED_DEP_TIME_SIN",
    "SCHED_DEP_TIME_COS",
    "SCHED_ARR_MINUTE_OF_DAY",
    "SCHED_ARR_TIME_SIN",
    "SCHED_ARR_TIME_COS",
    "TIME_OF_DAY",
    "IS_WEEKEND",
    "DAY_OF_WEEK_SIN",
    "DAY_OF_WEEK_COS",
    "DAY_OF_YEAR",
    "DAY_OF_YEAR_SIN",
    "DAY_OF_YEAR_COS",
    "MONTH_SIN",
    "MONTH_COS",
    "YEAR_PERIOD",
    "ROUTE",
    "AIRLINE_FLIGHT_ID",
    "AIRLINE_DEST",
    "LOG_DISTANCE",
    "SCHEDULED_SPEED_PROXY",
    "ASPM_PREVIOUS_TOTAL_SCHEDULED_TRAFFIC",
    "ASPM_CURRENT_TOTAL_SCHEDULED_TRAFFIC",
    "ASPM_NEXT_TOTAL_SCHEDULED_TRAFFIC",
    "ASPM_THREE_HOUR_SCHEDULED_DEPARTURES",
    "ASPM_THREE_HOUR_SCHEDULED_ARRIVALS",
    "ASPM_THREE_HOUR_TOTAL_SCHEDULED_TRAFFIC",
    "ASPM_CURRENT_MINUS_PREVIOUS_TRAFFIC",
    "ASPM_NEXT_MINUS_CURRENT_TRAFFIC",
    "ASPM_MAX_HOURLY_TRAFFIC",
    "TEMP_DEWPOINT_SPREAD",
    "LOG_PRECIPITATION",
    "WEATHER_CONDITION_COUNT",
    "ADVERSE_WEATHER",
]

POST_PUSHBACK_ENGINEERED_FEATURES = [
    "ACTUAL_DEP_MINUTE_OF_DAY",
    "ACTUAL_DEP_TIME_SIN",
    "ACTUAL_DEP_TIME_COS",
    "DEPARTED_EARLY",
    "LOG_DEP_DELAY_MINUTES",
]

POST_TAKEOFF_ENGINEERED_FEATURES = [
    "ACTUAL_TAKEOFF_MINUTE_OF_DAY",
    "ACTUAL_TAKEOFF_TIME_SIN",
    "ACTUAL_TAKEOFF_TIME_COS",
    "LOG_TAXI_OUT_MINUTES",
]

ALL_ARRIVAL_ENGINEERED_FEATURES = (
    PRE_ENGINEERED_FEATURES
    + POST_PUSHBACK_ENGINEERED_FEATURES
    + POST_TAKEOFF_ENGINEERED_FEATURES
)

PRE_RAW_CANDIDATE_FEATURES = [
    "Year",
    "Quarter",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "OriginState",
    "Dest",
    "DestState",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "DistanceGroup",
    "ASPM_PREVIOUS_SCHEDULED_DEPARTURES",
    "ASPM_PREVIOUS_SCHEDULED_ARRIVALS",
    "ASPM_CURRENT_SCHEDULED_DEPARTURES",
    "ASPM_CURRENT_SCHEDULED_ARRIVALS",
    "ASPM_NEXT_SCHEDULED_DEPARTURES",
    "ASPM_NEXT_SCHEDULED_ARRIVALS",
    "HourlyDewPointTemperature",
    "HourlyDryBulbTemperature",
    "HourlyPrecipitation",
    "HourlyRelativeHumidity",
    "HourlyVisibility",
    "HourlyWindSpeed",
    *WEATHER_CONDITION_COLUMNS,
    "PrecipOccurred",
    "WindX",
    "WindY",
    "NOAA_AGE_MINUTES",
]

POST_PUSHBACK_RAW_CANDIDATE_FEATURES = [
    "DepTime",
    "DepDelay",
    "DepDelayMinutes",
    "DepDel15",
    "DepartureDelayGroups",
]

POST_TAKEOFF_RAW_CANDIDATE_FEATURES = ["TaxiOut", "WheelsOff"]

MODEL_TARGETS = {
    "1A": "DepDel15",
    "2A": "ArrDel15",
    "2B": "ArrDel15",
    "2C": "ArrDel15",
}

MODEL_FEATURE_CANDIDATES = {
    "1A": PRE_RAW_CANDIDATE_FEATURES + PRE_ENGINEERED_FEATURES,
    "2A": PRE_RAW_CANDIDATE_FEATURES + PRE_ENGINEERED_FEATURES,
    "2B": (
        PRE_RAW_CANDIDATE_FEATURES
        + PRE_ENGINEERED_FEATURES
        + POST_PUSHBACK_RAW_CANDIDATE_FEATURES
        + POST_PUSHBACK_ENGINEERED_FEATURES
    ),
    "2C": (
        PRE_RAW_CANDIDATE_FEATURES
        + PRE_ENGINEERED_FEATURES
        + POST_PUSHBACK_RAW_CANDIDATE_FEATURES
        + POST_PUSHBACK_ENGINEERED_FEATURES
        + POST_TAKEOFF_RAW_CANDIDATE_FEATURES
        + POST_TAKEOFF_ENGINEERED_FEATURES
    ),
}

AUDIT_ONLY_COLUMNS = [
    "FlightDate",
    "Tail_Number",
    "DATE",
    "ASPM_PREVIOUS_LOOKUP_DATE",
    "ASPM_CURRENT_LOOKUP_DATE",
    "ASPM_NEXT_LOOKUP_DATE",
    "ASPM_PREVIOUS_AIRPORT",
    "ASPM_PREVIOUS_REPORT_DATE",
    "ASPM_PREVIOUS_HOUR",
    "ASPM_PREVIOUS_DATE",
    "ASPM_PREVIOUS_OFFSET_MINUTES",
    "ASPM_CURRENT_AIRPORT",
    "ASPM_CURRENT_REPORT_DATE",
    "ASPM_CURRENT_HOUR",
    "ASPM_CURRENT_DATE",
    "ASPM_CURRENT_OFFSET_MINUTES",
    "ASPM_NEXT_AIRPORT",
    "ASPM_NEXT_REPORT_DATE",
    "ASPM_NEXT_HOUR",
    "ASPM_NEXT_DATE",
    "ASPM_NEXT_OFFSET_MINUTES",
    "NOAA_AIRPORT",
    "NOAA_DATE",
]


def _require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise KeyError(f"{context} is missing required columns: {missing}")


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def hhmm_to_minute_of_day(values: pd.Series, *, allow_2400: bool = True) -> pd.Series:
    """Convert numeric HHMM values to nullable minutes after local midnight."""

    numeric = pd.to_numeric(values, errors="coerce")
    whole_number = numeric.notna() & numeric.eq(np.floor(numeric))
    hours = np.floor(numeric / 100)
    minutes = numeric % 100
    valid_clock = whole_number & hours.between(0, 23) & minutes.between(0, 59)
    if allow_2400:
        valid_clock |= whole_number & numeric.eq(2400)

    result = pd.Series(pd.NA, index=values.index, dtype="Int16")
    result.loc[valid_clock] = (
        ((hours.loc[valid_clock] * 60) + minutes.loc[valid_clock]) % 1440
    ).astype("int16")
    return result


def _cyclical(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(values, errors="coerce")
    angle = 2 * np.pi * numeric / period
    return np.sin(angle), np.cos(angle)


def _string_token(values: pd.Series) -> pd.Series:
    tokens = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return tokens.mask(tokens.eq(""))


def _binary_indicator(condition: pd.Series, available: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=condition.index, dtype="Int8")
    result.loc[available] = condition.loc[available].astype("int8")
    return result


def add_pre_features(source: pd.DataFrame) -> pd.DataFrame:
    """Add the Appendix C features available before pushback."""

    required = [
        "Year",
        "Month",
        "DayOfWeek",
        "FlightDate",
        "Reporting_Airline",
        "Flight_Number_Reporting_Airline",
        "Origin",
        "Dest",
        "CRSDepTime",
        "CRSArrTime",
        "Distance",
        "CRSElapsedTime",
        "ASPM_PREVIOUS_SCHEDULED_DEPARTURES",
        "ASPM_PREVIOUS_SCHEDULED_ARRIVALS",
        "ASPM_CURRENT_SCHEDULED_DEPARTURES",
        "ASPM_CURRENT_SCHEDULED_ARRIVALS",
        "ASPM_NEXT_SCHEDULED_DEPARTURES",
        "ASPM_NEXT_SCHEDULED_ARRIVALS",
        "HourlyDewPointTemperature",
        "HourlyDryBulbTemperature",
        "HourlyPrecipitation",
        *WEATHER_CONDITION_COLUMNS,
    ]
    _require_columns(source, required, "pre-pushback feature input")

    df = source.copy()
    scheduled_departure = hhmm_to_minute_of_day(df["CRSDepTime"])
    scheduled_arrival = hhmm_to_minute_of_day(df["CRSArrTime"])
    df["SCHED_DEP_MINUTE_OF_DAY"] = scheduled_departure
    df["SCHED_DEP_HOUR"] = (scheduled_departure // 60).astype("Int8")
    df["SCHED_DEP_TIME_SIN"], df["SCHED_DEP_TIME_COS"] = _cyclical(
        scheduled_departure, 1440
    )
    df["SCHED_ARR_MINUTE_OF_DAY"] = scheduled_arrival
    df["SCHED_ARR_TIME_SIN"], df["SCHED_ARR_TIME_COS"] = _cyclical(
        scheduled_arrival, 1440
    )

    time_of_day = pd.Series(pd.NA, index=df.index, dtype="string")
    time_of_day.loc[scheduled_departure.between(0, 359)] = "overnight"
    time_of_day.loc[scheduled_departure.between(360, 719)] = "morning"
    time_of_day.loc[scheduled_departure.between(720, 1079)] = "afternoon"
    time_of_day.loc[scheduled_departure.between(1080, 1439)] = "evening"
    df["TIME_OF_DAY"] = time_of_day

    day_of_week = _numeric(df, "DayOfWeek")
    valid_day_of_week = day_of_week.between(1, 7)
    df["IS_WEEKEND"] = _binary_indicator(day_of_week.isin([6, 7]), valid_day_of_week)
    df["DAY_OF_WEEK_SIN"], df["DAY_OF_WEEK_COS"] = _cyclical(
        day_of_week - 1, 7
    )
    df.loc[~valid_day_of_week, ["DAY_OF_WEEK_SIN", "DAY_OF_WEEK_COS"]] = np.nan

    flight_date = pd.to_datetime(df["FlightDate"], errors="coerce")
    day_of_year = flight_date.dt.dayofyear.astype("Int16")
    days_in_year = pd.Series(
        np.where(flight_date.dt.is_leap_year, 366, 365), index=df.index
    ).where(flight_date.notna())
    annual_angle = 2 * np.pi * (day_of_year.astype("Float64") - 1) / days_in_year
    df["DAY_OF_YEAR"] = day_of_year
    df["DAY_OF_YEAR_SIN"] = np.sin(annual_angle.astype(float))
    df["DAY_OF_YEAR_COS"] = np.cos(annual_angle.astype(float))

    month = _numeric(df, "Month")
    valid_month = month.between(1, 12)
    df["MONTH_SIN"], df["MONTH_COS"] = _cyclical(month - 1, 12)
    df.loc[~valid_month, ["MONTH_SIN", "MONTH_COS"]] = np.nan
    df["YEAR_PERIOD"] = _string_token(df["Year"])

    origin = _string_token(df["Origin"])
    destination = _string_token(df["Dest"])
    airline = _string_token(df["Reporting_Airline"])
    flight_number = _string_token(df["Flight_Number_Reporting_Airline"])
    df["ROUTE"] = origin + "_" + destination
    df["AIRLINE_FLIGHT_ID"] = airline + "_" + flight_number
    df["AIRLINE_DEST"] = airline + "_" + destination

    distance = _numeric(df, "Distance")
    elapsed = _numeric(df, "CRSElapsedTime")
    df["LOG_DISTANCE"] = np.log1p(distance.where(distance >= 0))
    df["SCHEDULED_SPEED_PROXY"] = (
        60 * distance.where(distance >= 0) / elapsed.where(elapsed > 0)
    )

    previous_departures = _numeric(df, "ASPM_PREVIOUS_SCHEDULED_DEPARTURES")
    previous_arrivals = _numeric(df, "ASPM_PREVIOUS_SCHEDULED_ARRIVALS")
    current_departures = _numeric(df, "ASPM_CURRENT_SCHEDULED_DEPARTURES")
    current_arrivals = _numeric(df, "ASPM_CURRENT_SCHEDULED_ARRIVALS")
    next_departures = _numeric(df, "ASPM_NEXT_SCHEDULED_DEPARTURES")
    next_arrivals = _numeric(df, "ASPM_NEXT_SCHEDULED_ARRIVALS")

    previous_total = previous_departures + previous_arrivals
    current_total = current_departures + current_arrivals
    next_total = next_departures + next_arrivals
    three_hour_departures = previous_departures + current_departures + next_departures
    three_hour_arrivals = previous_arrivals + current_arrivals + next_arrivals

    df["ASPM_PREVIOUS_TOTAL_SCHEDULED_TRAFFIC"] = previous_total
    df["ASPM_CURRENT_TOTAL_SCHEDULED_TRAFFIC"] = current_total
    df["ASPM_NEXT_TOTAL_SCHEDULED_TRAFFIC"] = next_total
    df["ASPM_THREE_HOUR_SCHEDULED_DEPARTURES"] = three_hour_departures
    df["ASPM_THREE_HOUR_SCHEDULED_ARRIVALS"] = three_hour_arrivals
    df["ASPM_THREE_HOUR_TOTAL_SCHEDULED_TRAFFIC"] = (
        three_hour_departures + three_hour_arrivals
    )
    df["ASPM_CURRENT_MINUS_PREVIOUS_TRAFFIC"] = current_total - previous_total
    df["ASPM_NEXT_MINUS_CURRENT_TRAFFIC"] = next_total - current_total
    df["ASPM_MAX_HOURLY_TRAFFIC"] = pd.concat(
        [previous_total, current_total, next_total], axis=1
    ).max(axis=1, skipna=False)

    dry_bulb = _numeric(df, "HourlyDryBulbTemperature")
    dew_point = _numeric(df, "HourlyDewPointTemperature")
    precipitation = _numeric(df, "HourlyPrecipitation")
    df["TEMP_DEWPOINT_SPREAD"] = dry_bulb - dew_point
    df["LOG_PRECIPITATION"] = np.log1p(precipitation.clip(lower=0))

    weather_values = df[WEATHER_CONDITION_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_weather = weather_values.notna() & ~weather_values.isin([0, 1])
    if invalid_weather.any().any():
        invalid_columns = invalid_weather.any()[lambda values: values].index.tolist()
        raise ValueError(f"Weather indicators must be binary: {invalid_columns}")
    weather_count = weather_values.sum(
        axis=1, min_count=len(WEATHER_CONDITION_COLUMNS)
    )
    df["WEATHER_CONDITION_COUNT"] = weather_count.astype("Int8")
    df["ADVERSE_WEATHER"] = _binary_indicator(
        weather_count.gt(0), weather_count.notna()
    )

    return df


def add_post_pushback_features(source: pd.DataFrame) -> pd.DataFrame:
    """Add features first available after gate departure."""

    required = ["DepTime", "DepDelay", "DepDelayMinutes"]
    _require_columns(source, required, "post-pushback feature input")
    df = source.copy()

    actual_departure = hhmm_to_minute_of_day(df["DepTime"])
    df["ACTUAL_DEP_MINUTE_OF_DAY"] = actual_departure
    df["ACTUAL_DEP_TIME_SIN"], df["ACTUAL_DEP_TIME_COS"] = _cyclical(
        actual_departure, 1440
    )

    departure_delay = _numeric(df, "DepDelay")
    delay_minutes = _numeric(df, "DepDelayMinutes")
    df["DEPARTED_EARLY"] = _binary_indicator(
        departure_delay.lt(0), departure_delay.notna()
    )
    df["LOG_DEP_DELAY_MINUTES"] = np.log1p(delay_minutes.where(delay_minutes >= 0))
    return df


def add_post_takeoff_features(source: pd.DataFrame) -> pd.DataFrame:
    """Add features first available once takeoff has occurred."""

    required = ["WheelsOff", "TaxiOut"]
    _require_columns(source, required, "post-takeoff feature input")
    df = source.copy()

    actual_takeoff = hhmm_to_minute_of_day(df["WheelsOff"], allow_2400=True)
    df["ACTUAL_TAKEOFF_MINUTE_OF_DAY"] = actual_takeoff
    df["ACTUAL_TAKEOFF_TIME_SIN"], df["ACTUAL_TAKEOFF_TIME_COS"] = _cyclical(
        actual_takeoff, 1440
    )
    taxi_out = _numeric(df, "TaxiOut")
    df["LOG_TAXI_OUT_MINUTES"] = np.log1p(taxi_out.where(taxi_out >= 0))
    return df


def add_all_arrival_features(source: pd.DataFrame) -> pd.DataFrame:
    """Add every deterministic feature available to Models 2A, 2B, or 2C."""

    return add_post_takeoff_features(add_post_pushback_features(add_pre_features(source)))


def validate_engineered_features(
    df: pd.DataFrame, engineered_columns: Iterable[str]
) -> pd.DataFrame:
    """Validate presence, finite numeric values, and key feature ranges."""

    columns = list(engineered_columns)
    _require_columns(df, columns, "engineered feature output")
    numeric = df[columns].select_dtypes(include=["number"])
    infinite_counts = pd.Series(
        np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum(axis=0),
        index=numeric.columns,
    )
    if infinite_counts.gt(0).any():
        raise ValueError(
            "Engineered features contain infinite values: "
            f"{infinite_counts[infinite_counts.gt(0)].to_dict()}"
        )

    range_checks = {
        "SCHED_DEP_MINUTE_OF_DAY": (0, 1439),
        "SCHED_DEP_HOUR": (0, 23),
        "SCHED_ARR_MINUTE_OF_DAY": (0, 1439),
        "DAY_OF_YEAR": (1, 366),
        "WEATHER_CONDITION_COUNT": (0, len(WEATHER_CONDITION_COLUMNS)),
        "ACTUAL_DEP_MINUTE_OF_DAY": (0, 1439),
        "ACTUAL_TAKEOFF_MINUTE_OF_DAY": (0, 1439),
    }
    for column, (minimum, maximum) in range_checks.items():
        if column in df:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if not values.between(minimum, maximum).all():
                raise ValueError(
                    f"{column} contains values outside {minimum} through {maximum}"
                )

    binary_columns = ["IS_WEEKEND", "ADVERSE_WEATHER", "DEPARTED_EARLY"]
    for column in binary_columns:
        if column in df:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if not values.isin([0, 1]).all():
                raise ValueError(f"{column} contains non-binary values")

    return pd.DataFrame(
        {
            "dtype": df[columns].dtypes.astype(str),
            "missing_count": df[columns].isna().sum(),
            "missing_percent": df[columns].isna().mean().mul(100),
        }
    )


def model_feature_candidates(model: str, available_columns: Iterable[str]) -> list[str]:
    """Return the ordered, prediction-time-safe candidate list for a model."""

    model = model.upper()
    if model not in MODEL_FEATURE_CANDIDATES:
        raise KeyError(f"Unknown model {model!r}; expected one of {sorted(MODEL_FEATURE_CANDIDATES)}")
    available = set(available_columns)
    missing = [column for column in MODEL_FEATURE_CANDIDATES[model] if column not in available]
    if missing:
        raise KeyError(f"Model {model} candidate columns are missing: {missing}")
    return list(MODEL_FEATURE_CANDIDATES[model])
