"""Shared loading and preparation helpers for the EDA notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


AIRPORTS = ("JFK",)
YEARS = (2019, 2023, 2024)
FEATURE_DATASETS = ("departures", "arrivals")
TARGET_COLUMNS = ("DepDel15", "ArrDel15")
DATE_COLUMNS = (
    "FlightDate",
    "DATE",
    "ASPM_PREVIOUS_LOOKUP_DATE",
    "ASPM_CURRENT_LOOKUP_DATE",
    "ASPM_NEXT_LOOKUP_DATE",
    "ASPM_PREVIOUS_REPORT_DATE",
    "ASPM_CURRENT_REPORT_DATE",
    "ASPM_NEXT_REPORT_DATE",
    "ASPM_PREVIOUS_DATE",
    "ASPM_CURRENT_DATE",
    "ASPM_NEXT_DATE",
    "NOAA_DATE",
)
ASPM_PERIODS = ("PREVIOUS", "CURRENT", "NEXT")
WEATHER_FLAGS = (
    "Rain",
    "Drizzle",
    "Snow",
    "Fog",
    "Mist",
    "Thunderstorm",
    "FreezingPrecip",
    "Showers",
)


def find_project_root(start: Path | None = None) -> Path:
    """Find the capstone root whether a notebook starts in root or eda/."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data" / "features").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate data/features. Run the notebook from the capstone "
        "directory or its eda subdirectory."
    )


def feature_file(
    airport: str,
    year: int,
    dataset: str = "departures",
    project_root: Path | None = None,
) -> Path:
    """Return the shared feature CSV path for an airport, year, and scenario."""
    airport = airport.upper()
    dataset = dataset.lower()
    if airport not in AIRPORTS:
        raise ValueError(f"AIRPORT must be one of {AIRPORTS}; received {airport!r}.")
    if year not in YEARS:
        raise ValueError(f"YEAR must be one of {YEARS}; received {year!r}.")
    if dataset not in FEATURE_DATASETS:
        raise ValueError(
            f"DATASET must be one of {FEATURE_DATASETS}; received {dataset!r}."
        )

    root = project_root or find_project_root()
    path = root / "data" / "features" / f"{airport}_{year}_{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")
    return path


def available_feature_files(
    dataset: str | None = None,
    project_root: Path | None = None,
) -> list[Path]:
    """List feature CSV files in a stable order, optionally for one scenario."""
    root = project_root or find_project_root()
    if dataset is not None and dataset.lower() not in FEATURE_DATASETS:
        raise ValueError(
            f"DATASET must be one of {FEATURE_DATASETS}; received {dataset!r}."
        )
    pattern = f"JFK_*_{dataset.lower()}.csv" if dataset else "JFK_*.csv"
    return sorted((root / "data" / "features").glob(pattern))


def load_features(
    airport: str,
    year: int,
    dataset: str = "departures",
    *,
    usecols: list[str] | None = None,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """Load one shared feature dataset and parse its timestamp columns."""
    path = feature_file(airport, year, dataset, project_root)
    parse_dates = [
        column
        for column in DATE_COLUMNS
        if usecols is None or column in usecols
    ]
    return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates)


def add_eda_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable helper columns used only for EDA."""
    data = frame.copy()

    if "DATE" in data:
        data["ScheduledDepartureHour"] = data["DATE"].dt.hour
        data["ScheduledDepartureMonth"] = data["DATE"].dt.month
        data["MonthName"] = data["DATE"].dt.strftime("%b")
        data["DayName"] = data["DATE"].dt.strftime("%a")
        data["TimeOfDay"] = pd.cut(
            data["ScheduledDepartureHour"],
            bins=[-1, 4, 11, 16, 20, 23],
            labels=["Overnight", "Morning", "Afternoon", "Evening", "Late night"],
        )

    period_total_columns = []
    for period in ASPM_PERIODS:
        departures = f"ASPM_{period}_SCHEDULED_DEPARTURES"
        arrivals = f"ASPM_{period}_SCHEDULED_ARRIVALS"
        total = f"ASPM_{period}_TOTAL_SCHEDULED_TRAFFIC"

        if total in data:
            period_total_columns.append(total)
        elif {departures, arrivals}.issubset(data.columns):
            data[total] = data[departures] + data[arrivals]
            period_total_columns.append(total)

    three_hour_departures = [
        f"ASPM_{period}_SCHEDULED_DEPARTURES" for period in ASPM_PERIODS
    ]
    three_hour_arrivals = [
        f"ASPM_{period}_SCHEDULED_ARRIVALS" for period in ASPM_PERIODS
    ]
    if set(three_hour_departures + three_hour_arrivals).issubset(data.columns):
        if "ASPM_THREE_HOUR_SCHEDULED_DEPARTURES" not in data:
            data["ASPM_THREE_HOUR_SCHEDULED_DEPARTURES"] = data[
                three_hour_departures
            ].sum(axis=1, min_count=len(three_hour_departures))
        if "ASPM_THREE_HOUR_SCHEDULED_ARRIVALS" not in data:
            data["ASPM_THREE_HOUR_SCHEDULED_ARRIVALS"] = data[
                three_hour_arrivals
            ].sum(axis=1, min_count=len(three_hour_arrivals))
        if "ASPM_THREE_HOUR_TOTAL_SCHEDULED_TRAFFIC" not in data:
            data["ASPM_THREE_HOUR_TOTAL_SCHEDULED_TRAFFIC"] = data[
                period_total_columns
            ].sum(axis=1, min_count=len(period_total_columns))

    present_weather_flags = [
        column for column in WEATHER_FLAGS if column in data.columns
    ]
    if "WEATHER_CONDITION_COUNT" in data:
        data["WeatherConditionCount"] = data["WEATHER_CONDITION_COUNT"]
    elif present_weather_flags:
        data["WeatherConditionCount"] = data[present_weather_flags].sum(
            axis=1, min_count=len(present_weather_flags)
        )

    if "ADVERSE_WEATHER" in data:
        data["AdverseWeather"] = data["ADVERSE_WEATHER"]
    elif "WeatherConditionCount" in data:
        data["AdverseWeather"] = (
            data["WeatherConditionCount"] > 0
        ).astype("int8")

    if "HourlyVisibility" in data:
        data["VisibilityCategory"] = pd.cut(
            data["HourlyVisibility"],
            bins=[-np.inf, 2, 5, 8, np.inf],
            labels=["Very low (<2)", "Low (2–5)", "Moderate (5–8)", "Good (8+)"],
            right=False,
        )

    for target in TARGET_COLUMNS:
        if target in data:
            data[f"{target}_Label"] = data[target].map(
                {0.0: "On time", 1.0: "Delayed 15+ min"}
            )

    return data


def category_rate(
    frame: pd.DataFrame,
    category: str,
    target: str,
    *,
    min_count: int = 1,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Return count and delayed-flight rate for a categorical column."""
    summary = (
        frame.groupby(category, observed=True)[target]
        .agg(Flights="size", DelayRate="mean")
        .reset_index()
    )
    summary = summary.loc[summary["Flights"] >= min_count]
    if top_n is not None:
        summary = summary.nlargest(top_n, "Flights")
    return summary


def safe_sample(
    frame: pd.DataFrame,
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a reproducible sample without failing on small datasets."""
    if len(frame) <= sample_size:
        return frame.copy()
    return frame.sample(sample_size, random_state=random_state)
