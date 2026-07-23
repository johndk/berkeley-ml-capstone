"""Shared loading and preparation helpers for the EDA notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


AIRPORTS = ("ATL", "JFK", "ORD")
YEARS = (2019, 2023, 2024)
TARGET_COLUMNS = ("DepDel15", "ArrDel15")
DATE_COLUMNS = (
    "FlightDate",
    "DATE",
    "ASPM_LOOKUP_DATE",
    "ASPM_ReportDate",
    "ASPM_DATE",
    "NOAA_DATE",
)
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
        if (candidate / "data" / "merged").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate data/merged. Run the notebook from the capstone "
        "directory or its eda subdirectory."
    )


def merged_file(airport: str, year: int, project_root: Path | None = None) -> Path:
    """Return the merged CSV path for an airport and year."""
    airport = airport.upper()
    if airport not in AIRPORTS:
        raise ValueError(f"AIRPORT must be one of {AIRPORTS}; received {airport!r}.")
    if year not in YEARS:
        raise ValueError(f"YEAR must be one of {YEARS}; received {year!r}.")

    root = project_root or find_project_root()
    path = root / "data" / "merged" / f"{airport}_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Merged dataset not found: {path}")
    return path


def available_merged_files(project_root: Path | None = None) -> list[Path]:
    """List merged airport-year CSV files in a stable order."""
    root = project_root or find_project_root()
    return sorted((root / "data" / "merged").glob("*.csv"))


def load_merged(
    airport: str,
    year: int,
    *,
    usecols: list[str] | None = None,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """Load one merged dataset and parse any requested timestamp columns."""
    path = merged_file(airport, year, project_root)
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

    traffic_columns = {"ASPM_Scheduled_Departures", "ASPM_Scheduled_Arrivals"}
    if traffic_columns.issubset(data.columns):
        data["ASPM_Total_Scheduled_Traffic"] = (
            data["ASPM_Scheduled_Departures"]
            + data["ASPM_Scheduled_Arrivals"]
        )

    present_weather_flags = [
        column for column in WEATHER_FLAGS if column in data.columns
    ]
    if present_weather_flags:
        data["WeatherConditionCount"] = data[present_weather_flags].sum(axis=1)
        data["AdverseWeather"] = (data["WeatherConditionCount"] > 0).astype("int8")

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
