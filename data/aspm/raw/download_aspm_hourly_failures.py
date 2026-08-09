#!/usr/bin/env python3
"""Retry failed FAA ASPM hourly downloads for one airport and year.

The script reads:

    aspm_output/run_YEAR_AIRPORT/failures.csv

Successful dates are merged into the run's existing
``aspm_YEAR_AIRPORT.csv`` file. Only dates that still fail after all attempts
remain in ``failures.csv``; the failures file is removed when every retry
succeeds.

Example:
    python download_aspm_hourly_failures.py OAK 2019
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from download_aspm_hourly_v3 import (
    establish_session,
    fetch_report,
    parse_airport,
)


FAILURE_COLUMNS = ("airport", "report_date", "error")
OUTPUT_KEY_COLUMNS = ("airport", "report_date", "Hour")


def parse_year(value: str) -> int:
    """Parse a four-digit calendar year."""
    if len(value) != 4 or not value.isdigit():
        raise argparse.ArgumentTypeError("Year must contain four digits, for example 2019.")
    year = int(value)
    if year < 1900 or year > 2100:
        raise argparse.ArgumentTypeError("Year must be between 1900 and 2100.")
    return year


def positive_integer(value: str) -> int:
    """Parse a strictly positive integer command-line value."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be a positive integer.") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return number


def nonnegative_number(value: str) -> float:
    """Parse a nonnegative floating-point command-line value."""
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be a nonnegative number.") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be a nonnegative number.")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retry the dates recorded in an FAA ASPM failures.csv file and merge "
            "successful hourly reports into the existing airport-year CSV."
        )
    )
    parser.add_argument("airport", type=parse_airport, help="Airport code, for example OAK")
    parser.add_argument("year", type=parse_year, help="Four-digit year, for example 2019")
    parser.add_argument(
        "--attempts",
        type=positive_integer,
        default=3,
        help="Maximum attempts per failed date (default: 3)",
    )
    parser.add_argument(
        "--delay",
        type=nonnegative_number,
        default=2.0,
        help="Base delay in seconds between attempts and dates (default: 2.0)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "aspm_output",
        help="Directory containing run_YEAR_AIRPORT folders",
    )
    parser.add_argument(
        "--no-leading-space",
        action="store_true",
        help="Do not include the downloader's legacy leading space in the FAA airport literal",
    )
    return parser.parse_args()


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            frame.to_csv(temporary_file, index=False)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def load_failures(failure_path: Path, airport: str, year: int) -> list[dict[str, object]]:
    """Load, validate, and deduplicate failed dates."""
    if not failure_path.is_file():
        raise FileNotFoundError(f"Failures file does not exist: {failure_path}")

    failures = pd.read_csv(failure_path)
    missing_columns = [column for column in FAILURE_COLUMNS if column not in failures.columns]
    if missing_columns:
        raise ValueError(
            f"{failure_path} is missing required columns: {', '.join(missing_columns)}"
        )

    normalized_airports = failures["airport"].astype("string").str.strip().str.upper()
    wrong_airports = sorted(
        normalized_airports.loc[normalized_airports.ne(airport)].dropna().unique()
    )
    if wrong_airports or normalized_airports.isna().any():
        raise ValueError(
            f"{failure_path} contains airport values other than {airport}: "
            + ", ".join(map(str, wrong_airports))
        )

    parsed_dates = pd.to_datetime(failures["report_date"], errors="coerce")
    if parsed_dates.isna().any():
        invalid_values = failures.loc[parsed_dates.isna(), "report_date"].astype(str).tolist()
        raise ValueError(f"{failure_path} contains invalid dates: {invalid_values}")
    if not parsed_dates.dt.year.eq(year).all():
        wrong_years = sorted(parsed_dates.loc[parsed_dates.dt.year.ne(year)].dt.year.unique())
        raise ValueError(
            f"{failure_path} contains dates outside {year}: {wrong_years}"
        )

    validated = pd.DataFrame({
        "airport": airport,
        "report_date": parsed_dates.dt.date,
        "error": failures["error"].fillna("").astype(str),
    })
    validated = validated.drop_duplicates(subset=["report_date"], keep="last")
    validated = validated.sort_values("report_date", kind="stable")
    return validated.to_dict("records")


def write_remaining_failures(
    failure_path: Path,
    airport: str,
    remaining: dict[date, str],
) -> None:
    """Persist only unresolved dates, or remove the file when none remain."""
    if not remaining:
        failure_path.unlink(missing_ok=True)
        return

    failure_frame = pd.DataFrame([
        {
            "airport": airport,
            "report_date": report_date.isoformat(),
            "error": remaining[report_date],
        }
        for report_date in sorted(remaining)
    ], columns=FAILURE_COLUMNS)
    atomic_write_csv(failure_frame, failure_path)


def merge_successful_report(output_path: Path, report: pd.DataFrame) -> int:
    """Merge one successful date into the airport-year output and return its row count."""
    missing_report_keys = [column for column in OUTPUT_KEY_COLUMNS if column not in report.columns]
    if missing_report_keys:
        raise ValueError(
            "Downloaded report is missing merge columns: " + ", ".join(missing_report_keys)
        )

    if output_path.is_file():
        existing = pd.read_csv(output_path)
        missing_output_keys = [
            column for column in OUTPUT_KEY_COLUMNS if column not in existing.columns
        ]
        if missing_output_keys:
            raise ValueError(
                f"{output_path} is missing merge columns: {', '.join(missing_output_keys)}"
            )
        combined = pd.concat([existing, report], ignore_index=True, sort=False)
    else:
        combined = report.copy()

    combined["airport"] = combined["airport"].astype("string").str.strip().str.upper()
    combined["report_date"] = pd.to_datetime(
        combined["report_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    combined["Hour"] = pd.to_numeric(combined["Hour"], errors="raise")
    combined = combined.drop_duplicates(subset=list(OUTPUT_KEY_COLUMNS), keep="last")
    combined = combined.sort_values(
        ["airport", "report_date", "Hour"], kind="stable"
    ).reset_index(drop=True)
    atomic_write_csv(combined, output_path)
    return len(combined)


def main() -> int:
    args = parse_args()
    airport = args.airport
    year = args.year
    run_dir = args.output_root.resolve() / f"run_{year}_{airport}"
    output_path = run_dir / f"aspm_{year}_{airport}.csv"
    raw_dir = run_dir / "raw_html"
    failure_path = run_dir / "failures.csv"

    try:
        failure_records = load_failures(failure_path, airport, year)
    except (OSError, ValueError) as exc:
        print(f"download_aspm_hourly_failures.py: {exc}", file=sys.stderr)
        return 2

    remaining = {
        record["report_date"]: str(record["error"])
        for record in failure_records
    }
    report_dates = sorted(remaining)
    total = len(report_dates)

    print(f"Failed dates to retry: {total}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Output CSV: {output_path}", flush=True)
    print(f"Failures CSV: {failure_path}", flush=True)

    if not report_dates:
        failure_path.unlink(missing_ok=True)
        print("No failed dates remain; failures.csv removed.", flush=True)
        return 0

    with requests.Session() as session:
        establish_session(session)
        print(f"Session cookies before POST: {list(session.cookies.keys())}", flush=True)

        for date_index, report_date in enumerate(report_dates, start=1):
            print(f"[{date_index}/{total}] {airport} {report_date}", flush=True)
            last_error = "Retry did not run"

            for attempt in range(1, args.attempts + 1):
                print(f"  Attempt {attempt}/{args.attempts}", flush=True)
                try:
                    report = fetch_report(
                        session,
                        airport,
                        report_date,
                        raw_dir,
                        leading_space=not args.no_leading_space,
                    )
                    row_count = merge_successful_report(output_path, report)
                    remaining.pop(report_date, None)
                    write_remaining_failures(failure_path, airport, remaining)
                    print(
                        f"  SUCCESS: merged {len(report):,} rows; "
                        f"output now contains {row_count:,} rows",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    last_error = str(exc)
                    print(f"  FAILED: {last_error}", file=sys.stderr, flush=True)
                    if attempt < args.attempts:
                        wait_seconds = args.delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                        print(f"  Waiting {wait_seconds:.1f} seconds before retry", flush=True)
                        time.sleep(wait_seconds)
            else:
                remaining[report_date] = last_error
                write_remaining_failures(failure_path, airport, remaining)
                print(
                    f"  Giving up after {args.attempts} attempts; failure retained",
                    file=sys.stderr,
                    flush=True,
                )

            if date_index < total and args.delay > 0:
                time.sleep(args.delay + random.uniform(0, 1))

    if remaining:
        print(
            f"{len(remaining)} failed date(s) remain in {failure_path}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print("All failed dates downloaded successfully; failures.csv removed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
