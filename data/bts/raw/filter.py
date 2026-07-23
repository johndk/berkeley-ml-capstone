#!/usr/bin/env python3
"""Filter BTS on-time performance CSV files by year and airport."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


REQUIRED_COLUMNS = ("Year", "Origin", "Dest")


def natural_sort_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.as_posix())
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter BTS on-time performance CSV files in subdirectories where "
            "Year == year and (Origin == airport or Dest == airport)."
        )
    )
    parser.add_argument("dir", help="Directory containing input subdirectories")
    parser.add_argument("year", help="Year to keep, for example 2024")
    parser.add_argument("airport", help="Origin or Dest airport code to keep, for example SFO")
    return parser.parse_args()


def validate_columns(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("Input CSV is empty or missing a header row.")

    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")


def iter_csv_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for child in directory.iterdir()
            if child.is_dir()
            for path in child.rglob("*.csv")
            if path.is_file()
        ),
        key=natural_sort_key,
    )


def filter_csv_file(
    input_file: Path,
    writer: csv.DictWriter,
    year: str,
    airport: str,
) -> int:
    rows_written = 0

    with input_file.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        validate_columns(reader.fieldnames)

        if reader.fieldnames != writer.fieldnames:
            raise ValueError(
                f"{input_file} has a different CSV header than the first input CSV."
            )

        for row in reader:
            if row["Year"] == year and (row["Origin"] == airport or row["Dest"] == airport):
                writer.writerow(row)
                rows_written += 1

    return rows_written


def filter_csvs(directory: Path, year: str, airport: str) -> int:
    if not directory.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")

    output_file = directory / f"{airport}.csv"
    input_files = iter_csv_files(directory)
    if not input_files:
        raise ValueError(f"No CSV files found in subdirectories of: {directory}")

    rows_written = 0
    with input_files[0].open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        validate_columns(reader.fieldnames)
        fieldnames = reader.fieldnames

    with output_file.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()

        for input_file in input_files:
            rows_written += filter_csv_file(input_file, writer, year, airport)

    return rows_written


def main() -> int:
    args = parse_args()

    try:
        rows_written = filter_csvs(Path(args.dir), args.year, args.airport)
    except OSError as exc:
        print(f"filter.py: file error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"filter.py: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {rows_written} rows to {Path(args.dir) / f'{args.airport}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
