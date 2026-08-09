#!/usr/bin/env python3
"""Build a destination-specific BTS table from cleaned airport-year CSV files."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path


AIRPORT_PATTERN = re.compile(r"^[A-Z0-9]{3,4}$")
YEAR_PATTERN = re.compile(r"^\d{4}$")
INPUT_FILE_PATTERN = re.compile(
    r"^(?P<airport>[A-Z0-9]{3,4})_(?P<year>\d{4})\.csv$"
)
REQUIRED_COLUMNS = (
    "Year",
    "Origin",
    "Dest",
    "DATE",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
)
SORT_COLUMNS = (
    "DATE",
    "Origin",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Tail_Number",
)
FLIGHT_NUMBER_COLUMN = "Flight_Number_Reporting_Airline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate cleaned BTS airport files for a year, retain flights whose "
            "destination is the requested airport, remove overlapping duplicate rows, "
            "and sort the result deterministically."
        )
    )
    parser.add_argument(
        "airport",
        help="Destination airport code, for example JFK",
    )
    parser.add_argument(
        "year",
        help="Four-digit flight year, for example 2019",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "cleaned",
        help=(
            "Directory containing cleaned AIRPORT_YEAR.csv files "
            "(default: data/bts/cleaned)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output CSV file or existing directory. When a directory is supplied, "
            "cat_AIRPORT_YEAR.csv is created inside it "
            "(default: data/bts/cat_AIRPORT_YEAR.csv)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file",
    )
    return parser.parse_args()


def normalize_airport(value: str) -> str:
    airport = value.strip().upper()
    if not AIRPORT_PATTERN.fullmatch(airport):
        raise ValueError(
            f"Invalid airport code {value!r}; expected 3 or 4 letters/digits."
        )
    return airport


def validate_year(value: str) -> str:
    year = value.strip()
    if not YEAR_PATTERN.fullmatch(year):
        raise ValueError(f"Invalid year {value!r}; expected four digits.")
    return year


def normalize_flight_number(value: str) -> str:
    """Return one integer representation for equivalent BTS flight numbers."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("flight number is empty")

    try:
        number = Decimal(stripped)
    except InvalidOperation as exc:
        raise ValueError(f"invalid flight number {value!r}") from exc

    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"flight number must be an integer; received {value!r}")

    return str(int(number))


def resolve_output_path(
    requested_output: Path | None,
    airport: str,
    year: str,
) -> Path:
    filename = f"cat_{airport}_{year}.csv"
    if requested_output is None:
        return Path(__file__).resolve().parent / filename

    output = requested_output.resolve()
    if output.is_dir():
        return output / filename
    return output


def discover_input_files(input_dir: Path, year: str) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    input_files = []
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        match = INPUT_FILE_PATTERN.fullmatch(path.name)
        if match and match.group("year") == year:
            input_files.append(path)

    input_files.sort(key=lambda path: path.name)
    if not input_files:
        raise ValueError(
            f"No cleaned airport files matching AIRPORT_{year}.csv found in {input_dir}"
        )
    return input_files


def validate_header(header: list[str] | None, input_file: Path) -> list[str]:
    if not header:
        raise ValueError(f"Input CSV is empty or missing a header: {input_file}")

    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        raise ValueError(
            f"{input_file} is missing required columns: {', '.join(missing)}"
        )
    return header


def read_destination_rows(
    input_files: list[Path],
    airport: str,
    year: str,
) -> tuple[list[str], list[tuple[str, ...]], int, int]:
    expected_header: list[str] | None = None
    unique_rows: set[tuple[str, ...]] = set()
    matched_rows = 0
    rows_scanned = 0

    for input_file in input_files:
        with input_file.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.reader(source)
            header = validate_header(next(reader, None), input_file)

            if expected_header is None:
                expected_header = header
            elif header != expected_header:
                raise ValueError(
                    f"{input_file} has a different CSV header than {input_files[0]}"
                )

            year_index = header.index("Year")
            dest_index = header.index("Dest")
            flight_number_index = header.index(FLIGHT_NUMBER_COLUMN)

            for line_number, row in enumerate(reader, start=2):
                rows_scanned += 1
                if len(row) != len(header):
                    raise ValueError(
                        f"{input_file}:{line_number} has {len(row)} fields; "
                        f"expected {len(header)}"
                    )
                if row[year_index] != year:
                    raise ValueError(
                        f"{input_file}:{line_number} contains Year={row[year_index]!r}; "
                        f"expected {year!r}"
                    )
                if row[dest_index].strip().upper() == airport:
                    try:
                        row[flight_number_index] = normalize_flight_number(
                            row[flight_number_index]
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"{input_file}:{line_number}: {exc}"
                        ) from exc
                    matched_rows += 1
                    unique_rows.add(tuple(row))

    if expected_header is None:
        raise ValueError("No readable input CSV headers were found.")

    sort_indexes = [
        expected_header.index(column)
        for column in SORT_COLUMNS
        if column in expected_header
    ]
    rows = sorted(unique_rows, key=lambda row: tuple(row[index] for index in sort_indexes))
    return expected_header, rows, rows_scanned, matched_rows


def write_csv_atomic(
    output_file: Path,
    header: list[str],
    rows: list[tuple[str, ...]],
    force: bool,
) -> None:
    if output_file.exists() and not force:
        raise ValueError(
            f"Output file already exists: {output_file}. Use --force to replace it."
        )
    if not output_file.parent.is_dir():
        raise ValueError(f"Output directory does not exist: {output_file.parent}")

    resolved_output = output_file.resolve()
    temporary_file: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output_file.parent,
            prefix=f".{output_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary_file = Path(destination.name)
            writer = csv.writer(destination)
            writer.writerow(header)
            writer.writerows(rows)

        os.replace(temporary_file, resolved_output)
    except Exception:
        if temporary_file is not None:
            temporary_file.unlink(missing_ok=True)
        raise


def concatenate_bts(
    input_dir: Path,
    output_file: Path,
    airport: str,
    year: str,
    force: bool = False,
) -> tuple[int, int, int, int]:
    input_files = discover_input_files(input_dir, year)
    resolved_output = output_file.resolve()
    if any(path.resolve() == resolved_output for path in input_files):
        raise ValueError("Output path cannot overwrite an input CSV file.")
    if output_file.exists() and not force:
        raise ValueError(
            f"Output file already exists: {output_file}. Use --force to replace it."
        )

    header, rows, rows_scanned, matched_rows = read_destination_rows(
        input_files=input_files,
        airport=airport,
        year=year,
    )
    if not rows:
        raise ValueError(
            f"No flights with Dest={airport!r} and Year={year!r} were found."
        )

    write_csv_atomic(output_file, header, rows, force)
    duplicate_rows = matched_rows - len(rows)
    return len(input_files), rows_scanned, len(rows), duplicate_rows


def main() -> int:
    args = parse_args()

    try:
        airport = normalize_airport(args.airport)
        year = validate_year(args.year)
        input_dir = args.input_dir.resolve()
        output_file = resolve_output_path(args.output, airport, year)
        files_read, rows_scanned, rows_written, duplicate_rows = concatenate_bts(
            input_dir=input_dir,
            output_file=output_file,
            airport=airport,
            year=year,
            force=args.force,
        )
    except (OSError, ValueError) as exc:
        print(f"cat_bts.py: {exc}", file=sys.stderr)
        return 1

    print(f"Read {files_read} cleaned BTS files and scanned {rows_scanned} rows.")
    print(
        f"Wrote {rows_written} unique flights with Dest={airport} and Year={year} "
        f"to {output_file}."
    )
    print(f"Removed {duplicate_rows} overlapping duplicate rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
