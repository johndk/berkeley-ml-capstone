#!/usr/bin/env python3
"""Build a destination-specific NOAA origin table from cleaned airport files."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
from pathlib import Path


AIRPORT_PATTERN = re.compile(r"^[A-Z0-9]{3,4}$")
YEAR_PATTERN = re.compile(r"^\d{4}$")
INPUT_FILE_PATTERN = re.compile(
    r"^(?P<airport>[A-Z0-9]{3,4})_(?P<year>\d{4})\.csv$"
)
REQUIRED_NOAA_COLUMNS = ("DATE",)
REQUIRED_BTS_COLUMNS = ("Year", "Origin", "Dest")
AIRPORT_COLUMN = "AIRPORT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate cleaned NOAA files for origin airports serving a requested "
            "destination airport and year. Required origins are derived from BTS arrivals."
        )
    )
    parser.add_argument("airport", help="Destination airport code, for example JFK")
    parser.add_argument("year", help="Four-digit flight year, for example 2019")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "cleaned",
        help="Directory containing cleaned AIRPORT_YEAR.csv files (default: data/noaa/cleaned)",
    )
    parser.add_argument(
        "--bts-file",
        type=Path,
        help=(
            "BTS arrivals CSV used to identify origins. By default, the script uses "
            "data/bts/cleaned/cat_AIRPORT_YEAR.csv when present, otherwise "
            "data/bts/cleaned/AIRPORT_YEAR.csv."
        ),
    )
    parser.add_argument(
        "--all-airports",
        action="store_true",
        help="Concatenate every available cleaned NOAA airport file for the year",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail when an origin in the BTS arrivals table has no cleaned NOAA file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output CSV file or existing directory. When a directory is supplied, "
            "cat_AIRPORT_YEAR.csv is created inside it "
            "(default: data/noaa/cat_AIRPORT_YEAR.csv)"
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


def resolve_bts_file(requested_file: Path | None, airport: str, year: str) -> Path:
    if requested_file is not None:
        bts_file = requested_file.resolve()
        if not bts_file.is_file():
            raise ValueError(f"BTS arrivals file does not exist: {bts_file}")
        return bts_file

    data_dir = Path(__file__).resolve().parents[1]
    cleaned_dir = data_dir / "bts" / "cleaned"
    candidates = (
        cleaned_dir / f"cat_{airport}_{year}.csv",
        cleaned_dir / f"{airport}_{year}.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(
        "No BTS arrivals source was found. Run data/bts/cat_bts.py first or "
        "provide --bts-file."
    )


def load_bts_origins(bts_file: Path, airport: str, year: str) -> set[str]:
    origins: set[str] = set()
    with bts_file.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"BTS CSV is empty or missing a header: {bts_file}")
        missing = [column for column in REQUIRED_BTS_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"{bts_file} is missing required columns: {', '.join(missing)}"
            )

        for row in reader:
            if row["Year"].strip() != year:
                continue
            if row["Dest"].strip().upper() != airport:
                continue
            origin = normalize_airport(row["Origin"])
            if origin != airport:
                origins.add(origin)

    if not origins:
        raise ValueError(
            f"No origins for flights with Dest={airport!r} and Year={year!r} "
            f"were found in {bts_file}"
        )
    return origins


def discover_input_files(input_dir: Path, year: str) -> dict[str, Path]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    files: dict[str, Path] = {}
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        match = INPUT_FILE_PATTERN.fullmatch(path.name)
        if not match or match.group("year") != year:
            continue
        source_airport = match.group("airport")
        if source_airport in files:
            raise ValueError(
                f"Multiple NOAA inputs found for {source_airport} {year}: "
                f"{files[source_airport]} and {path}"
            )
        files[source_airport] = path

    if not files:
        raise ValueError(
            f"No cleaned airport files matching AIRPORT_{year}.csv found in {input_dir}"
        )
    return files


def select_input_files(
    available_files: dict[str, Path],
    required_origins: set[str] | None,
    require_complete: bool,
) -> tuple[list[tuple[str, Path]], list[str]]:
    if required_origins is None:
        selected_airports = sorted(available_files)
        missing_airports: list[str] = []
    else:
        selected_airports = sorted(required_origins & available_files.keys())
        missing_airports = sorted(required_origins - available_files.keys())

    if missing_airports and require_complete:
        raise ValueError(
            "Missing cleaned NOAA files for required origins: "
            + ", ".join(missing_airports)
        )
    if not selected_airports:
        raise ValueError("No cleaned NOAA files matched the requested airport cohort.")
    return [(airport, available_files[airport]) for airport in selected_airports], missing_airports


def validate_header(header: list[str] | None, input_file: Path) -> list[str]:
    if not header:
        raise ValueError(f"Input CSV is empty or missing a header: {input_file}")
    missing = [column for column in REQUIRED_NOAA_COLUMNS if column not in header]
    if missing:
        raise ValueError(
            f"{input_file} is missing required columns: {', '.join(missing)}"
        )
    return header


def write_concatenated_csv(
    selected_files: list[tuple[str, Path]],
    output_file: Path,
    year: str,
    force: bool,
) -> tuple[int, int]:
    if output_file.exists() and not force:
        raise ValueError(
            f"Output file already exists: {output_file}. Use --force to replace it."
        )
    if not output_file.parent.is_dir():
        raise ValueError(f"Output directory does not exist: {output_file.parent}")
    if any(path.resolve() == output_file.resolve() for _, path in selected_files):
        raise ValueError("Output path cannot overwrite an input CSV file.")

    expected_header: list[str] | None = None
    rows_written = 0
    repeated_timestamps = 0
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

            for source_airport, input_file in selected_files:
                with input_file.open("r", newline="", encoding="utf-8-sig") as source:
                    reader = csv.reader(source)
                    header = validate_header(next(reader, None), input_file)
                    if AIRPORT_COLUMN in header:
                        output_header = header
                        airport_index = header.index(AIRPORT_COLUMN)
                    else:
                        output_header = [AIRPORT_COLUMN, *header]
                        airport_index = None

                    if expected_header is None:
                        expected_header = output_header
                        writer.writerow(output_header)
                    elif output_header != expected_header:
                        raise ValueError(
                            f"{input_file} has a different CSV header than "
                            f"{selected_files[0][1]}"
                        )

                    date_index = header.index("DATE")
                    previous_date: str | None = None

                    for line_number, row in enumerate(reader, start=2):
                        if len(row) != len(header):
                            raise ValueError(
                                f"{input_file}:{line_number} has {len(row)} fields; "
                                f"expected {len(header)}"
                            )
                        if airport_index is not None:
                            row_airport = row[airport_index].strip().upper()
                            if row_airport != source_airport:
                                raise ValueError(
                                    f"{input_file}:{line_number} contains AIRPORT="
                                    f"{row[airport_index]!r}; expected {source_airport!r}"
                                )
                            output_row = tuple(row)
                        else:
                            output_row = (source_airport, *row)

                        date_value = row[date_index].strip()
                        if not date_value.startswith(f"{year}-"):
                            raise ValueError(
                                f"{input_file}:{line_number} contains DATE={date_value!r}; "
                                f"expected year {year}"
                            )
                        if previous_date is not None and date_value < previous_date:
                            raise ValueError(
                                f"{input_file}:{line_number} is not sorted by DATE"
                            )
                        if date_value == previous_date:
                            repeated_timestamps += 1

                        writer.writerow(output_row)
                        rows_written += 1
                        previous_date = date_value

        os.replace(temporary_file, output_file.resolve())
    except Exception:
        if temporary_file is not None:
            temporary_file.unlink(missing_ok=True)
        raise

    return rows_written, repeated_timestamps


def main() -> int:
    args = parse_args()
    try:
        airport = normalize_airport(args.airport)
        year = validate_year(args.year)
        input_dir = args.input_dir.resolve()
        output_file = resolve_output_path(args.output, airport, year)
        available_files = discover_input_files(input_dir, year)

        bts_file: Path | None = None
        required_origins: set[str] | None = None
        if not args.all_airports:
            bts_file = resolve_bts_file(args.bts_file, airport, year)
            required_origins = load_bts_origins(bts_file, airport, year)

        selected_files, missing_airports = select_input_files(
            available_files,
            required_origins,
            args.require_complete,
        )
        rows_written, repeated_timestamps = write_concatenated_csv(
            selected_files,
            output_file,
            year,
            args.force,
        )
    except (OSError, ValueError) as exc:
        print(f"cat_noaa.py: {exc}", file=sys.stderr)
        return 1

    if bts_file is not None:
        print(f"Derived required origins from {bts_file}.")
    print(
        f"Wrote {rows_written} NOAA rows for {len(selected_files)} origin airports "
        f"to {output_file}."
    )
    print(
        f"Preserved {repeated_timestamps} additional NOAA reports that share an "
        "airport timestamp."
    )
    if missing_airports:
        print(
            "Warning: missing cleaned NOAA files for required origins: "
            + ", ".join(missing_airports),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
