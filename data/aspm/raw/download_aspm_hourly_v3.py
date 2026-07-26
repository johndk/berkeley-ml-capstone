#!/usr/bin/env python3
"""FAA ASPM hourly downloader, version 3.

Each execution writes to a start-date timestamped directory, for example:
    aspm_output/run_2024_JFK/
        aspm_2024_JFK.csv
        failures.csv          # only when failures occur
        raw_html/

Example:
    python download_aspm_hourly_v3.py --airport JFK \
        --start 2024-01-01 --end 2024-12-31 --continue-on-error
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.aspm.faa.gov"
REPORT_URL = f"{BASE_URL}/apm/sys/apm-server-x.asp"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
CONNECT_TIMEOUT_SECONDS = 10
SESSION_READ_TIMEOUT_SECONDS = 30
REPORT_READ_TIMEOUT_SECONDS = 60


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Dates must use YYYY-MM-DD.") from exc


def dates_between(start: date, end: date):
    if end < start:
        raise ValueError("End date must not be earlier than start date.")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def airport_literal(airport: str, leading_space: bool) -> str:
    airport = airport.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{3,4}", airport):
        raise ValueError(f"Invalid airport code: {airport!r}")
    return f"'{(' ' if leading_space else '')}{airport}'"


def parse_airport(value: str) -> str:
    airport = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{3,4}", airport):
        raise argparse.ArgumentTypeError(f"Invalid airport code: {value!r}.")
    return airport


def build_payload(airport: str, report_date: date, leading_space: bool) -> dict[str, str]:
    ymd = report_date.strftime("%Y%m%d")
    month = report_date.strftime("%m")
    day = report_date.strftime("%d")
    year = report_date.strftime("%Y")
    loc = airport_literal(airport, leading_space)

    return {
        "dstyle": "d",
        "dfld": "yyyymmdd",
        "dlist": ymd,
        "fromdate": "",
        "todate": "",
        "llist": loc,
        "clist": "",
        "keylist": "HR_LOCAL",
        "compdstyle": "",
        "compdfld": "",
        "compdlist": "",
        "compfromdate": "",
        "comptodate": "",
        "line": (
            "SELECT HR_LOCAL, ? FROM LOCID_TOTALS_DAY  "
            f"WHERE YYYYMMDD IN ({ymd}) AND LOCID IN ({loc}) "
            "GROUP BY HR_LOCAL ORDER BY HR_LOCAL"
        ),
        "cmd": "aa1",
        "nopage": "y",
        "nost": "y",
        "defs": "",
        "avgdays": "",
        "oktosave": "y",
        "sys": "ap",
        "where": f" where yyyymmdd in ({ymd}) and locid in ({loc})",
        "locInput_param": "",
        "locQuick_param": "",
        "locMode": "on",
        "dtype": "d",
        "fm_m": month,
        "fy_m": year,
        "tm_m": month,
        "ty_m": year,
        "daytype": "all",
        "fy_y": year,
        "ytype": "c",
        "ty_y": year,
        "ydaytype": "all",
        "fm_r": month,
        "fd_r": day,
        "fy_r": year,
        "tm_r": month,
        "td_r": day,
        "ty_r": year,
        "rdaytype": "all",
        "compdtype": "d",
        "compfm_m": month,
        "compfy_m": year,
        "comptm_m": month,
        "compty_m": year,
        "compdaytype": "all",
        "compfy_y": year,
        "compytype": "c",
        "compty_y": year,
        "ycompdaytype": "all",
        "compfm_r": month,
        "compfd_r": day,
        "compfy_r": year,
        "comptm_r": month,
        "comptd_r": day,
        "compty_r": year,
        "rcompdaytype": "all",
        "oagptm": "oag",
        "hrfrom": "??",
        "hrto": "??",
        "usedep": "y",
        "reptype": "r1",
        "reportformat": "asp",
        "nosubtot": "1",
    }


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        names = []
        keep = []
        for col in df.columns:
            last_part = str(col[-1]).strip()
            if not last_part or last_part.lower().startswith("unnamed"):
                names.append("unnamed")
                keep.append(False)
                continue
            parts = [
                str(part).strip()
                for part in col
                if str(part).strip() and not str(part).lower().startswith("unnamed")
            ]
            names.append(parts[-1] if parts else "unnamed")
            keep.append(True)
        df.columns = names
        df = df.loc[:, keep]
    else:
        df.columns = [str(col).strip() for col in df.columns]
        df = df.loc[:, [col and not col.lower().startswith("unnamed") for col in df.columns]]
    return df


def score_table(df: pd.DataFrame) -> int:
    if df.empty:
        return -100
    text = " ".join(str(c).lower() for c in df.columns)
    text += " " + " ".join(str(x).lower() for x in df.head(5).to_numpy().ravel())
    weights = {
        "hour": 8,
        "hr_local": 8,
        "scheduled departures": 6,
        "scheduled arrivals": 6,
        "taxi out": 5,
        "taxi in": 5,
        "gate departure": 5,
        "airport departure": 5,
        "gate arrival": 5,
        "delay": 3,
        "on-time": 3,
    }
    score = sum(weight for term, weight in weights.items() if term in text)
    if df.shape[0] >= 10:
        score += 4
    if df.shape[1] >= 4:
        score += 4
    return score


def extract_report_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    ranked = []
    for index, table in enumerate(tables):
        table = flatten_columns(table)
        ranked.append((score_table(table), index, table))
    ranked.sort(key=lambda item: item[0], reverse=True)
    score, index, table = ranked[0]
    print(f"  Found {len(tables)} HTML tables; selected table {index}, score={score}, shape={table.shape}", flush=True)
    if score < 5:
        raise ValueError("No table clearly resembled an ASPM hourly report.")
    return table


def establish_session(session: requests.Session) -> None:
    """
    Try to obtain fresh public guest cookies, but do not fail if the FAA blocks
    preliminary GET requests. The report POST may establish its own session.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    for url in (f"{BASE_URL}/", f"{BASE_URL}/apm/sys/Default.asp"):
        try:
            print(f"Session probe {url} ...", flush=True)
            response = session.get(
                url,
                headers=headers,
                timeout=(CONNECT_TIMEOUT_SECONDS, SESSION_READ_TIMEOUT_SECONDS),
                allow_redirects=True,
            )
            print(f"Session probe {url} -> HTTP {response.status_code}", flush=True)
            if response.ok:
                return
        except requests.RequestException as exc:
            print(f"Session probe {url} failed: {exc}", flush=True)

    print(
        "Preliminary GET requests did not establish a session; "
        "continuing with the report POST directly.",
        flush=True,
    )


def fetch_report(
    session: requests.Session,
    airport: str,
    report_date: date,
    raw_dir: Path,
    leading_space: bool,
) -> pd.DataFrame:
    print("  Requesting report from FAA ASPM ...", flush=True)
    response = session.post(
        REPORT_URL,
        data=build_payload(airport, report_date, leading_space),
        headers={
            "User-Agent": USER_AGENT,
            "Referer": f"{BASE_URL}/",
            "Origin": BASE_URL,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        timeout=(CONNECT_TIMEOUT_SECONDS, REPORT_READ_TIMEOUT_SECONDS),
        allow_redirects=True,
    )
    print(f"  Report response -> HTTP {response.status_code}", flush=True)
    response.raise_for_status()

    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{airport}_{report_date:%Y-%m-%d}.html"
    raw_path.write_bytes(response.content)

    response.encoding = response.encoding or response.apparent_encoding or "utf-8"
    html = response.text
    lower = re.sub(r"\s+", " ", html).lower()
    for phrase in ("session expired", "invalid query", "no records found"):
        if phrase in lower:
            raise RuntimeError(f"Response contains {phrase!r}; inspect {raw_path}")

    table = extract_report_table(html)
    if "Hour" in table.columns:
        table = table[table["Hour"].astype(str).str.strip() != "Total :"].copy()
    table.insert(0, "report_date", report_date.isoformat())
    table.insert(0, "airport", airport)
    print(f"  Saved raw response: {raw_path} ({len(response.content):,} bytes)", flush=True)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="Download hourly FAA ASPM reports into a timestamped run directory.")
    parser.add_argument("--airport", type=parse_airport, default="JFK")
    parser.add_argument("--start", type=parse_date, default=date(2024, 1, 1))
    parser.add_argument("--end", type=parse_date, default=date(2024, 1, 1))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("aspm_output"),
        help="Parent directory for timestamped run folders.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run-folder name. Default: run_STARTDATE.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional output CSV filename. Default: aspm_hourly_AIRPORT.csv.",
    )
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--no-leading-space", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    airport = args.airport
    report_dates = list(dates_between(args.start, args.end))
    total = len(report_dates)
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    timestamp = args.start.strftime("%Y%m%d")
    run_name = args.run_name or f"run_{timestamp}"
    run_dir = args.output_root / run_name
    output_filename = args.output_file or f"aspm_hourly_{airport}.csv"
    output_path = run_dir / output_filename
    raw_dir = run_dir / "raw_html"
    failure_path = run_dir / "failures.csv"
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"Planned requests: {total}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Output CSV: {output_path}", flush=True)
    print(f"Raw HTML: {raw_dir}", flush=True)
    with requests.Session() as session:
        establish_session(session)
        print(f"Session cookies before POST: {list(session.cookies.keys())}", flush=True)

        count = 0
        for report_date in report_dates:
            count += 1
            print(f"[{count}/{total}] {airport} {report_date}", flush=True)
            try:
                frames.append(
                    fetch_report(
                        session,
                        airport,
                        report_date,
                        raw_dir,
                        leading_space=not args.no_leading_space,
                    )
                )
                pd.concat(frames, ignore_index=True, sort=False).to_csv(output_path, index=False)
            except Exception as exc:
                print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
                failures.append({
                    "airport": airport,
                    "report_date": report_date.isoformat(),
                    "error": str(exc),
                })
                if not args.continue_on_error:
                    break
            if count < total:
                time.sleep(max(0, args.delay + random.uniform(0, 1)))

    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined.to_csv(output_path, index=False)
        print(f"Saved {len(combined):,} rows to {output_path}", flush=True)

    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        print(f"Saved failures to {failure_path}", file=sys.stderr, flush=True)

    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
