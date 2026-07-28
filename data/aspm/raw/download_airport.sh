#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/../../../../../.venv/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "$VENV_PYTHON" ]]; then
    PYTHON_BIN="$VENV_PYTHON"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi
PYTHON_SCRIPT="$SCRIPT_DIR/download_aspm_hourly_v3.py"

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 AIRPORT YEAR" >&2
    echo "Example: $0 JFK 2019" >&2
    exit 2
fi

AIRPORT="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
YEAR="$2"

if [[ ! "$AIRPORT" =~ ^[A-Z0-9]{3,4}$ ]]; then
    echo "Invalid airport code: $1" >&2
    exit 2
fi

if [[ ! "$YEAR" =~ ^[0-9]{4}$ ]]; then
    echo "Invalid year: $2" >&2
    exit 2
fi

START_DATE="${YEAR}-01-01"
END_DATE="${YEAR}-12-31"
RUN_NAME="run_${YEAR}_${AIRPORT}"
OUTPUT_FILE="aspm_${YEAR}_${AIRPORT}.csv"
RAW_HTML_DIR="aspm_output/${RUN_NAME}/raw_html"

if [[ -e "$RAW_HTML_DIR" ]]; then
    echo "Removing previous raw HTML directory: $RAW_HTML_DIR"
    rm -rf -- "$RAW_HTML_DIR"
fi

echo
echo "==========================================================="
echo "Downloading $AIRPORT $START_DATE through $END_DATE"
echo "==========================================================="

PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$PYTHON_SCRIPT" \
    --airport "$AIRPORT" \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --run-name "$RUN_NAME" \
    --output-file "$OUTPUT_FILE" \
    --continue-on-error

echo
echo "Finished."
