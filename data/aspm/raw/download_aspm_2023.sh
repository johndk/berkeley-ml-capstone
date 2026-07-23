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

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 AIRPORT" >&2
    echo "Example: $0 JFK" >&2
    exit 2
fi

AIRPORT="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"

if [[ ! "$AIRPORT" =~ ^[A-Z0-9]{3,4}$ ]]; then
    echo "Invalid airport code: $1" >&2
    exit 2
fi

START_DATE="2023-01-01"
END_DATE="2023-12-31"
RUN_NAME="run_2023_${AIRPORT}"
OUTPUT_FILE="aspm_2023_${AIRPORT}.csv"

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
