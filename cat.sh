#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [[ -n "${CAPSTONE_PYTHON:-}" ]]; then
    PYTHON_BIN="$CAPSTONE_PYTHON"
elif [[ -x "$VENV_PYTHON" ]]; then
    PYTHON_BIN="$VENV_PYTHON"
else
    PYTHON_BIN="$(command -v python3)"
fi

DESTINATION="JFK"
YEARS=(2019 2023 2024)

for year in "${YEARS[@]}"; do
    echo "Concatenating $DESTINATION arrival sources for $year"

    "$PYTHON_BIN" "$SCRIPT_DIR/data/bts/cat_bts.py" \
        "$DESTINATION" "$year" \
        --output "$SCRIPT_DIR/data/bts/cleaned_${DESTINATION}_${year}.csv" \
        --force

    "$PYTHON_BIN" "$SCRIPT_DIR/data/aspm/cat_aspm.py" \
        "$DESTINATION" "$year" \
        --output "$SCRIPT_DIR/data/aspm/cleaned_${DESTINATION}_${year}.csv" \
        --force

    "$PYTHON_BIN" "$SCRIPT_DIR/data/noaa/cat_noaa.py" \
        "$DESTINATION" "$year" \
        --output "$SCRIPT_DIR/data/noaa/cleaned_${DESTINATION}_${year}.csv" \
        --force
done

echo "Finished concatenating $DESTINATION arrival sources for ${YEARS[*]}"
