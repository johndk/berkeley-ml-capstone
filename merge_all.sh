#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

years=(2019 2023 2024)
airports=(JFK ORD ATL)

for year in "${years[@]}"; do
    for airport in "${airports[@]}"; do
        papermill \
            notebooks/merge.ipynb \
            "/tmp/merge_${year}_${airport}.ipynb" \
            -p YEAR "$year" \
            -p AIRPORT "$airport" \
            --cwd notebooks
    done
done
