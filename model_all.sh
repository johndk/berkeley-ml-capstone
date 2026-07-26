#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

years=(2019 2023 2024)
airports=(JFK)

for year in "${years[@]}"; do
    for airport in "${airports[@]}"; do
        papermill \
            notebooks/model_1a.ipynb \
            "/tmp/model_1a_${year}_${airport}.ipynb" \
            -p YEAR "$year" \
            -p AIRPORT "$airport" \
            --cwd notebooks
    done
done
