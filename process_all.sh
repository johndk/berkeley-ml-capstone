#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

years=(2019 2023 2024)
airports=(
    ABQ ANC AUS BNA BOS BUF BUR BWI CLE CLT
    CVG DCA DEN DFW DTW FLL HNL HOU IAD
    IAH IND JAX LAS LAX LGB MCI MCO MIA MKE
    MSP MSY OAK ONT PBI PDX PHL PHX PIT PSP
    RDU RSW SAN SAT SDF SEA SFO SJC SJU SLC
    SMF SNA TPA JFK ORD ATL
)

for year in "${years[@]}"; do
    for airport in "${airports[@]}"; do
        papermill \
            notebooks/process_bts.ipynb \
            "/tmp/process_bts_${year}_${airport}.ipynb" \
            -p YEAR "$year" \
            -p AIRPORT "$airport" \
            --cwd notebooks
    done
done

for year in "${years[@]}"; do
    for airport in "${airports[@]}"; do
        papermill \
            notebooks/process_aspm.ipynb \
            "/tmp/process_aspm_${year}_${airport}.ipynb" \
            -p YEAR "$year" \
            -p AIRPORT "$airport" \
            --cwd notebooks
    done
done

for year in "${years[@]}"; do
    for airport in "${airports[@]}"; do
        papermill \
            notebooks/process_noaa.ipynb \
            "/tmp/process_noaa_${year}_${airport}.ipynb" \
            -p YEAR "$year" \
            -p AIRPORT "$airport" \
            --cwd notebooks
    done
done

for year in "${years[@]}"; do
    for airport in "${airports[@]}"; do
        papermill \
            notebooks/clean_bts.ipynb \
            "/tmp/clean_bts_${year}_${airport}.ipynb" \
            -p YEAR "$year" \
            -p AIRPORT "$airport" \
            --cwd notebooks \
            --no-log-output \
            --no-progress-bar \
            --stdout-file /dev/stdout \
            --stderr-file /dev/stderr \
            --log-level WARNING
    done
done

for year in "${years[@]}"; do
    for airport in "${airports[@]}"; do
        papermill \
            notebooks/clean_aspm.ipynb \
            "/tmp/clean_aspm_${year}_${airport}.ipynb" \
            -p YEAR "$year" \
            -p AIRPORT "$airport" \
            --cwd notebooks \
            --no-log-output \
            --no-progress-bar \
            --stdout-file /dev/stdout \
            --stderr-file /dev/stderr \
            --log-level WARNING
    done
done

for year in "${years[@]}"; do
    for airport in "${airports[@]}"; do
        papermill \
            notebooks/clean_noaa.ipynb \
            "/tmp/clean_noaa_${year}_${airport}.ipynb" \
            -p YEAR "$year" \
            -p AIRPORT "$airport" \
            --cwd notebooks \
            --no-log-output \
            --no-progress-bar \
            --stdout-file /dev/stdout \
            --stderr-file /dev/stderr \
            --log-level WARNING
    done
done
