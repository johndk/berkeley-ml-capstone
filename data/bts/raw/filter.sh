#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

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
        "$PYTHON" "$SCRIPT_DIR/filter.py" "$SCRIPT_DIR/$year" "$year" "$airport"
    done
done
