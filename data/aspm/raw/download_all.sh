#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_SCRIPT="$SCRIPT_DIR/download_airport.sh"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 YEAR" >&2
    echo "Example: $0 2024" >&2
    exit 2
fi

YEAR="$1"

if [[ ! "$YEAR" =~ ^[0-9]{4}$ ]]; then
    echo "Invalid year: $1" >&2
    exit 2
fi

if [[ ! -x "$DOWNLOAD_SCRIPT" ]]; then
    echo "Download script is missing or not executable: $DOWNLOAD_SCRIPT" >&2
    exit 1
fi

AIRPORTS=(
    ANC AUS BNA BOS BUF BUR BWI CLE CLT
    CVG DCA DEN DFW DTW FLL HNL HOU IAD
    IAH IND JAX LAS LAX LGB MCI MCO MIA MKE
    MSP MSY OAK ONT PBI PDX PHL PHX PIT PSP
    RDU RSW SAN SAT SDF SEA SFO SJC SJU SLC
    SMF SNA TPA ORD ATL JFK
)

#AIRPORTS=(OAK RDU SAT SDF PBI SFO)

TOTAL="${#AIRPORTS[@]}"
DOWNLOADED=0

for INDEX in "${!AIRPORTS[@]}"; do
    AIRPORT="${AIRPORTS[$INDEX]}"
    echo
    echo "[$((INDEX + 1))/$TOTAL] Downloading $AIRPORT for $YEAR"
    "$DOWNLOAD_SCRIPT" "$AIRPORT" "$YEAR"
    DOWNLOADED=$((DOWNLOADED + 1))
done

echo
echo "Finished downloading $DOWNLOADED airports for $YEAR."
