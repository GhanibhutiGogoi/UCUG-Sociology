#!/usr/bin/env bash
# Chains the 5 pipeline scripts end-to-end.
set -euo pipefail
cd "$(dirname "$0")"
VENV="./venv/bin/python"
for s in scripts/01_build_data.py scripts/02_gpa_normalize.py scripts/03_statistics.py scripts/04_plots.py scripts/05_report.py; do
    echo ""
    echo ">>>>> running $s"
    "$VENV" "$s"
done
echo ""
echo "Pipeline complete. Outputs:"
echo "  data/   -> $(ls data | wc -l | tr -d ' ') files"
echo "  figures/ -> $(ls figures | wc -l | tr -d ' ') PNGs"
echo "  findings.md ($(wc -c < findings.md | tr -d ' ') bytes)"
