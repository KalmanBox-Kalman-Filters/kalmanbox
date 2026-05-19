#!/bin/bash
# Executa todas as solutions e reporta resultados
#
# Usage: bash run_solutions.sh
# Exit code: 0 se tudo OK, 1 se algum script falhar

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PASS=0
FAIL=0
FAILED_SCRIPTS=""

echo "============================================================"
echo "Running all solution scripts..."
echo "============================================================"
echo ""

for script in sol_01_local_level.py sol_02_local_linear_trend.py sol_03_bsm.py; do
    echo ">>> Running $script ..."
    if python3 "$script"; then
        PASS=$((PASS + 1))
        echo ""
        echo ">>> $script: PASSED"
    else
        FAIL=$((FAIL + 1))
        FAILED_SCRIPTS="$FAILED_SCRIPTS $script"
        echo ""
        echo ">>> $script: FAILED"
    fi
    echo ""
done

echo "============================================================"
echo "Summary: $PASS passed, $FAIL failed"
if [ $FAIL -gt 0 ]; then
    echo "Failed scripts:$FAILED_SCRIPTS"
    echo "============================================================"
    exit 1
else
    echo "All solutions passed!"
    echo "============================================================"
    exit 0
fi
