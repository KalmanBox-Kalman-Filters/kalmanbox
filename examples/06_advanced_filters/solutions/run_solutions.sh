#!/bin/bash
# Executa todas as solutions da FASE 6 (advanced filters) e reporta resultados.
#
# Usage:  bash run_solutions.sh
# Exit:   0 se tudo OK, 1 se algum script falhar.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PASS=0
FAIL=0
FAILED_SCRIPTS=""

echo "============================================================"
echo "Running FASE 6 solution scripts (advanced filters)..."
echo "============================================================"
echo ""

for script in \
    sol_01_ekf_ukf.py \
    sol_02_square_root_information.py \
    sol_03_ensemble_kalman.py; do
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
