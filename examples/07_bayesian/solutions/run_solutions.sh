#!/bin/bash
# Runs all Bayesian solution scripts and reports results.
#
# Usage: bash run_solutions.sh
# Exit code: 0 if every script succeeds, 1 if any fails.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PASS=0
FAIL=0
FAILED_SCRIPTS=""

echo "============================================================"
echo "Running FASE 7 Bayesian solution scripts..."
echo "============================================================"
echo ""

for script in sol_01_gibbs_ffbs.py sol_02_mcmc_diagnostics.py; do
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
    echo "All Bayesian solutions passed!"
    echo "============================================================"
    exit 0
fi
