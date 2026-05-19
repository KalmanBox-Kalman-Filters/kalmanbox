#!/bin/bash
# Run all solution scripts for structural models (FASE 2)
# Exit on first error
set -e

cd "$(dirname "$0")"

echo "========================================"
echo "Running Structural Models Solutions"
echo "========================================"
echo ""

python3 sol_01_ucm.py
echo ""

python3 sol_02_stochastic_cycle.py
echo ""

python3 sol_03_regression_ssm.py
echo ""

echo "========================================"
echo "ALL SOLUTIONS PASSED"
echo "========================================"
