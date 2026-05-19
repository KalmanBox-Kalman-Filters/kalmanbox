# =============================================================================
# Run all R validation scripts for FASE2 structural models
# Pacotes necessarios: KFAS
#
# Uso: cd examples/02_structural_models/validation/R && Rscript run_validation.R
#
# Scripts executados:
#   1. validate_ucm.R          - UCM (Brazil GDP): trend + cycle
#   2. validate_stochastic_cycle.R - Stochastic cycle (Brazil IPCA)
#   3. validate_regression_ssm.R   - Regression SSM (UK Gas): fixed vs TVP
# =============================================================================

cat("============================================\n")
cat("FASE2 - R Validation Suite (KFAS)\n")
cat("============================================\n\n")

# Verificar pacotes
required_pkgs <- c("KFAS")
for (pkg in required_pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        stop(sprintf("Package '%s' is not installed. Install with: install.packages('%s')", pkg, pkg))
    }
    cat(sprintf("[OK] Package '%s' available (version %s)\n",
        pkg, as.character(packageVersion(pkg))))
}
cat("\n")

# Track results
results <- list()
errors <- c()

# --- UCM ---
cat("Running UCM validation (Brazil GDP)...\n")
cat("--------------------------------------------\n")
tryCatch({
    source("validate_ucm.R")
    results[["UCM"]] <- "OK"
}, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    results[["UCM"]] <<- "FAILED"
    errors <<- c(errors, sprintf("UCM: %s", e$message))
})
cat("\n")

# --- Stochastic Cycle ---
cat("Running Stochastic Cycle validation (Brazil IPCA)...\n")
cat("--------------------------------------------\n")
tryCatch({
    source("validate_stochastic_cycle.R")
    results[["Stochastic Cycle"]] <- "OK"
}, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    results[["Stochastic Cycle"]] <<- "FAILED"
    errors <<- c(errors, sprintf("Stochastic Cycle: %s", e$message))
})
cat("\n")

# --- Regression SSM ---
cat("Running Regression SSM validation (UK Gas)...\n")
cat("--------------------------------------------\n")
tryCatch({
    source("validate_regression_ssm.R")
    results[["Regression SSM"]] <- "OK"
}, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    results[["Regression SSM"]] <<- "FAILED"
    errors <<- c(errors, sprintf("Regression SSM: %s", e$message))
})
cat("\n")

# --- Summary ---
cat("============================================\n")
cat("Validation Summary\n")
cat("============================================\n")
for (name in names(results)) {
    status <- ifelse(results[[name]] == "OK", "[PASS]", "[FAIL]")
    cat(sprintf("  %s %s\n", status, name))
}

if (length(errors) > 0) {
    cat("\nErrors:\n")
    for (err in errors) {
        cat(sprintf("  - %s\n", err))
    }
}

# --- List generated files ---
output_files <- c(
    "results_ucm_params.csv",
    "results_ucm_decomposition.csv",
    "states_ucm_kfas.csv",
    "results_stochastic_cycle_params.csv",
    "results_stochastic_cycle_decomposition.csv",
    "states_stochastic_cycle_kfas.csv",
    "results_regression_ssm_params.csv",
    "results_regression_ssm_tvp_coefs.csv",
    "results_regression_ssm_fixed_vs_ols.csv"
)

cat("\nGenerated files:\n")
for (f in output_files) {
    if (file.exists(f)) {
        cat(sprintf("  [OK] %s\n", f))
    } else {
        cat(sprintf("  [MISSING] %s\n", f))
    }
}

cat("\n============================================\n")
if (length(errors) == 0) {
    cat("All FASE2 R validations complete.\n")
} else {
    cat(sprintf("FASE2 R validations finished with %d error(s).\n", length(errors)))
}
cat("============================================\n")
