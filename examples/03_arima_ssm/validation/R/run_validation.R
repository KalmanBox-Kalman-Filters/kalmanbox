# =============================================================================
# Run all R validation scripts for ARIMA SSM (FASE3)
# Pacotes necessarios: KFAS
#
# Uso: Rscript run_validation.R
# =============================================================================

cat("=============================================================================\n")
cat("FASE3 - ARIMA SSM R Validation Suite (KFAS)\n")
cat("=============================================================================\n\n")

# Verificar pacotes
required_pkgs <- c("KFAS")
for (pkg in required_pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        stop(sprintf("Package '%s' is not installed. Install with: install.packages('%s')", pkg, pkg))
    }
}

cat("Running ARIMA SSM validation...\n")
cat("---------------------------------------------\n")
source("validate_arima_ssm.R")
cat("\n")

cat("Running MLE Comparison validation...\n")
cat("---------------------------------------------\n")
source("validate_comparison_mle.R")
cat("\n")

cat("=============================================================================\n")
cat("All ARIMA SSM R validations complete.\n")
cat("=============================================================================\n")

# Lista de arquivos gerados
output_files <- c(
    "results_arima_nile.csv",
    "results_arima_airline.csv",
    "results_comparison_mle_nile.csv",
    "results_comparison_mle_airline.csv",
    "results_comparison_mle_all.csv"
)

cat("\nGenerated files:\n")
for (f in output_files) {
    if (file.exists(f)) {
        cat(sprintf("  [OK] %s\n", f))
    } else {
        cat(sprintf("  [MISSING] %s\n", f))
    }
}
