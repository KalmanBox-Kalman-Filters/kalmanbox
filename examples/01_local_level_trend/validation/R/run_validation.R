# =============================================================================
# Run all R validation scripts for FASE1 models
# Pacotes necessarios: KFAS, dlm
#
# Uso: Rscript run_validation.R
# =============================================================================

cat("============================================\n")
cat("FASE1 - R Validation Suite (KFAS + dlm)\n")
cat("============================================\n\n")

# Verificar pacotes
required_pkgs <- c("KFAS", "dlm")
for (pkg in required_pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        stop(sprintf("Package '%s' is not installed. Install with: install.packages('%s')", pkg, pkg))
    }
}

cat("Running Local Level validation...\n")
cat("--------------------------------------------\n")
source("validate_local_level.R")
cat("\n")

cat("Running Local Linear Trend validation...\n")
cat("--------------------------------------------\n")
source("validate_local_linear_trend.R")
cat("\n")

cat("Running BSM validation...\n")
cat("--------------------------------------------\n")
source("validate_bsm.R")
cat("\n")

cat("============================================\n")
cat("All R validations complete.\n")
cat("============================================\n")

# Lista de arquivos gerados
output_files <- c(
    "results_local_level.csv",
    "results_local_linear_trend.csv",
    "results_bsm.csv",
    "states_local_level_kfas.csv",
    "states_llt_kfas.csv",
    "states_bsm_kfas.csv"
)

cat("\nGenerated files:\n")
for (f in output_files) {
    if (file.exists(f)) {
        cat(sprintf("  [OK] %s\n", f))
    } else {
        cat(sprintf("  [MISSING] %s\n", f))
    }
}
