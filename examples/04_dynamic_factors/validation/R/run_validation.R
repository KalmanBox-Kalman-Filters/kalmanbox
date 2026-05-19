# =============================================================================
# Run all R validation scripts for Dynamic Factor Models (FASE4)
# Pacotes necessarios: MARSS
#
# Uso (a partir deste diretorio):
#   Rscript run_validation.R
# =============================================================================

cat("=============================================================================\n")
cat("FASE4 - Dynamic Factor Models R Validation Suite (MARSS)\n")
cat("=============================================================================\n\n")

# Verificar pacotes
required_pkgs <- c("MARSS")
for (pkg in required_pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        stop(sprintf(
            "Pacote '%s' nao instalado. Instale com: install.packages('%s')",
            pkg, pkg
        ))
    }
}

# Garante que os scripts e os CSV de saida fiquem no diretorio do runner
script_dir <- tryCatch(
    dirname(normalizePath(sys.frame(1)$ofile)),
    error = function(e) getwd()
)
if (length(script_dir) == 0 || is.na(script_dir)) script_dir <- getwd()
setwd(script_dir)
cat(sprintf("Working directory: %s\n\n", getwd()))

# -----------------------------------------------------------------------------
# 1) DFM K=2
# -----------------------------------------------------------------------------
cat("Running DFM K=2 validation...\n")
cat("---------------------------------------------\n")
source("validate_dfm.R")
cat("\n")

# -----------------------------------------------------------------------------
# 2) Mixed Frequency Nowcast
# -----------------------------------------------------------------------------
cat("Running Mixed-Frequency Nowcast validation...\n")
cat("---------------------------------------------\n")
source("validate_mixed_freq.R")
cat("\n")

# -----------------------------------------------------------------------------
# 3) Variance Decomposition (R^2)
# -----------------------------------------------------------------------------
cat("Running Variance Decomposition validation...\n")
cat("---------------------------------------------\n")
source("validate_variance_decomp.R")
cat("\n")

cat("=============================================================================\n")
cat("All Dynamic Factor Models R validations complete.\n")
cat("=============================================================================\n")

# Lista de arquivos gerados
output_files <- c(
    # DFM K=2
    "results_dfm_factors.csv",
    "results_dfm_loadings.csv",
    "results_dfm_fit_summary.csv",
    "results_dfm_idio_var.csv",
    # Mixed frequency
    "results_mixed_freq_nowcast.csv",
    "results_mixed_freq_loadings.csv",
    "results_mixed_freq_summary.csv",
    # Variance decomposition
    "results_variance_decomp.csv",
    "results_variance_decomp_loadings.csv",
    "results_variance_decomp_summary.csv"
)

cat("\nGenerated files:\n")
all_ok <- TRUE
for (f in output_files) {
    if (file.exists(f)) {
        cat(sprintf("  [OK]      %s\n", f))
    } else {
        cat(sprintf("  [MISSING] %s\n", f))
        all_ok <- FALSE
    }
}

if (!all_ok) {
    cat("\n[WARN] Alguns arquivos esperados nao foram gerados.\n")
} else {
    cat("\nTodos os arquivos esperados foram gerados.\n")
}
