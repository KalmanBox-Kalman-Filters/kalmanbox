# =============================================================================
# Run all R validation scripts for FASE9 complete-workflow examples
# Pacotes necessarios: KFAS, dlm, MARSS
#
# Uso (a partir de examples/09_complete_workflow/validation/R/):
#   Rscript run_validation.R
# =============================================================================

cat("=============================================================================\n")
cat("  FASE9 - R Validation Suite (complete workflow)\n")
cat("    - Univariate (KFAS + dlm) on uk_drivers\n")
cat("    - Multivariate (MARSS)   on us_macro_panel\n")
cat("=============================================================================\n\n")

# -----------------------------------------------------------------------------
# Verificar pacotes
# -----------------------------------------------------------------------------

required_pkgs <- c("KFAS", "dlm", "MARSS")
missing_pkgs <- character(0)
for (pkg in required_pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        missing_pkgs <- c(missing_pkgs, pkg)
    }
}
if (length(missing_pkgs) > 0) {
    stop(sprintf(
        "Missing R packages: %s. Install with install.packages(c(%s)).",
        paste(missing_pkgs, collapse = ", "),
        paste(shQuote(missing_pkgs), collapse = ", ")
    ))
}

# -----------------------------------------------------------------------------
# Garantir que o working directory seja o diretorio deste script
# -----------------------------------------------------------------------------

script_dir <- tryCatch({
    # Rscript: --file=<path>
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("^--file=", args, value = TRUE)
    if (length(file_arg) > 0) {
        dirname(normalizePath(sub("^--file=", "", file_arg[1])))
    } else if (!is.null(sys.frames()) && length(sys.frames()) > 0) {
        # source()-based
        dirname(normalizePath(sys.frame(1)$ofile))
    } else {
        getwd()
    }
}, error = function(e) getwd())

if (!identical(normalizePath(getwd()), normalizePath(script_dir))) {
    setwd(script_dir)
    cat(sprintf("Working directory set to: %s\n\n", script_dir))
}

# -----------------------------------------------------------------------------
# Executar validacao univariada
# -----------------------------------------------------------------------------

cat("\n-----------------------------------------------------------------------------\n")
cat("  [1/2] Univariate workflow (validate_univariate.R)\n")
cat("-----------------------------------------------------------------------------\n")

uni_ok <- tryCatch({
    source("validate_univariate.R", echo = FALSE)
    TRUE
}, error = function(e) {
    cat(sprintf("\n[ERROR] univariate validation failed: %s\n", e$message))
    FALSE
})

# -----------------------------------------------------------------------------
# Executar validacao multivariada
# -----------------------------------------------------------------------------

cat("\n-----------------------------------------------------------------------------\n")
cat("  [2/2] Multivariate workflow (validate_multivariate.R)\n")
cat("-----------------------------------------------------------------------------\n")

mul_ok <- tryCatch({
    source("validate_multivariate.R", echo = FALSE)
    TRUE
}, error = function(e) {
    cat(sprintf("\n[ERROR] multivariate validation failed: %s\n", e$message))
    FALSE
})

# -----------------------------------------------------------------------------
# Resumo dos arquivos gerados
# -----------------------------------------------------------------------------

cat("\n=============================================================================\n")
cat("  Summary of generated CSV / TXT artefacts\n")
cat("=============================================================================\n")

expected_files <- c(
    # univariate
    "r_univariate_model_comparison.csv",
    "r_univariate_best_params.csv",
    "r_univariate_best_summary.txt",
    "r_univariate_decomposition.csv",
    "r_univariate_diagnostics.csv",
    "r_univariate_residuals.csv",
    "r_univariate_forecast.csv",
    "r_univariate_vs_kalmanbox.csv",
    # multivariate
    "r_multivariate_K_selection.csv",
    "r_multivariate_best_params.csv",
    "r_multivariate_best_summary.txt",
    "r_multivariate_factors.csv",
    "r_multivariate_loadings.csv",
    "r_multivariate_variance_decomp.csv",
    "r_multivariate_vs_kalmanbox.csv"
)

n_ok <- 0
for (f in expected_files) {
    if (file.exists(f)) {
        cat(sprintf("  [OK]      %s\n", f))
        n_ok <- n_ok + 1
    } else {
        cat(sprintf("  [MISSING] %s\n", f))
    }
}

cat(sprintf("\n%d / %d expected files present.\n",
            n_ok, length(expected_files)))

if (uni_ok && mul_ok) {
    cat("\nAll R validations completed successfully.\n")
    invisible(0)
} else {
    cat("\nOne or more validation scripts failed - see errors above.\n")
    quit(status = 1)
}
