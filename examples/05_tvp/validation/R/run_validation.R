# =============================================================================
# Run all R validation scripts for TVP (FASE5.4)
# Pacotes obrigatorios: dlm
# Pacotes opcionais:    bvarsv (TVP-VAR Bayesiano Primiceri 2005)
#
# Uso (a partir deste diretorio):
#   Rscript run_validation.R
# =============================================================================

cat("=============================================================================\n")
cat("FASE5.4 - TVP R Validation Suite (dlm + bvarsv)\n")
cat("=============================================================================\n\n")

# Verificar pacotes obrigatorios
required_pkgs <- c("dlm")
for (pkg in required_pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        stop(sprintf(
            "Pacote obrigatorio '%s' nao instalado. Instale com: install.packages('%s')",
            pkg, pkg
        ))
    }
}

# Pacotes opcionais
optional_pkgs <- c("bvarsv")
for (pkg in optional_pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        cat(sprintf("[info] Pacote opcional '%s' ausente - algumas validacoes serao puladas\n", pkg))
    }
}

# Garante que os scripts e os CSV de saida fiquem no diretorio do runner
script_dir <- tryCatch(
    dirname(normalizePath(sys.frame(1)$ofile)),
    error = function(e) getwd()
)
if (length(script_dir) == 0 || is.na(script_dir)) script_dir <- getwd()
setwd(script_dir)
cat(sprintf("\nWorking directory: %s\n\n", getwd()))

# -----------------------------------------------------------------------------
# 1) TVP Regression (dlm)
# -----------------------------------------------------------------------------
cat("Running TVP Regression validation...\n")
cat("---------------------------------------------\n")
source("validate_tvp_regression.R")
cat("\n")

# -----------------------------------------------------------------------------
# 2) Phillips Curve (dlm + bvarsv opcional)
# -----------------------------------------------------------------------------
cat("Running Phillips Curve validation...\n")
cat("---------------------------------------------\n")
source("validate_phillips_curve.R")
cat("\n")

cat("=============================================================================\n")
cat("All TVP R validations complete.\n")
cat("=============================================================================\n")

# Lista de arquivos esperados
output_files <- c(
    "results_tvp_regression_coefs.csv",
    "results_tvp_regression_summary.csv",
    "results_phillips_curve_dlm.csv",
    "results_phillips_curve_summary.csv"
)
optional_files <- c(
    "results_phillips_curve_bvarsv.csv"
)

cat("\nGenerated files:\n")
all_ok <- TRUE
for (f in output_files) {
    if (file.exists(f)) {
        cat(sprintf("  [OK]       %s\n", f))
    } else {
        cat(sprintf("  [MISSING]  %s\n", f))
        all_ok <- FALSE
    }
}
for (f in optional_files) {
    if (file.exists(f)) {
        cat(sprintf("  [OK opt]   %s\n", f))
    } else {
        cat(sprintf("  [skip opt] %s (bvarsv indisponivel)\n", f))
    }
}

if (!all_ok) {
    cat("\n[WARN] Alguns arquivos obrigatorios nao foram gerados.\n")
} else {
    cat("\nTodos os arquivos obrigatorios foram gerados.\n")
}
