# =============================================================================
# Run all R validation scripts for Advanced Filters (FASE6.4)
# Pacotes obrigatorios: KFAS, FKF
# Pacotes opcionais:    dlm
#
# NOTA: R NAO POSSUI pacote CRAN maduro para EKF, UKF ou EnKF. Esta suite
# usa KFAS e FKF como referencias lineares e implementa os filtros nao-
# lineares manualmente para fins de validacao cruzada com o kalmanbox.
#
# Uso (a partir deste diretorio):
#   Rscript run_validation.R
# =============================================================================

cat("=============================================================================\n")
cat("FASE6.4 - Advanced Filters R Validation Suite\n")
cat("  (KFAS + FKF; EKF/UKF/EnKF implementados manualmente em R)\n")
cat("=============================================================================\n\n")

# Verificar pacotes obrigatorios
required_pkgs <- c("KFAS", "FKF")
missing <- character(0)
for (pkg in required_pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) missing <- c(missing, pkg)
}
if (length(missing) > 0) {
    stop(sprintf(
        "Pacotes obrigatorios ausentes: %s. Instale com: install.packages(c(%s))",
        paste(missing, collapse = ", "),
        paste(sprintf("'%s'", missing), collapse = ", ")
    ))
}

# Pacotes opcionais
optional_pkgs <- c("dlm")
for (pkg in optional_pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        cat(sprintf("[info] Pacote opcional '%s' ausente\n", pkg))
    }
}

# Diretorio do runner
script_dir <- tryCatch(
    dirname(normalizePath(sys.frame(1)$ofile)),
    error = function(e) getwd()
)
if (length(script_dir) == 0 || is.na(script_dir)) script_dir <- getwd()
setwd(script_dir)
cat(sprintf("\nWorking directory: %s\n\n", getwd()))

scripts <- c("validate_ekf_ukf.R",
             "validate_sqrt_info.R",
             "validate_enkf.R")
labels  <- c("EKF / UKF (pendulum)",
             "SQRT / Information (linear KF)",
             "EnKF (Lorenz 63)")

for (i in seq_along(scripts)) {
    cat(sprintf("Running %s ...\n", labels[i]))
    cat("---------------------------------------------\n")
    source(scripts[i])
    cat("\n")
}

cat("=============================================================================\n")
cat("All FASE6.4 R validations complete.\n")
cat("=============================================================================\n")

# Lista de arquivos esperados
output_files <- c(
    "results_ekf_ukf_states.csv",
    "results_ekf_ukf_summary.csv",
    "results_sqrt_info_states.csv",
    "results_sqrt_info_summary.csv",
    "results_enkf_states.csv",
    "results_enkf_rmse_vs_N.csv"
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

if (!all_ok) {
    cat("\n[WARN] Alguns arquivos obrigatorios nao foram gerados.\n")
    quit(status = 1)
} else {
    cat("\nTodos os arquivos obrigatorios foram gerados com sucesso.\n")
}

cat("\nLimitacoes do R para filtros avancados:\n")
cat("  - Nao existe pacote CRAN completo para EKF/UKF/EnKF.\n")
cat("  - KFAS e FKF atendem apenas modelos Gaussianos lineares.\n")
cat("  - Implementacoes manuais servem como referencia cruzada, mas sao\n")
cat("    ~1-2 ordens de magnitude mais lentas do que kalmanbox/filterpy.\n")
