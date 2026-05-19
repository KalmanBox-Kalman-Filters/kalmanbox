# =============================================================================
# Run all R validation scripts for Bayesian SSM (FASE7.4)
#
# Pacotes obrigatorios: dlm, coda, KFAS
#
# NOTA: dlm::dlmGibbsDIG e a referencia CRAN mais direta para Gibbs+FFBS
# em modelos DLM. KFAS e usado apenas para o comparativo MLE (fitSSM).
#
# Uso (a partir deste diretorio):
#   Rscript run_validation.R
# =============================================================================

cat("=============================================================================\n")
cat("FASE7.4 - Bayesian SSM R Validation Suite\n")
cat("  (dlm::dlmGibbsDIG + coda diagnostics + KFAS::fitSSM para MLE)\n")
cat("=============================================================================\n\n")

# Verificar pacotes obrigatorios
required_pkgs <- c("dlm", "coda", "KFAS")
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

# Diretorio do runner
script_dir <- tryCatch(
    dirname(normalizePath(sys.frame(1)$ofile)),
    error = function(e) getwd()
)
if (length(script_dir) == 0 || is.na(script_dir)) script_dir <- getwd()
setwd(script_dir)
cat(sprintf("\nWorking directory: %s\n\n", getwd()))

scripts <- c("validate_gibbs_ffbs.R",
             "validate_mcmc_diagnostics.R")
labels  <- c("Gibbs + FFBS (Nile)",
             "MCMC diagnostics (4 chains)")

for (i in seq_along(scripts)) {
    cat(sprintf("Running %s ...\n", labels[i]))
    cat("---------------------------------------------\n")
    source(scripts[i])
    cat("\n")
}

cat("=============================================================================\n")
cat("All FASE7.4 R validations complete.\n")
cat("=============================================================================\n")

# Lista de arquivos esperados
output_files <- c(
    "results_gibbs_ffbs_draws.csv",
    "results_gibbs_ffbs_states.csv",
    "results_gibbs_ffbs_summary.csv",
    "results_mcmc_diagnostics.csv",
    "results_mcmc_draws.csv",
    "results_bayes_vs_mle.csv"
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

cat("\nNotas:\n")
cat("  - dlm e a referencia Bayesian direta (FFBS exato + IG conjugado).\n")
cat("  - Priors IG(1,1) nas variancias: weakly-informative, consistentes\n")
cat("    com os priors usados em kalmanbox.estimation.bayesian.\n")
cat("  - R-hat e ESS calculados com coda::gelman.diag e effectiveSize.\n")
cat("  - MLE de referencia via KFAS::fitSSM para Bernstein-von Mises check.\n")
