# =============================================================================
# Runner: executa todos os scripts de validacao R para FASE 8 (Diagnostics)
# Pacotes necessarios: KFAS, tseries, strucchange
#
# Uso: Rscript run_validation.R
#
# Espera-se que este script seja executado a partir do diretorio
# examples/08_diagnostics/validation/R/ para que os caminhos relativos
# aos arquivos de dados (../../data/*.csv) resolvam corretamente.
# =============================================================================

cat("=============================================================================\n")
cat("  FASE 8 - R Validation Suite (Diagnostics)\n")
cat("  Backends: KFAS + tseries + strucchange\n")
cat("=============================================================================\n\n")

# ---------------------------------------------------------------------------
# Checagem de pacotes
# ---------------------------------------------------------------------------
required_pkgs <- c("KFAS", "tseries", "strucchange")
missing_pkgs  <- required_pkgs[
    !vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_pkgs) > 0) {
    stop(sprintf(
        "Missing R packages: %s\n  Install with: install.packages(c(%s))",
        paste(missing_pkgs, collapse = ", "),
        paste(sprintf("'%s'", missing_pkgs), collapse = ", ")
    ))
}

# Mudar para o diretorio deste script (robustez quando chamado de outro lugar)
this_file <- tryCatch(
    sys.frame(1)$ofile,
    error = function(e) NULL
)
if (is.null(this_file) || !nzchar(this_file)) {
    # Rscript: usar commandArgs
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("--file=", args, value = TRUE)
    if (length(file_arg) > 0) {
        this_file <- sub("--file=", "", file_arg[1])
    }
}
if (!is.null(this_file) && nzchar(this_file)) {
    setwd(dirname(normalizePath(this_file)))
}
cat(sprintf("Working directory: %s\n\n", getwd()))

# ---------------------------------------------------------------------------
# Helper: executar um script e medir o tempo
# ---------------------------------------------------------------------------
run_script <- function(path) {
    cat(sprintf("\n>>> %s\n", path))
    cat(paste(rep("-", 77), collapse = ""), "\n", sep = "")
    t0 <- Sys.time()
    ok <- tryCatch({
        source(path, local = new.env())
        TRUE
    }, error = function(e) {
        cat(sprintf("[ERROR] %s failed: %s\n", path,
                    conditionMessage(e)))
        FALSE
    })
    t1 <- Sys.time()
    cat(sprintf("<<< %s (%.1fs)\n", path,
                as.numeric(difftime(t1, t0, units = "secs"))))
    invisible(ok)
}

# ---------------------------------------------------------------------------
# Executar validacoes
# ---------------------------------------------------------------------------
ok_res <- run_script("validate_residual_diagnostics.R")
ok_miss <- run_script("validate_missing_bootstrap.R")

# ---------------------------------------------------------------------------
# Relatorio de arquivos gerados
# ---------------------------------------------------------------------------
cat("\n=============================================================================\n")
cat("  Generated CSV files\n")
cat("=============================================================================\n")

output_files <- c(
    "results_residual_diagnostics_residuals.csv",
    "results_residual_diagnostics_auxiliary.csv",
    "results_residual_diagnostics_tests.csv",
    "results_residual_diagnostics_model.csv",
    "results_missing_bootstrap_params.csv",
    "results_missing_bootstrap_bands.csv",
    "results_missing_bootstrap_imputation.csv",
    "results_missing_bootstrap_summary.csv"
)
for (f in output_files) {
    if (file.exists(f)) {
        sz <- file.info(f)$size
        cat(sprintf("  [OK]      %-55s (%d bytes)\n", f, sz))
    } else {
        cat(sprintf("  [MISSING] %s\n", f))
    }
}

cat("\n=============================================================================\n")
if (ok_res && ok_miss) {
    cat("  All R validations completed successfully.\n")
} else {
    cat("  One or more validations FAILED - see errors above.\n")
}
cat("=============================================================================\n")

if (!(ok_res && ok_miss)) {
    quit(status = 1)
}
