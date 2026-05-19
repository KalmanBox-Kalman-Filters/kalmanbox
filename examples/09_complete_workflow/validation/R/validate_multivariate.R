# =============================================================================
# Validacao: Complete multivariate workflow - US macro panel
# Pacote: MARSS
#
# Workflow replicado do notebook 02_multivariate_workflow.ipynb:
#   1. Carregar us_macro_panel.csv (T x 15 series mensais).
#   2. Padronizar (mean=0, sd=1) cada serie.
#   3. Ajustar Dynamic Factor Model com K = 1, 2, 3 fatores.
#   4. Selecionar K via BIC (menor BIC).
#   5. Extrair fatores suavizados, cargas (Lambda), variance decomposition.
#   6. Exportar tudo em CSV para comparacao com kalmanbox.
#
# Especificacao DFM:
#   x_t = B * x_{t-1} + w_t,   w_t ~ N(0, I_K)     (fatores AR(1) indep.)
#   y_t = Z * x_t + v_t,       v_t ~ N(0, R)       (R diagonal)
# Identificacao: triangular superior zero em Z nas primeiras K linhas.
# =============================================================================

library(MARSS)

# -----------------------------------------------------------------------------
# 0. Utilitarios
# -----------------------------------------------------------------------------

build_Z_spec <- function(N, K) {
    # Matriz N x K: Z[i, j] livre, excepto Z[i, j] = 0 para j > i (i <= K-1).
    # Isto impoe identificacao triangular nas primeiras K-1 linhas.
    Z <- matrix(list(), N, K)
    for (i in seq_len(N)) {
        for (j in seq_len(K)) {
            if (i < j) {
                Z[[i, j]] <- 0
            } else {
                Z[[i, j]] <- paste0("z", i, "_", j)
            }
        }
    }
    Z
}

fit_dfm <- function(Y, K, maxit = 500, seed = 42) {
    N <- nrow(Y)
    model_list <- list(
        B = "diagonal and unequal",
        U = "zero",
        Q = "identity",
        Z = build_Z_spec(N, K),
        A = "zero",
        R = "diagonal and unequal",
        x0 = "zero",
        tinitx = 0
    )
    set.seed(seed)
    MARSS(
        Y,
        model = model_list,
        control = list(maxit = maxit, conv.test.slope.tol = 0.5,
                       allow.degen = FALSE, trace = -1),
        method = "kem",
        silent = TRUE
    )
}

marss_bic <- function(fit, n_obs) {
    # MARSS retorna $AIC mas nem sempre $BIC; computamos se faltar.
    ll <- fit$logLik
    k <- fit$num.params
    -2 * ll + k * log(n_obs)
}

# -----------------------------------------------------------------------------
# 1. Dados
# -----------------------------------------------------------------------------

panel <- read.csv("../../data/us_macro_panel.csv")
dates <- as.Date(panel$date)
y_mat <- as.matrix(panel[, -1])
series_names <- colnames(y_mat)
N <- ncol(y_mat)
T_obs <- nrow(y_mat)

cat("=============================================================================\n")
cat("  Multivariate workflow validation - US macro panel (DFM via MARSS)\n")
cat("=============================================================================\n\n")
cat(sprintf("Panel dimensions: T=%d x N=%d\n", T_obs, N))
cat(sprintf("Date range: %s -> %s\n", dates[1], dates[T_obs]))
cat(sprintf("Series: %s\n", paste(series_names, collapse = ", ")))

# Padronizar cada serie (consistente com StandardScaler do notebook)
y_means <- colMeans(y_mat, na.rm = TRUE)
y_sds <- apply(y_mat, 2, sd, na.rm = TRUE)
y_std <- sweep(y_mat, 2, y_means, "-")
y_std <- sweep(y_std, 2, y_sds, "/")

# MARSS espera N x T (series nas linhas)
Y <- t(y_std)
rownames(Y) <- series_names

# -----------------------------------------------------------------------------
# 2. Ajustar DFM com K = 1, 2, 3
# -----------------------------------------------------------------------------

fits <- list()
K_values <- c(1, 2, 3)
for (K in K_values) {
    cat(sprintf("\n--- Fitting DFM with K = %d ---\n", K))
    fit <- tryCatch(
        fit_dfm(Y, K = K),
        error = function(e) {
            warning(sprintf("DFM K=%d failed: %s", K, e$message))
            NULL
        }
    )
    fits[[as.character(K)]] <- fit
    if (!is.null(fit)) {
        cat(sprintf("  convergence = %s\n",
                    ifelse(fit$convergence == 0, "OK",
                           paste0("code=", fit$convergence))))
        cat(sprintf("  logLik = %.4f, AIC = %.4f, k = %d\n",
                    fit$logLik, fit$AIC, fit$num.params))
    }
}

# -----------------------------------------------------------------------------
# 3. Tabela de selecao por BIC
# -----------------------------------------------------------------------------

K_selection <- do.call(rbind, lapply(K_values, function(K) {
    fit <- fits[[as.character(K)]]
    if (is.null(fit)) {
        data.frame(K = K, k_params = NA, loglik = NA, aic = NA, bic = NA)
    } else {
        bic <- marss_bic(fit, n_obs = T_obs * N)
        data.frame(
            K = K,
            k_params = fit$num.params,
            loglik = fit$logLik,
            aic = fit$AIC,
            bic = bic
        )
    }
}))

cat("\nK selection table:\n")
print(K_selection, row.names = FALSE, digits = 6)

write.csv(K_selection, "r_multivariate_K_selection.csv", row.names = FALSE)
cat("Saved: r_multivariate_K_selection.csv\n")

best_K_idx <- which.min(K_selection$bic)
best_K <- K_selection$K[best_K_idx]
best_fit <- fits[[as.character(best_K)]]

cat(sprintf("\nSelected best K (min BIC): K = %d\n", best_K))

# -----------------------------------------------------------------------------
# 4. Fatores suavizados + erros padrao
# -----------------------------------------------------------------------------

factors_smoothed <- best_fit$states       # K x T
factors_se <- best_fit$states.se          # K x T

factors_df <- data.frame(date = as.character(dates))
for (k in seq_len(best_K)) {
    factors_df[[paste0("factor_", k)]] <- factors_smoothed[k, ]
}
for (k in seq_len(best_K)) {
    factors_df[[paste0("factor_", k, "_se")]] <- factors_se[k, ]
}
write.csv(factors_df, "r_multivariate_factors.csv", row.names = FALSE)
cat("Saved: r_multivariate_factors.csv\n")

# -----------------------------------------------------------------------------
# 5. Cargas (Lambda = Z), coeficientes AR(1), variancias idiossincraticas
# -----------------------------------------------------------------------------

coef_mat <- coef(best_fit, type = "matrix")
Lambda_hat <- coef_mat$Z       # N x K
phi_hat <- diag(coef_mat$B)    # K
R_diag <- diag(coef_mat$R)     # N

loadings_df <- data.frame(series = series_names)
for (k in seq_len(best_K)) {
    loadings_df[[paste0("f", k)]] <- Lambda_hat[, k]
}
write.csv(loadings_df, "r_multivariate_loadings.csv", row.names = FALSE)
cat("Saved: r_multivariate_loadings.csv\n")

# Parametros do melhor modelo (estilo kalmanbox: "parameter,estimate")
params_list <- list()
for (i in seq_len(N)) {
    for (k in seq_len(best_K)) {
        params_list[[length(params_list) + 1]] <- data.frame(
            parameter = sprintf("lambda_%s_f%d", series_names[i], k - 1),
            estimate = Lambda_hat[i, k]
        )
    }
}
for (k in seq_len(best_K)) {
    params_list[[length(params_list) + 1]] <- data.frame(
        parameter = sprintf("phi_f%d", k - 1),
        estimate = phi_hat[k]
    )
}
for (i in seq_len(N)) {
    params_list[[length(params_list) + 1]] <- data.frame(
        parameter = sprintf("sigma2_eps_%s", series_names[i]),
        estimate = R_diag[i]
    )
}
best_params_df <- do.call(rbind, params_list)
write.csv(best_params_df, "r_multivariate_best_params.csv", row.names = FALSE)
cat("Saved: r_multivariate_best_params.csv\n")

# -----------------------------------------------------------------------------
# 6. Variance decomposition
#
# Para cada serie i:
#   Var(y_i) = sum_k Lambda_{ik}^2 * Var(f_k) + R_ii
# Os fatores estao com Var estimada pelos estados suavizados; podemos usar
# var(factors_smoothed[k, ]) como Var(f_k) (amostral).
# -----------------------------------------------------------------------------

factor_var <- apply(factors_smoothed, 1, var)   # K

var_decomp <- matrix(NA, nrow = N, ncol = best_K + 2)
colnames(var_decomp) <- c(paste0("factor_", seq_len(best_K)),
                          "idiosyncratic", "R2")
rownames(var_decomp) <- series_names

for (i in seq_len(N)) {
    # Como as series sao padronizadas (Var=1), os shares somam ~1
    shares <- numeric(best_K)
    for (k in seq_len(best_K)) {
        shares[k] <- (Lambda_hat[i, k]^2) * factor_var[k]
    }
    idio <- R_diag[i]
    total <- sum(shares) + idio
    # Normaliza para somar 1 (em caso de divergencia numerica leve)
    if (total > 0) {
        shares_n <- shares / total
        idio_n <- idio / total
    } else {
        shares_n <- shares
        idio_n <- idio
    }
    r2 <- sum(shares_n)
    var_decomp[i, seq_len(best_K)] <- shares_n
    var_decomp[i, best_K + 1] <- idio_n
    var_decomp[i, best_K + 2] <- r2
}

var_decomp_df <- data.frame(series = series_names,
                            as.data.frame(var_decomp),
                            check.names = FALSE)
rownames(var_decomp_df) <- NULL

cat("\nVariance decomposition (best K):\n")
print(var_decomp_df, row.names = FALSE, digits = 4)

write.csv(var_decomp_df, "r_multivariate_variance_decomp.csv", row.names = FALSE)
cat("Saved: r_multivariate_variance_decomp.csv\n")

# -----------------------------------------------------------------------------
# 7. Sumario textual do melhor DFM
# -----------------------------------------------------------------------------

best_bic <- K_selection$bic[best_K_idx]
summary_lines <- c(
    strrep("=", 70),
    sprintf("  Best DFM (R / MARSS): K = %d", best_K),
    strrep("=", 70),
    sprintf("Log-Likelihood:    %12.4f", best_fit$logLik),
    sprintf("AIC:               %12.4f", best_fit$AIC),
    sprintf("BIC:               %12.4f", best_bic),
    sprintf("k_params:          %12d", best_fit$num.params),
    sprintf("T (obs):           %12d", T_obs),
    sprintf("N (series):        %12d", N),
    sprintf("Convergence:       %12s",
            ifelse(best_fit$convergence == 0, "OK",
                   paste0("code=", best_fit$convergence))),
    "",
    "AR(1) coefficients of factors:"
)
for (k in seq_len(best_K)) {
    summary_lines <- c(summary_lines,
                       sprintf("  phi_%d = %.4f", k, phi_hat[k]))
}
writeLines(summary_lines, "r_multivariate_best_summary.txt")
cat("Saved: r_multivariate_best_summary.txt\n")

# -----------------------------------------------------------------------------
# 8. Tabela de comparacao kalmanbox vs R
# -----------------------------------------------------------------------------

kbox_K_path <- "../../output/multivariate_K_selection.csv"
if (file.exists(kbox_K_path)) {
    kbox <- read.csv(kbox_K_path)
    merged <- merge(
        kbox[, c("K", "loglik", "aic", "bic")],
        K_selection[, c("K", "loglik", "aic", "bic")],
        by = "K",
        suffixes = c("_kalmanbox", "_R")
    )
    merged$d_loglik <- merged$loglik_R - merged$loglik_kalmanbox
    merged$d_aic <- merged$aic_R - merged$aic_kalmanbox
    merged$d_bic <- merged$bic_R - merged$bic_kalmanbox
    write.csv(merged, "r_multivariate_vs_kalmanbox.csv", row.names = FALSE)
    cat("Saved: r_multivariate_vs_kalmanbox.csv\n")

    cat("\nkalmanbox vs MARSS comparison (K selection):\n")
    print(merged, row.names = FALSE, digits = 5)
}

cat("\nMultivariate R validation complete.\n")
