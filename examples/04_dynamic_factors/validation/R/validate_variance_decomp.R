# =============================================================================
# Validacao: Decomposicao de Variancia (R^2) do DFM via MARSS
#
# Para cada serie y_i a variancia amostral decompoe-se em:
#   Var(y_i) = sum_{j=1..K} lambda_ij^2 * Var(f_j)          (comum)
#            + 2 * sum_{j<l} lambda_ij*lambda_il*Cov(f_j,f_l)
#            + sigma2_eps_i                                  (idiossincratica)
#
# Com Q=I e fatores aproximadamente ortogonais, o termo cruzado e ~0 e
#   R^2_i (comum) = (sum_j lambda_ij^2 * Var(f_j)) / Var(y_i)
#
# Aqui calculamos:
#   - Var(y_i) amostral (series padronizadas -> ~1 por construcao)
#   - Var(f_j) a partir dos estados suavizados (ou Var teorica = 1/(1-phi^2))
#   - Cov(f_j, f_l) via estados suavizados
#   - Decomposicao R^2 por fator + idiossincratica
#
# Dataset: us_macro_panel.csv com K=2 fatores
# Pacote: MARSS
# =============================================================================

library(MARSS)

panel <- read.csv("../../data/us_macro_panel.csv")
dates <- as.Date(panel$date)
y_mat <- as.matrix(panel[, -1])
series_names <- colnames(y_mat)
N <- ncol(y_mat)
T_obs <- nrow(y_mat)

cat("=============================================================================\n")
cat("  Variance Decomposition (R^2) - MARSS Validation\n")
cat("=============================================================================\n\n")

# Padronizacao: Var(y_i) = 1 por construcao
Y_std_mat <- scale(y_mat, center = TRUE, scale = TRUE)
Y <- t(Y_std_mat)
rownames(Y) <- series_names

# =============================================================================
# Especificacao DFM K=2 (mesma do validate_dfm.R, para consistencia)
# =============================================================================

K <- 2

Z_vals <- vector("list", N * K)
dim(Z_vals) <- c(N, K)
for (i in seq_len(N)) {
    for (j in seq_len(K)) {
        if (i == 1 && j == 2) {
            Z_vals[i, j] <- 0
        } else {
            Z_vals[i, j] <- paste0("z", i, "_", j)
        }
    }
}

model_list <- list(
    B = "diagonal and unequal",
    U = "zero",
    Q = "identity",
    Z = Z_vals,
    A = "zero",
    R = "diagonal and unequal",
    x0 = "zero",
    tinitx = 0
)

cat("Estimando MARSS DFM (K=2) para decomposicao de variancia...\n")
set.seed(42)
fit <- MARSS(
    Y,
    model = model_list,
    control = list(maxit = 500, conv.test.slope.tol = 0.5),
    method = "kem",
    silent = TRUE
)

cat(sprintf("Convergencia: %s | logLik=%.4f\n\n",
            ifelse(fit$convergence == 0, "OK", paste0("code=", fit$convergence)),
            fit$logLik))

# =============================================================================
# Calculo da decomposicao de variancia
# =============================================================================

Lambda_hat <- coef(fit, type = "matrix")$Z          # N x K
factors_smooth <- fit$states                        # K x T
R_diag <- diag(coef(fit, type = "matrix")$R)        # N

# Variancia (amostral) e covariancia dos fatores suavizados
F_cov <- cov(t(factors_smooth))                     # K x K
F_var <- diag(F_cov)                                # K

cat("--- Estatisticas dos fatores suavizados ---\n")
for (j in seq_len(K)) {
    cat(sprintf("  Var(f%d) = %.4f\n", j, F_var[j]))
}
cat(sprintf("  Cov(f1,f2) = %+.4f\n", F_cov[1, 2]))

# Variancia total de cada y_i (com series padronizadas ~ 1)
y_var <- apply(Y_std_mat, 2, var)                   # N

# Contribuicao de cada fator: lambda_ij^2 * Var(f_j)
contrib_factor <- matrix(0, nrow = N, ncol = K)
for (i in seq_len(N)) {
    for (j in seq_len(K)) {
        contrib_factor[i, j] <- Lambda_hat[i, j]^2 * F_var[j]
    }
}

# Termo cruzado (se houver correlacao entre fatores): 2*lambda_i1*lambda_i2*Cov(f1,f2)
cross_term <- rep(0, N)
if (K >= 2) {
    for (i in seq_len(N)) {
        for (j in 1:(K - 1)) {
            for (l in (j + 1):K) {
                cross_term[i] <- cross_term[i] +
                    2 * Lambda_hat[i, j] * Lambda_hat[i, l] * F_cov[j, l]
            }
        }
    }
}

# Variancia total explicada pela componente comum
common_var <- rowSums(contrib_factor) + cross_term
# Variancia idiossincratica estimada
idio_var <- R_diag
# Total implicado pelo modelo
total_implied <- common_var + idio_var

# Shares normalizados pela variancia implicada do modelo
# (garante linhas que somam 1 exatamente)
shares_factor <- sweep(contrib_factor, 1, total_implied, "/")
shares_cross <- cross_term / total_implied
shares_idio <- idio_var / total_implied
r2_common <- 1 - shares_idio

cat("\n--- Decomposicao de variancia (shares da variancia implicada) ---\n")
header <- sprintf("  %-25s%10s%10s%10s%10s",
                  "series", "f1", "f2", "idio", "R^2")
cat(header, "\n")
cat("  ", strrep("-", 65), "\n", sep = "")
for (i in seq_len(N)) {
    cat(sprintf("  %-25s%10.3f%10.3f%10.3f%10.3f\n",
                series_names[i],
                shares_factor[i, 1],
                shares_factor[i, 2],
                shares_idio[i],
                r2_common[i]))
}

mean_r2 <- mean(r2_common)
cat(sprintf("\nMedia R^2 (componente comum): %.4f\n", mean_r2))

# Sanity checks
row_sums <- rowSums(shares_factor) + shares_idio + shares_cross
cat(sprintf("Soma das shares (deve ~1): min=%.6f, max=%.6f\n",
            min(row_sums), max(row_sums)))

# =============================================================================
# Exportacao para CSV
# =============================================================================

decomp_df <- data.frame(
    series = series_names,
    share_f1 = shares_factor[, 1],
    share_f2 = shares_factor[, 2],
    share_cross = shares_cross,
    share_idio = shares_idio,
    r2_common = r2_common,
    row_sum = row_sums
)
write.csv(decomp_df, "results_variance_decomp.csv", row.names = FALSE)
cat("\nSaved: results_variance_decomp.csv\n")

# Cargas (para inspecao)
loadings_df <- data.frame(
    series = series_names,
    lambda_f1 = Lambda_hat[, 1],
    lambda_f2 = Lambda_hat[, 2]
)
write.csv(loadings_df, "results_variance_decomp_loadings.csv", row.names = FALSE)
cat("Saved: results_variance_decomp_loadings.csv\n")

# Sumario
summary_df <- data.frame(
    metric = c("logLik", "AIC", "mean_r2_common", "var_f1", "var_f2",
               "cov_f1_f2", "K", "N", "T"),
    value = c(fit$logLik, fit$AIC, mean_r2, F_var[1], F_var[2],
              F_cov[1, 2], K, N, T_obs)
)
write.csv(summary_df, "results_variance_decomp_summary.csv", row.names = FALSE)
cat("Saved: results_variance_decomp_summary.csv\n")

cat("\nVariance decomposition validation complete.\n")
