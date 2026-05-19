# =============================================================================
# Validacao: Mixed-Frequency DFM com missing data via MARSS
#
# Dataset: mixed_freq_macro.csv
#   - 5 series, T=288 meses
#   - gdp_growth: observado apenas em fim de trimestre (Mar/Jun/Set/Dez)
#   - 4 series mensais: industrial_production, unemployment, cpi, pmi
#
# Modelo:
#   - K = 1 fator latente (atividade economica comum)
#   - MARSS trata NaN no Y como missing -> Kalman skips update naqueles t
#   - Nowcast de GDP = Lambda_gdp * fator_suavizado
#
# Pacote: MARSS
# =============================================================================

library(MARSS)

# ---- Dados ----
mf <- read.csv("../../data/mixed_freq_macro.csv")
dates <- as.Date(mf$date)
y_mat <- as.matrix(mf[, -1])  # remove date
series_names <- colnames(y_mat)
N <- ncol(y_mat)
T_obs <- nrow(y_mat)
gdp_idx <- which(series_names == "gdp_growth")

cat("=============================================================================\n")
cat("  Mixed-Frequency DFM Nowcasting - MARSS Validation\n")
cat("=============================================================================\n\n")

cat(sprintf("Panel: T=%d months x N=%d series\n", T_obs, N))
cat(sprintf("Series: %s\n", paste(series_names, collapse = ", ")))
cat(sprintf("GDP index: %d\n", gdp_idx))
cat("\nMissing por serie:\n")
for (j in seq_len(N)) {
    n_miss <- sum(!is.finite(y_mat[, j]))
    cat(sprintf("  %-25s missing = %d / %d\n", series_names[j], n_miss, T_obs))
}

n_gdp_obs <- sum(is.finite(y_mat[, gdp_idx]))
cat(sprintf("\nGDP observado: %d trimestres (esperado ~%d = 24a x 4q)\n",
            n_gdp_obs, 24 * 4))

# Padronizacao por serie (ignorando NA)
Y_std <- y_mat
for (j in seq_len(N)) {
    col <- y_mat[, j]
    mu <- mean(col, na.rm = TRUE)
    sd_j <- sd(col, na.rm = TRUE)
    Y_std[, j] <- (col - mu) / sd_j
}

# MARSS exige (N x T); converte NaN -> NA (NA e o codigo de missing em MARSS)
Y_marss <- t(Y_std)
Y_marss[!is.finite(Y_marss)] <- NA
rownames(Y_marss) <- series_names

cat(sprintf("\nTotal NA em Y_marss: %d\n", sum(is.na(Y_marss))))

# =============================================================================
# Especificacao DFM K=1
# =============================================================================

K <- 1

# Z (Lambda) N x 1: todas livres, sem restricoes triangulares (K=1 -> apenas
# escala precisa de fixar; usamos Q="identity")
Z_vals <- matrix(list(), nrow = N, ncol = K)
for (i in seq_len(N)) {
    Z_vals[i, 1] <- paste0("z", i)
}

model_list <- list(
    B = "diagonal and unequal",   # AR(1) no fator
    U = "zero",
    Q = "identity",                # escala do fator fixada
    Z = Z_vals,
    A = "zero",
    R = "diagonal and unequal",    # idiossincratica
    x0 = "zero",
    tinitx = 0
)

cat("\n--- Estimando MARSS DFM K=1 com missing data ---\n")
set.seed(42)
fit <- MARSS(
    Y_marss,
    model = model_list,
    control = list(maxit = 500, conv.test.slope.tol = 0.5),
    method = "kem",
    silent = TRUE
)

cat(sprintf("Convergencia: %s\n",
            ifelse(fit$convergence == 0, "OK", paste0("code=", fit$convergence))))
cat(sprintf("Log-likelihood: %.4f\n", fit$logLik))
cat(sprintf("AIC:            %.4f\n", fit$AIC))

# =============================================================================
# Extracao do fator e cargas
# =============================================================================

factor_smooth <- as.numeric(fit$states[1, ])      # T
factor_se <- as.numeric(fit$states.se[1, ])        # T
Lambda_hat <- coef(fit, type = "matrix")$Z         # N x 1
phi_hat <- diag(coef(fit, type = "matrix")$B)[1]
R_diag <- diag(coef(fit, type = "matrix")$R)

cat(sprintf("\nphi (AR1 fator) = %.4f\n", phi_hat))
cat("\nCargas estimadas:\n")
for (i in seq_len(N)) {
    cat(sprintf("  %-25s lambda = %+.4f\n", series_names[i], Lambda_hat[i, 1]))
}

# =============================================================================
# Nowcast de GDP
# =============================================================================

lambda_gdp <- Lambda_hat[gdp_idx, 1]
gdp_nowcast_std <- lambda_gdp * factor_smooth

# Reverter padronizacao para escala original
gdp_mu <- mean(y_mat[, gdp_idx], na.rm = TRUE)
gdp_sd <- sd(y_mat[, gdp_idx], na.rm = TRUE)
gdp_nowcast <- gdp_nowcast_std * gdp_sd + gdp_mu

# RMSE nas observacoes (fim de trimestre)
gdp_obs_idx <- which(is.finite(y_mat[, gdp_idx]))
gdp_obs <- y_mat[gdp_obs_idx, gdp_idx]
gdp_pred <- gdp_nowcast[gdp_obs_idx]
rmse_dfm <- sqrt(mean((gdp_obs - gdp_pred)^2))
naive_mean <- mean(gdp_obs)
rmse_mean <- sqrt(mean((gdp_obs - naive_mean)^2))

cat(sprintf("\nRMSE DFM nowcast (quarter-end): %.4f\n", rmse_dfm))
cat(sprintf("RMSE naive mean:                 %.4f\n", rmse_mean))
cat(sprintf("Improvement:                     %.1f%%\n",
            100 * (rmse_mean - rmse_dfm) / rmse_mean))

# Correlacao com GDP observado
corr_gdp <- abs(cor(gdp_pred, gdp_obs))
cat(sprintf("|corr(nowcast, GDP observado)|:  %.4f\n", corr_gdp))

# =============================================================================
# Exportacao para CSV
# =============================================================================

# 1. Nowcast com datas
nowcast_out <- data.frame(
    date = dates,
    factor = factor_smooth,
    factor_se = factor_se,
    gdp_nowcast = gdp_nowcast,
    gdp_observed = y_mat[, gdp_idx],
    is_quarter_end = is.finite(y_mat[, gdp_idx])
)
write.csv(nowcast_out, "results_mixed_freq_nowcast.csv", row.names = FALSE)
cat("\nSaved: results_mixed_freq_nowcast.csv\n")

# 2. Cargas
loadings_mf <- data.frame(
    series = series_names,
    lambda = as.numeric(Lambda_hat),
    sigma2_eps = R_diag
)
write.csv(loadings_mf, "results_mixed_freq_loadings.csv", row.names = FALSE)
cat("Saved: results_mixed_freq_loadings.csv\n")

# 3. Sumario com metricas de nowcast
nowcast_summary <- data.frame(
    metric = c("logLik", "AIC", "phi", "lambda_gdp",
               "rmse_dfm", "rmse_naive_mean", "rel_improvement_pct",
               "corr_nowcast_gdp", "n_gdp_obs", "convergence"),
    value = c(fit$logLik, fit$AIC, phi_hat, lambda_gdp,
              rmse_dfm, rmse_mean,
              100 * (rmse_mean - rmse_dfm) / rmse_mean,
              corr_gdp, n_gdp_obs, fit$convergence)
)
write.csv(nowcast_summary, "results_mixed_freq_summary.csv", row.names = FALSE)
cat("Saved: results_mixed_freq_summary.csv\n")

cat("\nMixed-frequency MARSS validation complete.\n")
