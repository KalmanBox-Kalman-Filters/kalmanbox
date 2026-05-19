# =============================================================================
# Validacao: Missing data + simulation smoother (BSM no Airline)
# Pacotes: KFAS
#
# Etapas:
#   1. Ajustar BSM (nivel + tendencia + sazonalidade 12) no log(airline) com
#      serie completa e serie com NAs; comparar parametros estimados.
#   2. Rodar KFAS::simulateSSM para gerar N draws do estado suavizado e
#      construir bandas de incerteza via simulation smoother (Durbin-Koopman,
#      2002). Comparar com as bandas analiticas do Kalman smoother.
#
# O mesmo fluxo e executado no Python (sol_02_missing_data_bootstrap.py)
# para comparacao cruzada. KFAS trata NAs na serie de forma nativa.
#
# Referencias:
#   Durbin & Koopman (2002), 'A simple and efficient simulation smoother
#     for state space time series analysis', Biometrika 89(3): 603-616
#   Durbin & Koopman (2012), cap. 7
# =============================================================================

suppressPackageStartupMessages({
    library(KFAS)
})

set.seed(20260416)

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
airline      <- read.csv("../../data/airline.csv")
airline_miss <- read.csv("../../data/airline_missing.csv")

y_full <- log(airline$passengers)
# airline_missing.csv tem strings vazias para NAs; read.csv ja converte
# para NA em colunas numericas, mas garantimos o tipo explicitamente.
y_miss <- suppressWarnings(as.numeric(airline_miss$passengers))
y_miss <- log(y_miss)

n         <- length(y_full)
n_missing <- sum(is.na(y_miss))

cat("=============================================================================\n")
cat("  Missing data + Simulation smoother - BSM on log(airline)\n")
cat("=============================================================================\n\n")
cat(sprintf("Observations   : %d\n", n))
cat(sprintf("Missing points : %d (%.1f%%)\n", n_missing,
            100 * n_missing / n))

# ---------------------------------------------------------------------------
# 1. BSM via KFAS: serie completa
# ---------------------------------------------------------------------------
cat("\n--- 1. BSM fit: complete series ---\n")
model_full <- SSModel(
    y_full ~ SSMtrend(degree = 2, Q = list(NA, NA)) +
             SSMseasonal(period = 12, Q = NA),
    H = NA
)
# 4 parametros: H (obs), Q_level, Q_trend, Q_seasonal
fit_full <- fitSSM(model_full,
                   inits  = log(c(1e-3, 1e-3, 1e-5, 1e-3)),
                   method = "BFGS")
if (fit_full$optim.out$convergence != 0) {
    warning("KFAS BSM (complete) did not converge!")
}
kfs_full <- KFS(fit_full$model,
                filtering = "state", smoothing = c("state", "disturbance"))

H_full    <- as.numeric(fit_full$model$H[1])
Q_full    <- diag(fit_full$model$Q[, , 1])   # (level, trend, seas)
logL_full <- as.numeric(logLik(fit_full$model))

cat(sprintf("sigma2_obs     = %.6g\n", H_full))
cat(sprintf("sigma2_level   = %.6g\n", Q_full[1]))
cat(sprintf("sigma2_trend   = %.6g\n", Q_full[2]))
cat(sprintf("sigma2_seasonal= %.6g\n", Q_full[3]))
cat(sprintf("logLik         = %.4f\n", logL_full))

# ---------------------------------------------------------------------------
# 2. BSM via KFAS: serie com NAs
# ---------------------------------------------------------------------------
cat("\n--- 2. BSM fit: series with missing values ---\n")
model_miss <- SSModel(
    y_miss ~ SSMtrend(degree = 2, Q = list(NA, NA)) +
             SSMseasonal(period = 12, Q = NA),
    H = NA
)
fit_miss <- fitSSM(model_miss,
                   inits  = log(c(1e-3, 1e-3, 1e-5, 1e-3)),
                   method = "BFGS")
if (fit_miss$optim.out$convergence != 0) {
    warning("KFAS BSM (missing) did not converge!")
}
kfs_miss <- KFS(fit_miss$model,
                filtering = "state", smoothing = c("state", "disturbance"))

H_miss    <- as.numeric(fit_miss$model$H[1])
Q_miss    <- diag(fit_miss$model$Q[, , 1])
logL_miss <- as.numeric(logLik(fit_miss$model))

cat(sprintf("sigma2_obs     = %.6g\n", H_miss))
cat(sprintf("sigma2_level   = %.6g\n", Q_miss[1]))
cat(sprintf("sigma2_trend   = %.6g\n", Q_miss[2]))
cat(sprintf("sigma2_seasonal= %.6g\n", Q_miss[3]))
cat(sprintf("logLik         = %.4f\n", logL_miss))

# ---------------------------------------------------------------------------
# 3. Comparacao dos parametros (complete vs missing)
# ---------------------------------------------------------------------------
cat("\n--- 3. Parameter comparison (complete vs missing) ---\n")
params_df <- data.frame(
    parameter       = c("sigma2_obs", "sigma2_level",
                        "sigma2_trend", "sigma2_seasonal"),
    complete        = c(H_full, Q_full[1], Q_full[2], Q_full[3]),
    with_missing    = c(H_miss, Q_miss[1], Q_miss[2], Q_miss[3]),
    abs_diff        = c(abs(H_full - H_miss),
                        abs(Q_full[1] - Q_miss[1]),
                        abs(Q_full[2] - Q_miss[2]),
                        abs(Q_full[3] - Q_miss[3]))
)
params_df$rel_diff_pct <- 100 * params_df$abs_diff /
    pmax(abs(params_df$complete), .Machine$double.eps)
print(params_df, row.names = FALSE)

# Smoothed level agreement (signal the user consumes)
level_full <- as.numeric(kfs_full$alphahat[, "level"])
level_miss <- as.numeric(kfs_miss$alphahat[, "level"])
obs_mask   <- !is.na(y_miss)
level_diff <- mean(abs(level_full[obs_mask] - level_miss[obs_mask]))
level_scale <- mean(abs(level_full[obs_mask]))
cat(sprintf("\nSmoothed level agreement (observed points):\n"))
cat(sprintf("  mean |level_full - level_miss| = %.6f\n", level_diff))
cat(sprintf("  mean |level_full|              = %.6f\n", level_scale))
cat(sprintf("  relative difference            = %.4f\n",
            level_diff / level_scale))

# ---------------------------------------------------------------------------
# 4. Imputacao dos pontos faltantes (smoothed signal)
# ---------------------------------------------------------------------------
cat("\n--- 4. Missing imputation (smoother signal) ---\n")
# signal() extrai a media observacional E[y|Y^n] e sua variancia
sig_miss <- signal(kfs_miss, states = "all")
y_hat    <- as.numeric(sig_miss$signal)
y_hat_v  <- as.numeric(sig_miss$variance)

miss_mask     <- is.na(y_miss)
imputed_vals  <- y_hat[miss_mask]
imputed_vars  <- y_hat_v[miss_mask]
truth_vals    <- y_full[miss_mask]

rmse <- sqrt(mean((imputed_vals - truth_vals)^2))
mae  <- mean(abs(imputed_vals - truth_vals))
se95 <- 1.96 * sqrt(imputed_vars)
coverage <- mean(abs(imputed_vals - truth_vals) <= se95)
cat(sprintf("  RMSE = %.6f, MAE = %.6f, 95%% CI coverage = %.1f%%\n",
            rmse, mae, 100 * coverage))

# ---------------------------------------------------------------------------
# 5. Simulation smoother (Durbin-Koopman 2002) via simulateSSM
#    type = "states" -> draws do vetor de estado suavizado
# ---------------------------------------------------------------------------
cat("\n--- 5. simulateSSM: state draws ---\n")
n_sim <- 200
sim_draws <- simulateSSM(fit_full$model,
                         type       = "states",
                         nsim       = n_sim,
                         antithetics = FALSE)
# sim_draws tem shape (n, n_states, n_sim)
level_idx <- which(colnames(kfs_full$alphahat) == "level")
sim_level <- sim_draws[, level_idx, ]   # (n, n_sim)
sim_low   <- apply(sim_level, 1, quantile, probs = 0.025)
sim_high  <- apply(sim_level, 1, quantile, probs = 0.975)

# Bandas analiticas do Kalman smoother
level_var_full <- as.numeric(kfs_full$V[level_idx, level_idx, ])
level_se_full  <- sqrt(level_var_full)
kalman_low  <- level_full - 1.96 * level_se_full
kalman_high <- level_full + 1.96 * level_se_full

kalman_width <- mean(kalman_high - kalman_low)
sim_width    <- mean(sim_high   - sim_low)
width_ratio  <- sim_width / kalman_width

cat(sprintf("  draws                    = %d\n", n_sim))
cat(sprintf("  Kalman 95%% mean width    = %.6f\n", kalman_width))
cat(sprintf("  simulation 95%% mean width= %.6f\n", sim_width))
cat(sprintf("  width ratio (sim/Kalman) = %.4f\n", width_ratio))

# ---------------------------------------------------------------------------
# 6. Exportar resultados
# ---------------------------------------------------------------------------
cat("\n--- 6. Exporting results ---\n")

# 6a. Parametros (complete vs missing)
params_out <- params_df
params_out$loglik_complete <- c(logL_full, NA, NA, NA)
params_out$loglik_missing  <- c(logL_miss, NA, NA, NA)
write.csv(params_out, "results_missing_bootstrap_params.csv",
          row.names = FALSE)
cat("  saved: results_missing_bootstrap_params.csv\n")

# 6b. Smoothed level e bandas
bands_df <- data.frame(
    index        = seq_len(n),
    month        = airline$month,
    y_full       = y_full,
    y_miss       = y_miss,
    level_full   = level_full,
    level_miss   = level_miss,
    kalman_low   = kalman_low,
    kalman_high  = kalman_high,
    sim_low      = sim_low,
    sim_high     = sim_high
)
write.csv(bands_df, "results_missing_bootstrap_bands.csv",
          row.names = FALSE)
cat("  saved: results_missing_bootstrap_bands.csv\n")

# 6c. Imputacao (ponto-a-ponto, observacoes faltantes)
imp_df <- data.frame(
    index      = which(miss_mask),
    month      = airline$month[miss_mask],
    truth      = truth_vals,
    imputation = imputed_vals,
    se         = sqrt(imputed_vars),
    ci_low     = imputed_vals - 1.96 * sqrt(imputed_vars),
    ci_high    = imputed_vals + 1.96 * sqrt(imputed_vars),
    covered    = abs(imputed_vals - truth_vals) <= se95
)
write.csv(imp_df, "results_missing_bootstrap_imputation.csv",
          row.names = FALSE)
cat("  saved: results_missing_bootstrap_imputation.csv\n")

# 6d. Resumo agregado
summary_df <- data.frame(
    metric = c("n_obs", "n_missing", "pct_missing",
               "loglik_complete", "loglik_missing",
               "rmse_imputation", "mae_imputation",
               "coverage_95_imputation",
               "kalman_width_mean", "sim_width_mean",
               "width_ratio_sim_over_kalman",
               "n_sim_draws"),
    value  = c(n, n_missing, 100 * n_missing / n,
               logL_full, logL_miss,
               rmse, mae, coverage,
               kalman_width, sim_width, width_ratio,
               n_sim)
)
write.csv(summary_df, "results_missing_bootstrap_summary.csv",
          row.names = FALSE)
cat("  saved: results_missing_bootstrap_summary.csv\n")

cat("\nMissing data + simulation smoother validation complete.\n")
