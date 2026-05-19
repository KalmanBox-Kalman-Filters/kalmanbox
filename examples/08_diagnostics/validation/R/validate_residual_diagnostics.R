# =============================================================================
# Validacao: Residual diagnostics para Local Level no Nile
# Pacotes: KFAS, tseries, strucchange
#
# Compara com Python/kalmanbox:
#   - Residuos padronizados recursivos (rstandard, type = "recursive")
#   - Residuos de Pearson (rstandard, type = "pearson")
#   - Auxiliary residuals: residuals(fit, type = "response" / "state")
#
# Testes:
#   - Jarque-Bera  (tseries::jarque.bera.test)
#   - Ljung-Box    (stats::Box.test, type = "Ljung-Box")
#   - Shapiro-Wilk (stats::shapiro.test)
#   - CUSUM via strucchange::efp(..., type = "Rec-CUSUM")
#
# Referencias:
#   Durbin & Koopman (2012), Time Series Analysis by State Space Methods
#   Harvey (1989), Forecasting, Structural Time Series Models, CUP
#   de Jong & Penzer (1998), Diagnosing Shocks in Time Series, JASA 93(442)
# =============================================================================

suppressPackageStartupMessages({
    library(KFAS)
    library(tseries)
    library(strucchange)
})

# ---------------------------------------------------------------------------
# Dados: Nile (clean) e Nile com outliers injetados
# ---------------------------------------------------------------------------
nile      <- read.csv("../../data/nile.csv")
nile_out  <- read.csv("../../data/nile_outliers.csv")

y       <- nile$volume
years   <- nile$year
T       <- length(y)

y_out   <- nile_out$volume
years_o <- nile_out$year
is_out  <- as.logical(nile_out$is_outlier)

cat("=============================================================================\n")
cat("  Residual Diagnostics - Local Level (Nile)\n")
cat("=============================================================================\n\n")
cat(sprintf("Clean series : T = %d (%d - %d)\n", T, years[1], years[T]))
cat(sprintf("Outlier series: T = %d, %d injected outliers at years: %s\n",
            length(y_out), sum(is_out),
            paste(years_o[is_out], collapse = ", ")))

# ---------------------------------------------------------------------------
# 1. Ajuste do Local Level via KFAS (referencia)
# ---------------------------------------------------------------------------
cat("\n--- 1. Fit Local Level via KFAS (clean) ---\n")
model_kfas <- SSModel(y ~ SSMtrend(degree = 1, Q = list(NA)), H = NA)
fit_kfas   <- fitSSM(model_kfas,
                     inits = log(c(var(y) / 10, var(y) / 10)),
                     method = "BFGS")

if (fit_kfas$optim.out$convergence != 0) {
    warning("KFAS optimization did not converge!")
}

sigma2_eps <- as.numeric(fit_kfas$model$H[1])
sigma2_eta <- as.numeric(fit_kfas$model$Q[1])
loglik_val <- as.numeric(logLik(fit_kfas$model))

cat(sprintf("sigma2_eps = %.4f\n", sigma2_eps))
cat(sprintf("sigma2_eta = %.4f\n", sigma2_eta))
cat(sprintf("logLik     = %.4f\n", loglik_val))

kfs_clean <- KFS(fit_kfas$model,
                 filtering = c("state", "mean"),
                 smoothing = c("state", "mean", "disturbance"))

# ---------------------------------------------------------------------------
# 2. Residuos padronizados e de Pearson
# ---------------------------------------------------------------------------
cat("\n--- 2. Residuals (rstandard) ---\n")

# Recursive (one-step-ahead, standardized) residuals
res_rec   <- as.numeric(rstandard(kfs_clean, type = "recursive"))
# Pearson residuals (scaled by observation SD)
res_pears <- as.numeric(rstandard(kfs_clean, type = "pearson"))

# Diffuse burn-in: drop initial NA + 1 (local level has 1 diffuse state)
valid_rec   <- !is.na(res_rec)
res_rec_v   <- res_rec[valid_rec]
years_rec_v <- years[valid_rec]

cat(sprintf(
    "recursive residuals: n = %d (after diffuse burn-in), mean = %+.4f, sd = %.4f\n",
    length(res_rec_v), mean(res_rec_v), sd(res_rec_v)))
cat(sprintf("pearson   residuals: n = %d, mean = %+.4f, sd = %.4f\n",
            sum(!is.na(res_pears)),
            mean(res_pears, na.rm = TRUE), sd(res_pears, na.rm = TRUE)))

# ---------------------------------------------------------------------------
# 3. Testes de especificacao
# ---------------------------------------------------------------------------
cat("\n--- 3. Specification tests ---\n")

# Jarque-Bera (normality)
jb <- jarque.bera.test(res_rec_v)
cat(sprintf("Jarque-Bera  : stat = %.4f, df = %d, p-value = %.4f\n",
            as.numeric(jb$statistic), as.numeric(jb$parameter), jb$p.value))

# Ljung-Box at lag 10 and 20
lb10 <- Box.test(res_rec_v, lag = 10, type = "Ljung-Box")
lb20 <- Box.test(res_rec_v, lag = 20, type = "Ljung-Box")
cat(sprintf("Ljung-Box(10): stat = %.4f, df = %d, p-value = %.4f\n",
            as.numeric(lb10$statistic), as.numeric(lb10$parameter),
            lb10$p.value))
cat(sprintf("Ljung-Box(20): stat = %.4f, df = %d, p-value = %.4f\n",
            as.numeric(lb20$statistic), as.numeric(lb20$parameter),
            lb20$p.value))

# Shapiro-Wilk (normality)
sw <- shapiro.test(res_rec_v)
cat(sprintf("Shapiro-Wilk : W = %.4f, p-value = %.4f\n",
            as.numeric(sw$statistic), sw$p.value))

# ---------------------------------------------------------------------------
# 4. CUSUM via strucchange (parameter stability)
#    Usa empirical fluctuation process recursivo sobre os residuos.
# ---------------------------------------------------------------------------
cat("\n--- 4. CUSUM test (strucchange::efp) ---\n")
# efp precisa de um data.frame e formula. Usamos o processo recursivo
# CUSUM sobre um modelo trivial ~1 para obter fluctuation process dos
# residuos recursivos padronizados (proxy direto).
efp_df <- data.frame(e = res_rec_v)
efp_mod <- efp(e ~ 1, data = efp_df, type = "Rec-CUSUM")
sc_test <- sctest(efp_mod)
max_cusum_stat <- max(abs(as.numeric(efp_mod$process)))
cat(sprintf("Rec-CUSUM empirical process max|W| = %.4f\n", max_cusum_stat))
cat(sprintf("sctest stat = %.4f, p-value = %.4f\n",
            as.numeric(sc_test$statistic), sc_test$p.value))

# Harvey (1989) Brownian-bridge critical values (CUSUM para LL):
#   5%: 0.948, 1%: 1.143
n_e <- length(res_rec_v)
sigma_w <- sd(res_rec_v)
cusum_series <- cumsum(res_rec_v) / (sigma_w * sqrt(n_e))
max_harvey <- max(abs(cusum_series))
cat(sprintf("Harvey-style max|CUSUM| = %.4f (5%%=0.948, 1%%=1.143)\n",
            max_harvey))

# ---------------------------------------------------------------------------
# 5. Auxiliary residuals (KFAS) - deteccao de outliers
#    residuals(KFS(...), type = "response")  -> observation-level (e_eps)
#    residuals(KFS(...), type = "state")     -> state-level       (e_eta)
# ---------------------------------------------------------------------------
cat("\n--- 5. Auxiliary residuals on nile_outliers.csv ---\n")
model_out <- SSModel(y_out ~ SSMtrend(degree = 1, Q = list(NA)), H = NA)
fit_out   <- fitSSM(model_out,
                    inits = log(c(var(y_out) / 10, var(y_out) / 10)),
                    method = "BFGS")
kfs_out   <- KFS(fit_out$model,
                 filtering = c("state", "mean"),
                 smoothing = c("state", "mean", "disturbance"))

aux_resp <- as.numeric(residuals(kfs_out, type = "response"))
aux_stat <- as.numeric(residuals(kfs_out, type = "state"))
# KFAS retorna residuos nao padronizados; padronizar por desvio-padrao
aux_resp_std <- aux_resp / sd(aux_resp, na.rm = TRUE)
aux_stat_std <- aux_stat / sd(aux_stat, na.rm = TRUE)

threshold <- 2.0
flagged_mask <- !is.na(aux_resp_std) & abs(aux_resp_std) > threshold
flagged_years <- years_o[flagged_mask]
true_years    <- years_o[is_out]
tp <- intersect(true_years, flagged_years)
fp <- setdiff(flagged_years, true_years)
fn <- setdiff(true_years, flagged_years)
cat(sprintf("threshold |e_eps_std| > %.1f\n", threshold))
cat(sprintf("  flagged (%d): %s\n",
            length(flagged_years), paste(flagged_years, collapse = ", ")))
cat(sprintf("  true positives (%d): %s\n", length(tp),
            paste(tp, collapse = ", ")))
cat(sprintf("  false negatives (%d): %s\n", length(fn),
            paste(fn, collapse = ", ")))
cat(sprintf("  false positives (%d): %s\n", length(fp),
            paste(fp, collapse = ", ")))

# ---------------------------------------------------------------------------
# 6. Exportar resultados
# ---------------------------------------------------------------------------
cat("\n--- 6. Exporting results ---\n")

# 6a. Residuos (clean)
res_df <- data.frame(
    year                   = years,
    residual_recursive     = res_rec,
    residual_pearson       = res_pears
)
write.csv(res_df, "results_residual_diagnostics_residuals.csv",
          row.names = FALSE)
cat("  saved: results_residual_diagnostics_residuals.csv\n")

# 6b. Auxiliary residuals (outlier series)
aux_df <- data.frame(
    year          = years_o,
    y             = y_out,
    is_outlier    = is_out,
    aux_response  = aux_resp,
    aux_state     = aux_stat,
    aux_response_std = aux_resp_std,
    aux_state_std    = aux_stat_std,
    flagged_gt_2sd   = flagged_mask
)
write.csv(aux_df, "results_residual_diagnostics_auxiliary.csv",
          row.names = FALSE)
cat("  saved: results_residual_diagnostics_auxiliary.csv\n")

# 6c. Test summary
tests_df <- data.frame(
    test       = c("jarque_bera", "ljung_box_10", "ljung_box_20",
                   "shapiro_wilk", "cusum_sctest",
                   "cusum_harvey_max", "aux_outliers_detected"),
    statistic  = c(as.numeric(jb$statistic),
                   as.numeric(lb10$statistic),
                   as.numeric(lb20$statistic),
                   as.numeric(sw$statistic),
                   as.numeric(sc_test$statistic),
                   max_harvey,
                   length(tp)),
    df_or_n    = c(as.numeric(jb$parameter),
                   as.numeric(lb10$parameter),
                   as.numeric(lb20$parameter),
                   n_e,
                   NA_real_,
                   n_e,
                   sum(is_out)),
    p_value    = c(jb$p.value, lb10$p.value, lb20$p.value,
                   sw$p.value, sc_test$p.value,
                   NA_real_, NA_real_)
)
write.csv(tests_df, "results_residual_diagnostics_tests.csv",
          row.names = FALSE)
cat("  saved: results_residual_diagnostics_tests.csv\n")

# 6d. Model summary
model_df <- data.frame(
    source     = "KFAS",
    sigma2_eps = sigma2_eps,
    sigma2_eta = sigma2_eta,
    loglik     = loglik_val,
    n_obs      = T,
    n_resid    = n_e
)
write.csv(model_df, "results_residual_diagnostics_model.csv",
          row.names = FALSE)
cat("  saved: results_residual_diagnostics_model.csv\n")

cat("\nResidual diagnostics validation complete.\n")
