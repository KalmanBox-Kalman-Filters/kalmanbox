# =============================================================================
# Validacao: Complete univariate workflow - UK Drivers (log-transformed)
# Pacotes: KFAS, dlm
#
# Workflow replicado do notebook 01_univariate_workflow.ipynb:
#   1. Carregar uk_drivers.csv, aplicar log-transform e construir dummy
#      belt_law (Feb 1983 onwards).
#   2. Ajustar 4 candidatos:
#        - Local Level
#        - Local Linear Trend (LLT)
#        - Basic Structural Model (BSM, s=12)
#        - BSM + intervencao (seatbelt law)
#   3. Tabela AIC / BIC / logLik para selecao de modelo.
#   4. Decomposicao do melhor modelo em nivel + tendencia + sazonal +
#      intervencao + irregular.
#   5. Diagnosticos de residuos (Ljung-Box, normalidade).
#   6. Forecast 24 meses com bandas de 95%.
#   7. Exportar tudo em CSV para comparacao com kalmanbox.
# =============================================================================

library(KFAS)
library(dlm)

# -----------------------------------------------------------------------------
# 0. Utilitarios
# -----------------------------------------------------------------------------

kfas_bic <- function(fit, n_params, n_obs) {
    ll <- as.numeric(logLik(fit$model))
    -2 * ll + n_params * log(n_obs)
}

kfas_aic <- function(fit, n_params) {
    ll <- as.numeric(logLik(fit$model))
    -2 * ll + 2 * n_params
}

safe_fitSSM <- function(model, inits, label) {
    fit <- tryCatch(
        fitSSM(model, inits = inits, method = "BFGS"),
        error = function(e) {
            warning(sprintf("[%s] fitSSM failed: %s", label, e$message))
            NULL
        }
    )
    if (!is.null(fit) && fit$optim.out$convergence != 0) {
        warning(sprintf("[%s] optim did not converge (code=%d)",
                        label, fit$optim.out$convergence))
    }
    fit
}

# -----------------------------------------------------------------------------
# 1. Dados
# -----------------------------------------------------------------------------

drivers <- read.csv("../../data/uk_drivers.csv")
drivers$date <- as.Date(drivers$date)
y <- log(drivers$deaths)
n <- length(y)
dates <- drivers$date
belt_law <- as.numeric(dates >= as.Date("1983-02-01"))

cat("=============================================================================\n")
cat("  Univariate workflow validation - UK driver deaths (log scale)\n")
cat("=============================================================================\n\n")
cat(sprintf("Observations: %d\n", n))
cat(sprintf("Period: %s -> %s\n", dates[1], dates[n]))
cat(sprintf("Intervention (seatbelt law) obs: %d\n", sum(belt_law)))

init_var <- var(diff(y))

# -----------------------------------------------------------------------------
# 2. Ajuste dos 4 modelos candidatos
# -----------------------------------------------------------------------------

cat("\n--- Fitting 4 candidate models (KFAS) ---\n")

# ---- Model 1: Local Level ----
m1_model <- SSModel(y ~ SSMtrend(1, Q = NA), H = NA)
m1 <- safe_fitSSM(m1_model, inits = c(log(init_var), log(init_var / 10)),
                  label = "LocalLevel")
m1_k <- 2
m1_ll <- as.numeric(logLik(m1$model))
m1_aic <- kfas_aic(m1, m1_k)
m1_bic <- kfas_bic(m1, m1_k, n)

# ---- Model 2: Local Linear Trend ----
m2_model <- SSModel(y ~ SSMtrend(2, Q = list(NA, NA)), H = NA)
m2 <- safe_fitSSM(m2_model,
                  inits = c(log(init_var), log(init_var / 100), log(init_var / 1000)),
                  label = "LocalLinearTrend")
m2_k <- 3
m2_ll <- as.numeric(logLik(m2$model))
m2_aic <- kfas_aic(m2, m2_k)
m2_bic <- kfas_bic(m2, m2_k, n)

# ---- Model 3: Basic Structural Model (BSM) ----
m3_model <- SSModel(
    y ~ SSMtrend(2, Q = list(NA, NA)) + SSMseasonal(12, Q = NA),
    H = NA
)
m3 <- safe_fitSSM(m3_model,
                  inits = c(log(init_var), log(init_var / 100),
                            log(init_var / 100), log(init_var / 100)),
                  label = "BSM")
m3_k <- 4
m3_ll <- as.numeric(logLik(m3$model))
m3_aic <- kfas_aic(m3, m3_k)
m3_bic <- kfas_bic(m3, m3_k, n)

# ---- Model 4: BSM + intervention (seatbelt law) ----
m4_model <- SSModel(
    y ~ SSMtrend(2, Q = list(NA, NA)) + SSMseasonal(12, Q = NA) + belt_law,
    H = NA
)
m4 <- safe_fitSSM(m4_model,
                  inits = c(log(init_var), log(init_var / 100),
                            log(init_var / 100), log(init_var / 100)),
                  label = "BSM+intervention")
m4_k <- 4  # 4 variance parameters (coef regressao entra nos estados)
m4_ll <- as.numeric(logLik(m4$model))
m4_aic <- kfas_aic(m4, m4_k)
m4_bic <- kfas_bic(m4, m4_k, n)

# -----------------------------------------------------------------------------
# 3. Tabela AIC/BIC (comparacao de modelos)
# -----------------------------------------------------------------------------

model_comparison <- data.frame(
    model = c("LocalLevel", "LocalLinearTrend", "BSM", "BSM+intervention"),
    k_params = c(m1_k, m2_k, m3_k, m4_k),
    loglik = c(m1_ll, m2_ll, m3_ll, m4_ll),
    aic = c(m1_aic, m2_aic, m3_aic, m4_aic),
    bic = c(m1_bic, m2_bic, m3_bic, m4_bic)
)
model_comparison$delta_aic <- model_comparison$aic - min(model_comparison$aic)
model_comparison <- model_comparison[order(model_comparison$aic), ]

cat("\nModel comparison (sorted by AIC):\n")
print(model_comparison, row.names = FALSE, digits = 5)

write.csv(model_comparison, "r_univariate_model_comparison.csv",
          row.names = FALSE)
cat("Saved: r_univariate_model_comparison.csv\n")

# -----------------------------------------------------------------------------
# 4. Selecionar melhor modelo (menor AIC)
# -----------------------------------------------------------------------------

best_name <- as.character(model_comparison$model[1])
cat(sprintf("\nSelected best model (min AIC): %s\n", best_name))

fits <- list(LocalLevel = m1, LocalLinearTrend = m2,
             BSM = m3, `BSM+intervention` = m4)
best_fit <- fits[[best_name]]
best_out <- KFS(best_fit$model,
                filtering = c("state", "signal"),
                smoothing = c("state", "signal"))

# -----------------------------------------------------------------------------
# 5. Parametros do melhor modelo
# -----------------------------------------------------------------------------

H_best <- as.numeric(best_fit$model$H[1])
Q_best_diag <- diag(best_fit$model$Q[, , 1])

best_params <- data.frame(
    parameter = c("sigma2_obs", paste0("sigma2_state_", seq_along(Q_best_diag))),
    estimate = c(H_best, Q_best_diag)
)
write.csv(best_params, "r_univariate_best_params.csv", row.names = FALSE)
cat("Saved: r_univariate_best_params.csv\n")

# -----------------------------------------------------------------------------
# 6. Decomposicao (observed = level + slope + seasonal + intervention + irregular)
# -----------------------------------------------------------------------------

state_names <- colnames(best_out$alphahat)
get_state <- function(name) {
    idx <- which(state_names == name)
    if (length(idx) == 0) return(rep(0, n))
    as.numeric(best_out$alphahat[, idx])
}

level_comp <- get_state("level")
slope_comp <- get_state("slope")
seasonal_comp <- if ("sea_dummy1" %in% state_names) {
    get_state("sea_dummy1")
} else {
    rep(0, n)
}
intervention_comp <- if ("belt_law" %in% state_names) {
    belt_law * get_state("belt_law")
} else {
    rep(0, n)
}

signal <- as.numeric(best_out$muhat)
irregular <- y - signal

decomposition <- data.frame(
    date = as.character(dates),
    observed = y,
    level = level_comp,
    slope = slope_comp,
    seasonal = seasonal_comp,
    intervention = intervention_comp,
    irregular = irregular
)
write.csv(decomposition, "r_univariate_decomposition.csv", row.names = FALSE)
cat("Saved: r_univariate_decomposition.csv\n")

# -----------------------------------------------------------------------------
# 7. Diagnosticos: residuos padronizados, Ljung-Box, normalidade
# -----------------------------------------------------------------------------

# Standardised innovations (one-step-ahead prediction errors)
resid_std <- as.numeric(rstandard(best_out, type = "pearson"))
resid_std <- resid_std[!is.na(resid_std)]

# Ljung-Box test em diferentes lags
lb_lags <- c(10, 12, 20, 24)
lb_results <- do.call(rbind, lapply(lb_lags, function(L) {
    tst <- Box.test(resid_std, lag = L, type = "Ljung-Box")
    data.frame(
        test = sprintf("Ljung-Box(%d)", L),
        statistic = as.numeric(tst$statistic),
        p_value = as.numeric(tst$p.value),
        reject_H0_5pct = as.numeric(tst$p.value) < 0.05
    )
}))

# Jarque-Bera manual (sem dependencia extra)
n_res <- length(resid_std)
m_r <- mean(resid_std)
s_r <- sd(resid_std)
skew <- mean((resid_std - m_r)^3) / s_r^3
kurt <- mean((resid_std - m_r)^4) / s_r^4
jb_stat <- n_res * (skew^2 / 6 + (kurt - 3)^2 / 24)
jb_pval <- 1 - pchisq(jb_stat, df = 2)

# Shapiro-Wilk (valido ate n=5000)
sw <- shapiro.test(resid_std)

diagnostics <- rbind(
    lb_results,
    data.frame(
        test = "Jarque-Bera",
        statistic = jb_stat,
        p_value = jb_pval,
        reject_H0_5pct = jb_pval < 0.05
    ),
    data.frame(
        test = "Shapiro-Wilk",
        statistic = as.numeric(sw$statistic),
        p_value = as.numeric(sw$p.value),
        reject_H0_5pct = as.numeric(sw$p.value) < 0.05
    )
)

cat("\nDiagnostic tests:\n")
print(diagnostics, row.names = FALSE)

write.csv(diagnostics, "r_univariate_diagnostics.csv", row.names = FALSE)
cat("Saved: r_univariate_diagnostics.csv\n")

# Residuos padronizados em CSV (para plot comparativo)
residuals_df <- data.frame(
    date = as.character(dates),
    standardized_residual = c(rep(NA, n - length(resid_std)), resid_std)
)
write.csv(residuals_df, "r_univariate_residuals.csv", row.names = FALSE)
cat("Saved: r_univariate_residuals.csv\n")

# -----------------------------------------------------------------------------
# 8. Forecast 24 meses com intervalo de 95%
# -----------------------------------------------------------------------------

h_forecast <- 24
last_date <- dates[n]
future_dates <- seq(last_date, by = "month", length.out = h_forecast + 1)[-1]

if (best_name == "BSM+intervention") {
    # Com newdata para a covariavel belt_law (sempre 1 apos 1983)
    newdata <- SSModel(
        rep(NA, h_forecast) ~
            SSMtrend(2, Q = list(NA, NA)) +
            SSMseasonal(12, Q = NA) +
            rep(1, h_forecast),
        H = NA
    )
    pred <- predict(best_fit$model, newdata = newdata,
                    n.ahead = h_forecast, interval = "prediction", level = 0.95)
} else {
    pred <- predict(best_fit$model, n.ahead = h_forecast,
                    interval = "prediction", level = 0.95)
}
pred_mat <- as.matrix(pred)

forecast_df <- data.frame(
    date = as.character(future_dates),
    mean = pred_mat[, "fit"],
    lower = pred_mat[, "lwr"],
    upper = pred_mat[, "upr"]
)
write.csv(forecast_df, "r_univariate_forecast.csv", row.names = FALSE)
cat("Saved: r_univariate_forecast.csv\n")

# -----------------------------------------------------------------------------
# 9. Sumario textual do melhor modelo
# -----------------------------------------------------------------------------

summary_lines <- c(
    strrep("=", 70),
    sprintf("  Best model (R / KFAS): %s", best_name),
    strrep("=", 70),
    sprintf("Log-Likelihood:    %12.4f", m1_ll * 0 + as.numeric(logLik(best_fit$model))),
    sprintf("AIC:               %12.4f", model_comparison$aic[1]),
    sprintf("BIC:               %12.4f", model_comparison$bic[1]),
    sprintf("k_params:          %12d", model_comparison$k_params[1]),
    sprintf("n_obs:             %12d", n),
    sprintf("sigma2_obs (H):    %12.6f", H_best),
    paste0("sigma2_state:      ",
           paste(sprintf("%.6f", Q_best_diag), collapse = ", "))
)
writeLines(summary_lines, "r_univariate_best_summary.txt")
cat("Saved: r_univariate_best_summary.txt\n")

# -----------------------------------------------------------------------------
# 10. Tabela de comparacao kalmanbox vs R (se output kalmanbox existir)
# -----------------------------------------------------------------------------

kbox_comp_path <- "../../output/univariate_model_comparison.csv"
if (file.exists(kbox_comp_path)) {
    kbox <- read.csv(kbox_comp_path)
    # Merge por nome do modelo (kalmanbox usa mesma nomenclatura)
    merged <- merge(
        kbox[, c("model", "loglik", "aic", "bic")],
        model_comparison[, c("model", "loglik", "aic", "bic")],
        by = "model",
        suffixes = c("_kalmanbox", "_R")
    )
    merged$d_loglik <- merged$loglik_R - merged$loglik_kalmanbox
    merged$d_aic <- merged$aic_R - merged$aic_kalmanbox
    merged$d_bic <- merged$bic_R - merged$bic_kalmanbox
    write.csv(merged, "r_univariate_vs_kalmanbox.csv", row.names = FALSE)
    cat("Saved: r_univariate_vs_kalmanbox.csv\n")

    cat("\nkalmanbox vs KFAS comparison:\n")
    print(merged[, c("model", "loglik_kalmanbox", "loglik_R",
                     "aic_kalmanbox", "aic_R", "bic_kalmanbox", "bic_R")],
          row.names = FALSE, digits = 5)
}

cat("\nUnivariate R validation complete.\n")
