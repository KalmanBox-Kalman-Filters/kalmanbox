# =============================================================================
# Validacao: TVP Regression via dlm
#
# Modelo TVP (equivalente ao kalmanbox TimeVaryingParameters):
#   y_t = x_t' * beta_t + eps_t,        eps_t ~ N(0, V)
#   beta_t = beta_{t-1} + eta_t,        eta_t ~ N(0, W)
#
# Especificacao dlm:
#   - dlmModReg(X, addInt = TRUE) constroi o modelo com regressores em X
#     e intercepto adicional, ja no formato state-space com F_t = [1, x_t]
#     e estado beta_t evoluindo como random walk (G = I_k).
#   - Parametros livres estimados por MLE: V (variancia da observacao) e
#     os elementos da diagonal de W (variancias das inovacoes de beta).
#   - Usamos parametrizacao exp(.) para garantir positividade.
#
# Dataset: us_inflation_unemployment.csv (quarterly 1960Q1-2023Q4)
#   Regressao: inflation ~ 1 + gdp_gap  (mesma spec dos notebooks kalmanbox)
#
# Pacote: dlm (Petris et al. 2009)
# =============================================================================

library(dlm)

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
df <- read.csv("../../data/us_inflation_unemployment.csv")
df$date <- as.Date(df$date)
dates  <- df$date
y_infl <- df$inflation
gap    <- df$gdp_gap
n      <- length(y_infl)

cat("=============================================================================\n")
cat("  TVP Regression - dlm Validation\n")
cat("=============================================================================\n\n")
cat(sprintf("Sample size: n = %d quarters\n", n))
cat(sprintf("Date range: %s -> %s\n",
            as.character(dates[1]), as.character(dates[n])))

# ---------------------------------------------------------------------------
# Especificacao do modelo TVP: inflation ~ 1 + gdp_gap
# ---------------------------------------------------------------------------
# dlmModReg constroi F_t = cbind(1, X) por default (addInt = TRUE),
# G = I_k (random walk nos coeficientes), W e V como parametros livres.
# pars = (log V, log w_intercept, log w_slope)
build_tvp <- function(pars) {
    dlmModReg(
        X      = matrix(gap, ncol = 1),
        addInt = TRUE,
        dV     = exp(pars[1]),
        dW     = exp(pars[2:3])
    )
}

cat("\n--- Estimando dlm TVP por MLE ---\n")
cat("Parametros livres: V (obs), W_intercept, W_slope (random walk)\n\n")

set.seed(42)
init <- c(log(0.1), log(0.1), log(0.01))
fit_mle <- dlmMLE(y_infl, parm = init, build = build_tvp,
                  method = "L-BFGS-B",
                  control = list(maxit = 500, trace = 0))

cat(sprintf("Convergencia: %s\n",
            ifelse(fit_mle$convergence == 0, "OK",
                   paste0("code=", fit_mle$convergence))))
cat(sprintf("loglike (neg) retornado pelo otimizador: %.4f\n", fit_mle$value))

V_hat    <- exp(fit_mle$par[1])
W_hat    <- exp(fit_mle$par[2:3])
loglike  <- -fit_mle$value
cat(sprintf("V (sigma2_obs)   = %.6f\n", V_hat))
cat(sprintf("W_intercept      = %.6f\n", W_hat[1]))
cat(sprintf("W_slope (gdp_gap)= %.6f\n", W_hat[2]))
cat(sprintf("loglike (dlm)    = %.4f\n", loglike))

# ---------------------------------------------------------------------------
# Filtragem e suavizacao
# ---------------------------------------------------------------------------
mod_hat  <- build_tvp(fit_mle$par)
filt     <- dlmFilter(y_infl, mod_hat)
smooth   <- dlmSmooth(filt)

# smooth$s tem dimensao (n+1) x k (inclui estado em t=0); descarta linha 1
beta_smooth <- smooth$s[-1, , drop = FALSE]  # (n x 2)
colnames(beta_smooth) <- c("intercept", "gdp_gap")

# Variancia do smoother: dlmSvd2var(smooth$U.S, smooth$D.S) retorna lista
# de matrizes de covariancia, uma para cada t=0..n. Descarta t=0.
cov_list <- dlmSvd2var(smooth$U.S, smooth$D.S)
cov_list <- cov_list[-1]
beta_var <- t(sapply(cov_list, function(M) diag(M)))  # (n x 2)
beta_se  <- sqrt(pmax(beta_var, 0))

# Filtrado (recursive): filt$m tem (n+1) x k
beta_filtered <- filt$m[-1, , drop = FALSE]
colnames(beta_filtered) <- c("intercept", "gdp_gap")

# ---------------------------------------------------------------------------
# Referencia: OLS full-sample
# ---------------------------------------------------------------------------
ols <- lm(y_infl ~ gap)
alpha_ols <- unname(coef(ols)[1])
beta_ols  <- unname(coef(ols)[2])
cat("\n--- Referencia OLS (pool, full sample) ---\n")
cat(sprintf("  intercept = %.4f\n", alpha_ols))
cat(sprintf("  gdp_gap   = %.4f\n", beta_ols))

# ---------------------------------------------------------------------------
# Estatisticas finais
# ---------------------------------------------------------------------------
cat("\n--- Coeficientes TV (amostras selecionadas) ---\n")
idx <- c(1, floor(n / 4), floor(n / 2), floor(3 * n / 4), n)
for (i in idx) {
    cat(sprintf("  t=%3d (%s)  intercept=%.4f (se=%.4f)  slope=%.4f (se=%.4f)\n",
                i, as.character(dates[i]),
                beta_smooth[i, 1], beta_se[i, 1],
                beta_smooth[i, 2], beta_se[i, 2]))
}

cat(sprintf("\nTerminal beta (t=T): intercept=%.4f, slope=%.4f\n",
            beta_smooth[n, 1], beta_smooth[n, 2]))
cat(sprintf("OLS reference:       intercept=%.4f, slope=%.4f\n",
            alpha_ols, beta_ols))

# ---------------------------------------------------------------------------
# Exportacao CSV
# ---------------------------------------------------------------------------
out_df <- data.frame(
    date                = dates,
    beta_intercept      = beta_smooth[, 1],
    beta_intercept_se   = beta_se[, 1],
    beta_intercept_lo   = beta_smooth[, 1] - 1.96 * beta_se[, 1],
    beta_intercept_hi   = beta_smooth[, 1] + 1.96 * beta_se[, 1],
    beta_slope          = beta_smooth[, 2],
    beta_slope_se       = beta_se[, 2],
    beta_slope_lo       = beta_smooth[, 2] - 1.96 * beta_se[, 2],
    beta_slope_hi       = beta_smooth[, 2] + 1.96 * beta_se[, 2],
    filt_intercept      = beta_filtered[, 1],
    filt_slope          = beta_filtered[, 2]
)
write.csv(out_df, "results_tvp_regression_coefs.csv", row.names = FALSE)
cat("\nSaved: results_tvp_regression_coefs.csv\n")

summary_df <- data.frame(
    metric = c("logLik", "V_sigma2_obs", "W_intercept", "W_slope",
               "ols_intercept", "ols_slope",
               "terminal_intercept", "terminal_slope",
               "n_obs", "convergence"),
    value = c(loglike, V_hat, W_hat[1], W_hat[2],
              alpha_ols, beta_ols,
              beta_smooth[n, 1], beta_smooth[n, 2],
              n, fit_mle$convergence)
)
write.csv(summary_df, "results_tvp_regression_summary.csv", row.names = FALSE)
cat("Saved: results_tvp_regression_summary.csv\n")

cat("\ndlm TVP regression validation complete.\n")
