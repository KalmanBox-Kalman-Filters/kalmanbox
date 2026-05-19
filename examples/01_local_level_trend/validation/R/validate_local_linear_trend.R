# =============================================================================
# Validacao: Local Linear Trend Model - Airline passengers (log-transformed)
# Pacotes: KFAS, dlm
#
# State-space form:
#   y_t = mu_t + eps_t,       eps_t ~ N(0, sigma2_eps)
#   mu_t = mu_{t-1} + nu_{t-1} + eta_t, eta_t ~ N(0, sigma2_eta)
#   nu_t = nu_{t-1} + zeta_t, zeta_t ~ N(0, sigma2_zeta)
#
# 3 variance parameters: sigma2_eps, sigma2_eta, sigma2_zeta
# =============================================================================

library(KFAS)
library(dlm)

# Carregar dados e aplicar log-transform (como no notebook Python)
airline <- read.csv("../../data/airline.csv")
y <- log(airline$passengers)
n <- length(y)
dates <- as.Date(airline$date)

cat("=== Local Linear Trend Model: Airline (log-passengers) ===\n")
cat(sprintf("Observations: %d\n", n))
cat(sprintf("Period: %s - %s\n", dates[1], dates[n]))

# --- KFAS ---
cat("\n--- KFAS ---\n")
model_kfas <- SSModel(y ~ SSMtrend(2, Q = list(NA, NA)), H = NA)
init_var <- var(diff(y))
fit_kfas <- fitSSM(model_kfas,
    inits = c(log(init_var / 10), log(init_var / 100), log(init_var / 100))
)

if (fit_kfas$optim.out$convergence != 0) {
    warning("KFAS optimization did not converge!")
}

out_kfas <- KFS(fit_kfas$model, filtering = "state", smoothing = "state")

# Extrair parametros
sigma2_eps_kfas <- as.numeric(fit_kfas$model$H[1])
sigma2_eta_kfas <- as.numeric(fit_kfas$model$Q[1, 1, 1])
sigma2_zeta_kfas <- as.numeric(fit_kfas$model$Q[2, 2, 1])
loglik_kfas <- as.numeric(logLik(fit_kfas$model))

cat(sprintf("sigma2_eps  = %.6f\n", sigma2_eps_kfas))
cat(sprintf("sigma2_eta  = %.6f\n", sigma2_eta_kfas))
cat(sprintf("sigma2_zeta = %.6f\n", sigma2_zeta_kfas))
cat(sprintf("logLik      = %.4f\n", loglik_kfas))

# --- dlm ---
cat("\n--- dlm ---\n")
build_fn <- function(pars) {
    dlmModPoly(order = 2,
        dV = exp(pars[1]),
        dW = c(exp(pars[2]), exp(pars[3]))
    )
}
fit_dlm <- dlmMLE(y,
    parm = c(log(init_var / 10), log(init_var / 100), log(init_var / 100)),
    build = build_fn
)

if (fit_dlm$convergence != 0) {
    warning("dlm optimization did not converge!")
}

mod_dlm <- build_fn(fit_dlm$par)
filt_dlm <- dlmFilter(y, mod_dlm)
smooth_dlm <- dlmSmooth(filt_dlm)

sigma2_eps_dlm <- as.numeric(mod_dlm$V[1])
sigma2_eta_dlm <- as.numeric(diag(mod_dlm$W)[1])
sigma2_zeta_dlm <- as.numeric(diag(mod_dlm$W)[2])
loglik_dlm <- -fit_dlm$value

cat(sprintf("sigma2_eps  = %.6f\n", sigma2_eps_dlm))
cat(sprintf("sigma2_eta  = %.6f\n", sigma2_eta_dlm))
cat(sprintf("sigma2_zeta = %.6f\n", sigma2_zeta_dlm))
cat(sprintf("logLik      = %.4f\n", loglik_dlm))

# --- Exportar resultados ---
results <- data.frame(
    source = c("KFAS", "dlm"),
    sigma2_eps = c(sigma2_eps_kfas, sigma2_eps_dlm),
    sigma2_eta = c(sigma2_eta_kfas, sigma2_eta_dlm),
    sigma2_zeta = c(sigma2_zeta_kfas, sigma2_zeta_dlm),
    loglik = c(loglik_kfas, loglik_dlm)
)
write.csv(results, "results_local_linear_trend.csv", row.names = FALSE)
cat("\nSaved: results_local_linear_trend.csv\n")

# --- Exportar estados (KFAS) ---
# State vector: [level, trend]
# a = filtered (one-step ahead), alphahat = smoothed
states_kfas <- data.frame(
    date = as.character(dates),
    filtered_level = as.numeric(out_kfas$a[-1, 1]),
    filtered_trend = as.numeric(out_kfas$a[-1, 2]),
    smoothed_level = as.numeric(out_kfas$alphahat[, 1]),
    smoothed_trend = as.numeric(out_kfas$alphahat[, 2])
)
write.csv(states_kfas, "states_llt_kfas.csv", row.names = FALSE)
cat("Saved: states_llt_kfas.csv\n")

# --- Forecast (12 meses) ---
cat("\n--- Forecast (12 steps ahead) ---\n")
pred_kfas <- predict(fit_kfas$model, n.ahead = 12, interval = "prediction")
cat("Forecast from KFAS (log-scale, first 6):\n")
print(head(pred_kfas))

cat("\nLocal Linear Trend validation complete.\n")
