# =============================================================================
# Validacao: Basic Structural Model - UK Drivers (log-transformed)
# Pacotes: KFAS, dlm
#
# State-space form:
#   y_t = mu_t + gamma_t + lambda * x_t + eps_t
#   mu_t = mu_{t-1} + nu_{t-1} + eta_t       (level)
#   nu_t = nu_{t-1} + zeta_t                  (trend)
#   gamma_t = -sum(gamma_{t-j}) + omega_t     (seasonal, s=12)
#
# 4 variance parameters: sigma2_eps, sigma2_eta, sigma2_zeta, sigma2_omega
# Intervention: seatbelt law, Feb 1983 (step dummy)
# =============================================================================

library(KFAS)
library(dlm)

# Carregar dados e aplicar log-transform
drivers <- read.csv("../../data/uk_drivers.csv")
y <- log(drivers$deaths)
n <- length(y)
dates <- as.Date(drivers$date)

cat("=== Basic Structural Model: UK Drivers (log-deaths) ===\n")
cat(sprintf("Observations: %d\n", n))
cat(sprintf("Period: %s - %s\n", dates[1], dates[n]))

# Criar variavel de intervencao (seatbelt law: Feb 1983 onwards)
seatbelt <- as.numeric(dates >= as.Date("1983-02-01"))
cat(sprintf("Seatbelt law observations: %d (from Feb 1983)\n", sum(seatbelt)))

# =========================================================================
# MODEL 1: BSM sem intervencao
# =========================================================================
cat("\n========================================\n")
cat("MODEL 1: BSM without intervention\n")
cat("========================================\n")

# --- KFAS (sem intervencao) ---
cat("\n--- KFAS (no intervention) ---\n")
model_kfas_noint <- SSModel(y ~
    SSMtrend(2, Q = list(NA, NA)) +
    SSMseasonal(12, Q = NA),
    H = NA
)

init_var <- var(diff(y))
inits <- c(
    log(init_var / 10),   # H (obs variance)
    log(init_var / 100),  # Q level
    log(init_var / 100),  # Q trend
    log(init_var / 100)   # Q seasonal
)
fit_kfas_noint <- fitSSM(model_kfas_noint, inits = inits)

if (fit_kfas_noint$optim.out$convergence != 0) {
    warning("KFAS (no intervention) optimization did not converge!")
}

out_kfas_noint <- KFS(fit_kfas_noint$model, filtering = "state", smoothing = "state")

sigma2_eps_noint <- as.numeric(fit_kfas_noint$model$H[1])
sigma2_eta_noint <- as.numeric(fit_kfas_noint$model$Q[1, 1, 1])
sigma2_zeta_noint <- as.numeric(fit_kfas_noint$model$Q[2, 2, 1])
sigma2_omega_noint <- as.numeric(fit_kfas_noint$model$Q[3, 3, 1])
loglik_kfas_noint <- as.numeric(logLik(fit_kfas_noint$model))

cat(sprintf("sigma2_eps   = %.6f\n", sigma2_eps_noint))
cat(sprintf("sigma2_eta   = %.6f\n", sigma2_eta_noint))
cat(sprintf("sigma2_zeta  = %.6f\n", sigma2_zeta_noint))
cat(sprintf("sigma2_omega = %.6f\n", sigma2_omega_noint))
cat(sprintf("logLik       = %.4f\n", loglik_kfas_noint))

# --- dlm (sem intervencao) ---
cat("\n--- dlm (no intervention) ---\n")
build_fn_noint <- function(pars) {
    mod_trend <- dlmModPoly(order = 2,
        dV = exp(pars[1]),
        dW = c(exp(pars[2]), exp(pars[3]))
    )
    mod_seas <- dlmModSeas(frequency = 12, dV = 0, dW = c(exp(pars[4]), rep(0, 10)))
    mod_trend + mod_seas
}

fit_dlm_noint <- dlmMLE(y, parm = inits, build = build_fn_noint)

if (fit_dlm_noint$convergence != 0) {
    warning("dlm (no intervention) optimization did not converge!")
}

mod_dlm_noint <- build_fn_noint(fit_dlm_noint$par)
filt_dlm_noint <- dlmFilter(y, mod_dlm_noint)
smooth_dlm_noint <- dlmSmooth(filt_dlm_noint)

sigma2_eps_dlm_noint <- as.numeric(mod_dlm_noint$V[1])
sigma2_eta_dlm_noint <- as.numeric(diag(mod_dlm_noint$W)[1])
sigma2_zeta_dlm_noint <- as.numeric(diag(mod_dlm_noint$W)[2])
sigma2_omega_dlm_noint <- as.numeric(diag(mod_dlm_noint$W)[3])
loglik_dlm_noint <- -fit_dlm_noint$value

cat(sprintf("sigma2_eps   = %.6f\n", sigma2_eps_dlm_noint))
cat(sprintf("sigma2_eta   = %.6f\n", sigma2_eta_dlm_noint))
cat(sprintf("sigma2_zeta  = %.6f\n", sigma2_zeta_dlm_noint))
cat(sprintf("sigma2_omega = %.6f\n", sigma2_omega_dlm_noint))
cat(sprintf("logLik       = %.4f\n", loglik_dlm_noint))

# =========================================================================
# MODEL 2: BSM com intervencao (seatbelt law)
# =========================================================================
cat("\n========================================\n")
cat("MODEL 2: BSM with intervention (seatbelt law)\n")
cat("========================================\n")

# --- KFAS (com intervencao) ---
cat("\n--- KFAS (with intervention) ---\n")
model_kfas_int <- SSModel(y ~
    SSMtrend(2, Q = list(NA, NA)) +
    SSMseasonal(12, Q = NA) +
    seatbelt,
    H = NA
)

fit_kfas_int <- fitSSM(model_kfas_int, inits = inits)

if (fit_kfas_int$optim.out$convergence != 0) {
    warning("KFAS (with intervention) optimization did not converge!")
}

out_kfas_int <- KFS(fit_kfas_int$model, filtering = "state", smoothing = "state")

sigma2_eps_int <- as.numeric(fit_kfas_int$model$H[1])
sigma2_eta_int <- as.numeric(fit_kfas_int$model$Q[1, 1, 1])
sigma2_zeta_int <- as.numeric(fit_kfas_int$model$Q[2, 2, 1])
sigma2_omega_int <- as.numeric(fit_kfas_int$model$Q[3, 3, 1])
loglik_kfas_int <- as.numeric(logLik(fit_kfas_int$model))

# Coeficiente da intervencao (regression coefficient is in state "seatbelt")
seatbelt_idx <- which(colnames(out_kfas_int$alphahat) == "seatbelt")
seatbelt_coef_kfas <- out_kfas_int$alphahat[n, seatbelt_idx]

cat(sprintf("sigma2_eps   = %.6f\n", sigma2_eps_int))
cat(sprintf("sigma2_eta   = %.6f\n", sigma2_eta_int))
cat(sprintf("sigma2_zeta  = %.6f\n", sigma2_zeta_int))
cat(sprintf("sigma2_omega = %.6f\n", sigma2_omega_int))
cat(sprintf("logLik       = %.4f\n", loglik_kfas_int))
cat(sprintf("Seatbelt coef (KFAS) = %.6f\n", seatbelt_coef_kfas))

# --- dlm (com intervencao) ---
cat("\n--- dlm (with intervention) ---\n")
# dlm does not natively support regression; use KFAS-adjusted y
# We include the intervention via dlmModReg
build_fn_int <- function(pars) {
    mod_trend <- dlmModPoly(order = 2,
        dV = exp(pars[1]),
        dW = c(exp(pars[2]), exp(pars[3]))
    )
    mod_seas <- dlmModSeas(frequency = 12, dV = 0, dW = c(exp(pars[4]), rep(0, 10)))
    mod_reg <- dlmModReg(X = seatbelt, addInt = FALSE, dV = 0, dW = 0)
    mod_trend + mod_seas + mod_reg
}

fit_dlm_int <- dlmMLE(y, parm = inits, build = build_fn_int)

if (fit_dlm_int$convergence != 0) {
    warning("dlm (with intervention) optimization did not converge!")
}

mod_dlm_int <- build_fn_int(fit_dlm_int$par)
filt_dlm_int <- dlmFilter(y, mod_dlm_int)
smooth_dlm_int <- dlmSmooth(filt_dlm_int)

sigma2_eps_dlm_int <- as.numeric(mod_dlm_int$V[1])
sigma2_eta_dlm_int <- as.numeric(diag(mod_dlm_int$W)[1])
sigma2_zeta_dlm_int <- as.numeric(diag(mod_dlm_int$W)[2])
sigma2_omega_dlm_int <- as.numeric(diag(mod_dlm_int$W)[3])
loglik_dlm_int <- -fit_dlm_int$value

# Intervention coefficient from smoothed states (last state = regression coef)
n_states <- ncol(smooth_dlm_int$s)
seatbelt_coef_dlm <- smooth_dlm_int$s[n + 1, n_states]

cat(sprintf("sigma2_eps   = %.6f\n", sigma2_eps_dlm_int))
cat(sprintf("sigma2_eta   = %.6f\n", sigma2_eta_dlm_int))
cat(sprintf("sigma2_zeta  = %.6f\n", sigma2_zeta_dlm_int))
cat(sprintf("sigma2_omega = %.6f\n", sigma2_omega_dlm_int))
cat(sprintf("logLik       = %.4f\n", loglik_dlm_int))
cat(sprintf("Seatbelt coef (dlm) = %.6f\n", seatbelt_coef_dlm))

# =========================================================================
# Exportar resultados
# =========================================================================

# Parametros - ambos os modelos
results <- data.frame(
    source = c("KFAS_no_intervention", "dlm_no_intervention",
               "KFAS_with_intervention", "dlm_with_intervention"),
    sigma2_eps = c(sigma2_eps_noint, sigma2_eps_dlm_noint,
                   sigma2_eps_int, sigma2_eps_dlm_int),
    sigma2_eta = c(sigma2_eta_noint, sigma2_eta_dlm_noint,
                   sigma2_eta_int, sigma2_eta_dlm_int),
    sigma2_zeta = c(sigma2_zeta_noint, sigma2_zeta_dlm_noint,
                    sigma2_zeta_int, sigma2_zeta_dlm_int),
    sigma2_omega = c(sigma2_omega_noint, sigma2_omega_dlm_noint,
                     sigma2_omega_int, sigma2_omega_dlm_int),
    loglik = c(loglik_kfas_noint, loglik_dlm_noint,
               loglik_kfas_int, loglik_dlm_int)
)
write.csv(results, "results_bsm.csv", row.names = FALSE)
cat("\nSaved: results_bsm.csv\n")

# Estados decompostos (KFAS, modelo com intervencao)
# State ordering in KFAS with SSMtrend(2) + SSMseasonal(12) + seatbelt:
#   state 1: seatbelt (regression coefficient)
#   states 2-3: level, slope
#   states 4-14: seasonal dummies (11 states for s=12)
state_names <- colnames(out_kfas_int$alphahat)
cat("State ordering:", paste(state_names, collapse = ", "), "\n")

# Find correct column indices
idx_level <- which(state_names == "level")
idx_slope <- which(state_names == "slope")
idx_seas <- which(state_names == "sea_dummy1")

states_bsm <- data.frame(
    date = as.character(dates),
    smoothed_level = as.numeric(out_kfas_int$alphahat[, idx_level]),
    smoothed_trend = as.numeric(out_kfas_int$alphahat[, idx_slope]),
    smoothed_seasonal = as.numeric(out_kfas_int$alphahat[, idx_seas]),
    filtered_level = as.numeric(out_kfas_int$a[-1, idx_level]),
    filtered_trend = as.numeric(out_kfas_int$a[-1, idx_slope]),
    filtered_seasonal = as.numeric(out_kfas_int$a[-1, idx_seas])
)
write.csv(states_bsm, "states_bsm_kfas.csv", row.names = FALSE)
cat("Saved: states_bsm_kfas.csv\n")

# --- Forecast (12 meses, modelo sem intervencao) ---
cat("\n--- Forecast (12 steps ahead, no-intervention model) ---\n")
pred_bsm <- predict(fit_kfas_noint$model, n.ahead = 12, interval = "prediction")
cat("Forecast from KFAS (log-scale, first 6):\n")
print(head(pred_bsm))

cat("\nBSM validation complete.\n")
