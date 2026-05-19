# =============================================================================
# Validacao: Local Level Model - Nile data
# Pacotes: KFAS, dlm
#
# Resultados esperados (Durbin & Koopman 2012):
#   sigma2_eps = 15099
#   sigma2_eta = 1469.1
#   logLik = -632.54
# =============================================================================

library(KFAS)
library(dlm)

# Carregar dados
nile <- read.csv("../../data/nile.csv")
y <- nile$flow
n <- length(y)

cat("=== Local Level Model: Nile ===\n")
cat(sprintf("Observations: %d\n", n))
cat(sprintf("Period: %d - %d\n", nile$year[1], nile$year[n]))

# --- KFAS ---
cat("\n--- KFAS ---\n")
model_kfas <- SSModel(y ~ SSMtrend(1, Q = NA), H = NA)
fit_kfas <- fitSSM(model_kfas, inits = c(log(var(y) / 10), log(var(y) / 10)))

if (fit_kfas$optim.out$convergence != 0) {
    warning("KFAS optimization did not converge!")
}

out_kfas <- KFS(fit_kfas$model, filtering = "state", smoothing = "state")

# Extrair parametros
sigma2_eps_kfas <- as.numeric(fit_kfas$model$H[1])
sigma2_eta_kfas <- as.numeric(fit_kfas$model$Q[1])
loglik_kfas <- as.numeric(logLik(fit_kfas$model))

cat(sprintf("sigma2_eps = %.2f\n", sigma2_eps_kfas))
cat(sprintf("sigma2_eta = %.2f\n", sigma2_eta_kfas))
cat(sprintf("logLik     = %.2f\n", loglik_kfas))

# --- dlm ---
cat("\n--- dlm ---\n")
build_fn <- function(pars) {
    dlmModPoly(order = 1, dV = exp(pars[1]), dW = exp(pars[2]))
}
fit_dlm <- dlmMLE(y, parm = c(log(var(y) / 10), log(var(y) / 10)), build = build_fn)

if (fit_dlm$convergence != 0) {
    warning("dlm optimization did not converge!")
}

mod_dlm <- build_fn(fit_dlm$par)
filt_dlm <- dlmFilter(y, mod_dlm)
smooth_dlm <- dlmSmooth(filt_dlm)

sigma2_eps_dlm <- as.numeric(mod_dlm$V[1])
sigma2_eta_dlm <- as.numeric(mod_dlm$W[1])
loglik_dlm <- -fit_dlm$value

cat(sprintf("sigma2_eps = %.2f\n", sigma2_eps_dlm))
cat(sprintf("sigma2_eta = %.2f\n", sigma2_eta_dlm))
cat(sprintf("logLik     = %.2f\n", loglik_dlm))

# --- Exportar resultados ---
results <- data.frame(
    source = c("KFAS", "dlm"),
    sigma2_eps = c(sigma2_eps_kfas, sigma2_eps_dlm),
    sigma2_eta = c(sigma2_eta_kfas, sigma2_eta_dlm),
    loglik = c(loglik_kfas, loglik_dlm)
)
write.csv(results, "results_local_level.csv", row.names = FALSE)
cat("\nSaved: results_local_level.csv\n")

# --- Exportar estados (KFAS) ---
# a[t] = filtered state at time t (one-step-ahead prediction of state)
# alphahat = smoothed state
states_kfas <- data.frame(
    year = nile$year,
    filtered = as.numeric(out_kfas$a[-1]),
    smoothed = as.numeric(out_kfas$alphahat)
)
write.csv(states_kfas, "states_local_level_kfas.csv", row.names = FALSE)
cat("Saved: states_local_level_kfas.csv\n")

cat("\nLocal Level validation complete.\n")
