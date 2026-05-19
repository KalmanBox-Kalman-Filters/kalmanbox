# =============================================================================
# Validacao: UCM - Unobserved Components Model (Brazil GDP)
# Pacote: KFAS
#
# Modelo: local linear trend + stochastic cycle
#   y_t = mu_t + psi_t + eps_t
#   mu_t = mu_{t-1} + nu_{t-1} + xi_t      (level, xi_t ~ N(0, sigma2_level))
#   nu_t = nu_{t-1} + zeta_t                (slope, zeta_t ~ N(0, sigma2_slope))
#   psi_t = cos(lambda) * psi_{t-1} + sin(lambda) * psi*_{t-1} + kappa_t
#   psi*_t = -sin(lambda) * psi_{t-1} + cos(lambda) * psi*_{t-1} + kappa*_t
#
# KFAS parametrization:
#   SSMtrend(degree=2, Q) - local linear trend with separate level/slope variances
#   SSMcycle(period, Q) - stochastic cycle with fixed period
#
# Strategy: Profile likelihood over a grid of cycle periods to find the
#   optimal period, since KFAS does not natively estimate the period.
#   Grid: 16-48 quarters (4-12 years), typical for business cycles.
#
# Dados: Brazil quarterly GDP index (log), 2000-2023
# =============================================================================

library(KFAS)

cat("=== UCM Validation: Brazil GDP (KFAS) ===\n\n")

# --- Load data ---
gdp <- read.csv("../../data/brazil_gdp.csv")
y_raw <- gdp$gdp_index
y <- log(y_raw)  # Log transformation (multiplicative decomposition)
n <- length(y)
dates <- as.Date(gdp$date)

cat(sprintf("Observations: %d\n", n))
cat(sprintf("Period: %s to %s\n", dates[1], dates[n]))
cat(sprintf("Frequency: quarterly\n\n"))

# --- Profile likelihood: find optimal cycle period ---
# For each candidate period, fit the model and record log-likelihood.
# KFAS requires a fixed period for SSMcycle; we search over a grid.
periods <- seq(16, 48, by = 0.5)  # quarters (4-12 years)
logliks <- numeric(length(periods))

cat("Searching for optimal cycle period (profile likelihood)...\n")

for (j in seq_along(periods)) {
    p <- periods[j]
    m <- SSModel(
        y ~ SSMtrend(degree = 2, Q = list(matrix(NA), matrix(NA))) +
            SSMcycle(period = p, Q = matrix(NA)),
        H = matrix(NA)
    )
    # Default updatefn: 5 NAs -> 5 inits (H, Q_level, Q_slope, Q_cycle, Q_cycle*)
    f <- fitSSM(m, inits = rep(-2, 5), method = "BFGS",
                control = list(maxit = 500))
    if (f$optim.out$convergence == 0) {
        logliks[j] <- as.numeric(logLik(f$model))
    } else {
        logliks[j] <- -Inf
    }
}

# Best period
best_idx <- which.max(logliks)
best_period <- periods[best_idx]
cat(sprintf("Best period: %.1f quarters (%.2f years), logLik = %.4f\n\n",
            best_period, best_period / 4, logliks[best_idx]))

# --- Fit final model with best period ---
model <- SSModel(
    y ~ SSMtrend(degree = 2, Q = list(matrix(NA), matrix(NA))) +
        SSMcycle(period = best_period, Q = matrix(NA)),
    H = matrix(NA)
)

fit <- fitSSM(model, inits = rep(-2, 5), method = "BFGS",
              control = list(maxit = 1000))

if (fit$optim.out$convergence != 0) {
    warning("KFAS optimization did not converge!")
} else {
    cat("Final model optimization converged.\n\n")
}

# --- Extract results ---
fitted_model <- fit$model

# Run Kalman smoother
out <- KFS(fitted_model, filtering = "state", smoothing = "state")

# Extract estimated variances
sigma2_obs <- as.numeric(fitted_model$H[1])
sigma2_level <- as.numeric(fitted_model$Q[1, 1, 1])
sigma2_slope <- as.numeric(fitted_model$Q[2, 2, 1])
sigma2_cycle <- as.numeric(fitted_model$Q[3, 3, 1])

# Cycle frequency
lambda_c <- 2 * pi / best_period
period_est <- best_period

# Log-likelihood
loglik <- as.numeric(logLik(fitted_model))

cat("--- Estimated Parameters ---\n")
cat(sprintf("sigma2_obs   = %.6f\n", sigma2_obs))
cat(sprintf("sigma2_level = %.6f\n", sigma2_level))
cat(sprintf("sigma2_slope = %.6f\n", sigma2_slope))
cat(sprintf("sigma2_cycle = %.6f\n", sigma2_cycle))
cat(sprintf("lambda_c     = %.6f rad\n", lambda_c))
cat(sprintf("period       = %.2f quarters (%.2f years)\n", period_est, period_est / 4))
cat(sprintf("logLik       = %.4f\n", loglik))

# --- Decomposition: extract smoothed states ---
# State order in KFAS: level, slope, cycle, cycle*
state_names <- colnames(out$alphahat)
cat(sprintf("State names: %s\n", paste(state_names, collapse = ", ")))

level_smooth <- as.numeric(out$alphahat[, "level"])
slope_smooth <- as.numeric(out$alphahat[, "slope"])
cycle_smooth <- as.numeric(out$alphahat[, "cycle"])

# Irregular component (residuals)
irregular <- y - level_smooth - cycle_smooth

cat("\n--- Decomposition Summary ---\n")
cat(sprintf("Level range:     [%.4f, %.4f]\n", min(level_smooth), max(level_smooth)))
cat(sprintf("Slope range:     [%.6f, %.6f]\n", min(slope_smooth), max(slope_smooth)))
cat(sprintf("Cycle range:     [%.4f, %.4f]\n", min(cycle_smooth), max(cycle_smooth)))
cat(sprintf("Irregular range: [%.4f, %.4f]\n", min(irregular), max(irregular)))

# --- Export parameters ---
params <- data.frame(
    parameter = c("sigma2_obs", "sigma2_level", "sigma2_slope", "sigma2_cycle",
                   "lambda_c", "period_quarters", "period_years", "loglik"),
    value = c(sigma2_obs, sigma2_level, sigma2_slope, sigma2_cycle,
              lambda_c, period_est, period_est / 4, loglik)
)
write.csv(params, "results_ucm_params.csv", row.names = FALSE)
cat("\nSaved: results_ucm_params.csv\n")

# --- Export profile likelihood (period search) ---
profile <- data.frame(
    period_quarters = periods,
    period_years = periods / 4,
    loglik = logliks
)
write.csv(profile, "results_ucm_profile_likelihood.csv", row.names = FALSE)
cat("Saved: results_ucm_profile_likelihood.csv\n")

# --- Export decomposition ---
decomp <- data.frame(
    date = as.character(dates),
    y = y,
    level = level_smooth,
    slope = slope_smooth,
    cycle = cycle_smooth,
    irregular = irregular
)
write.csv(decomp, "results_ucm_decomposition.csv", row.names = FALSE)
cat("Saved: results_ucm_decomposition.csv\n")

# --- Export smoothed states with filtered states ---
states <- data.frame(
    date = as.character(dates),
    level_filtered = as.numeric(out$a[-1, "level"]),
    level_smoothed = level_smooth,
    slope_filtered = as.numeric(out$a[-1, "slope"]),
    slope_smoothed = slope_smooth,
    cycle_filtered = as.numeric(out$a[-1, "cycle"]),
    cycle_smoothed = cycle_smooth
)
write.csv(states, "states_ucm_kfas.csv", row.names = FALSE)
cat("Saved: states_ucm_kfas.csv\n")

cat("\nUCM validation complete.\n")
