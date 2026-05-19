# =============================================================================
# Validacao: Stochastic Cycle Model (Brazil IPCA 12-month)
# Pacote: KFAS
#
# Modelo: local level + stochastic damped cycle
#   y_t = mu_t + psi_t + eps_t
#   mu_t = mu_{t-1} + eta_t                 (random walk level)
#   psi_t = rho * cos(lambda) * psi_{t-1} + rho * sin(lambda) * psi*_{t-1} + kappa_t
#   psi*_t = -rho * sin(lambda) * psi_{t-1} + rho * cos(lambda) * psi*_{t-1} + kappa*_t
#
# KFAS parametrization:
#   SSMtrend(degree=1, Q) - random walk level (local level, no slope)
#   SSMcycle(period, Q) - stochastic cycle
#     Damping (rho) is estimated via a custom updatefn that modifies the T matrix.
#     The cycle T matrix is: rho * [[cos(lambda), sin(lambda)], [-sin(lambda), cos(lambda)]]
#     When rho=1, cycle is undamped; rho<1 gives damped (stationary) cycle.
#
# Strategy: Two-stage estimation
#   Stage 1: Profile likelihood over period grid (undamped) to find period
#   Stage 2: Fit with best period, then add damping via custom updatefn
#
# Key outputs: rho, lambda_c, period, extracted cycle component
#
# Dados: Brazil IPCA 12-month inflation, 2000-2023 (monthly)
# =============================================================================

library(KFAS)

cat("=== Stochastic Cycle Validation: Brazil IPCA (KFAS) ===\n\n")

# --- Load data ---
ipca <- read.csv("../../data/brazil_ipca.csv")
y <- ipca$ipca_12m
n <- length(y)
dates <- as.Date(ipca$date)

cat(sprintf("Observations: %d\n", n))
cat(sprintf("Period: %s to %s\n", dates[1], dates[n]))
cat(sprintf("Frequency: monthly\n\n"))

# --- Stage 1: Profile likelihood over cycle period ---
# Search range: 18-96 months (1.5-8 years)
# SSMtrend(1) = local level (random walk)
# SSMcycle(period, Q) = undamped stochastic cycle
periods <- seq(18, 96, by = 1)  # months
logliks_stage1 <- numeric(length(periods))

cat("Stage 1: Searching for optimal cycle period (undamped)...\n")

for (j in seq_along(periods)) {
    p <- periods[j]
    m <- SSModel(
        y ~ SSMtrend(degree = 1, Q = matrix(NA)) +
            SSMcycle(period = p, Q = matrix(NA)),
        H = matrix(NA)
    )
    # 4 NAs: H, Q_level, Q_cycle, Q_cycle*
    f <- fitSSM(m, inits = rep(-1, 4), method = "BFGS",
                control = list(maxit = 500))
    if (f$optim.out$convergence == 0) {
        logliks_stage1[j] <- as.numeric(logLik(f$model))
    } else {
        logliks_stage1[j] <- -Inf
    }
}

best_idx <- which.max(logliks_stage1)
best_period <- periods[best_idx]
cat(sprintf("Best period (undamped): %d months (%.2f years), logLik = %.4f\n\n",
            best_period, best_period / 12, logliks_stage1[best_idx]))

# --- Stage 2: Fit with damping (rho) ---
# Use custom updatefn to include damping factor rho in the T matrix.
# rho scales the cycle rotation matrix: T_cycle = rho * R(lambda)
# Parameters: log(H), log(Q_level), log(Q_cycle), logit(rho)

cat("Stage 2: Fitting damped cycle model...\n")

# Build model with best period
model <- SSModel(
    y ~ SSMtrend(degree = 1, Q = matrix(NA)) +
        SSMcycle(period = best_period, Q = matrix(NA)),
    H = matrix(NA)
)

# Custom update function with damping
# pars[1] = log(H), pars[2] = log(Q_level), pars[3] = log(Q_cycle), pars[4] = logit(rho)
updatefn <- function(pars, model) {
    model$H[1] <- exp(pars[1])
    model$Q[1, 1, 1] <- exp(pars[2])
    model$Q[2, 2, 1] <- exp(pars[3])
    model$Q[3, 3, 1] <- exp(pars[3])  # Same variance for cycle and cycle*

    # Damping factor rho in (0.01, 0.999) via logistic transform
    rho <- 0.01 + 0.989 / (1 + exp(-pars[4]))

    # Modify the T matrix for the cycle block (time-invariant, dim 3 = 1)
    # KFAS stores T as a 3D array; when time-invariant, the third dimension is 1
    lambda_c <- 2 * pi / best_period
    model$T[2, 2, 1] <- rho * cos(lambda_c)
    model$T[2, 3, 1] <- rho * sin(lambda_c)
    model$T[3, 2, 1] <- -rho * sin(lambda_c)
    model$T[3, 3, 1] <- rho * cos(lambda_c)
    model
}

init_pars <- c(
    log(var(y) / 10),  # H
    log(var(y) / 100), # Q_level
    log(var(y) / 10),  # Q_cycle
    2                   # logit(rho) -> rho ~ 0.88
)

fit <- fitSSM(model, inits = init_pars, updatefn = updatefn,
              method = "BFGS", control = list(maxit = 1000))

if (fit$optim.out$convergence != 0) {
    warning("Damped model: optimization did not converge!")
    cat("Falling back to undamped model...\n")
    # Fallback: use undamped model
    model_fallback <- SSModel(
        y ~ SSMtrend(degree = 1, Q = matrix(NA)) +
            SSMcycle(period = best_period, Q = matrix(NA)),
        H = matrix(NA)
    )
    fit <- fitSSM(model_fallback, inits = rep(-1, 4), method = "BFGS")
    rho_est <- 1.0
    cat("Using rho = 1.0 (undamped)\n\n")
} else {
    cat("Damped model optimization converged.\n\n")
    rho_est <- NULL  # will extract from T matrix
}

# --- Extract results ---
fitted_model <- fit$model
out <- KFS(fitted_model, filtering = "state", smoothing = "state")

# Extract variances
sigma2_obs <- as.numeric(fitted_model$H[1])
sigma2_level <- as.numeric(fitted_model$Q[1, 1, 1])
sigma2_cycle <- as.numeric(fitted_model$Q[2, 2, 1])

# Extract rho from T matrix
# T[2,2] = rho * cos(lambda), T[2,3] = rho * sin(lambda)
T22 <- fitted_model$T[2, 2, 1]
T23 <- fitted_model$T[2, 3, 1]
rho <- sqrt(T22^2 + T23^2)
lambda_c <- 2 * pi / best_period
period_est <- best_period

loglik <- as.numeric(logLik(fitted_model))

cat("--- Estimated Parameters ---\n")
cat(sprintf("sigma2_obs   = %.6f\n", sigma2_obs))
cat(sprintf("sigma2_level = %.6f\n", sigma2_level))
cat(sprintf("sigma2_cycle = %.6f\n", sigma2_cycle))
cat(sprintf("rho (damping) = %.6f\n", rho))
cat(sprintf("lambda_c     = %.6f rad\n", lambda_c))
cat(sprintf("period       = %.2f months (%.2f years)\n", period_est, period_est / 12))
cat(sprintf("logLik       = %.4f\n", loglik))

# --- Extract smoothed components ---
state_names <- colnames(out$alphahat)
cat(sprintf("State names: %s\n", paste(state_names, collapse = ", ")))

level_smooth <- as.numeric(out$alphahat[, 1])  # level
cycle_smooth <- as.numeric(out$alphahat[, 2])  # cycle (psi)
cycle_star   <- as.numeric(out$alphahat[, 3])  # cycle* (psi*)

# Irregular
irregular <- y - level_smooth - cycle_smooth

# Cycle amplitude (sqrt(psi^2 + psi*^2))
amplitude <- sqrt(cycle_smooth^2 + cycle_star^2)

cat("\n--- Component Summary ---\n")
cat(sprintf("Level range:     [%.4f, %.4f]\n", min(level_smooth), max(level_smooth)))
cat(sprintf("Cycle range:     [%.4f, %.4f]\n", min(cycle_smooth), max(cycle_smooth)))
cat(sprintf("Amplitude range: [%.4f, %.4f]\n", min(amplitude), max(amplitude)))
cat(sprintf("Irregular SD:    %.4f\n", sd(irregular)))

# --- Export parameters ---
params <- data.frame(
    parameter = c("sigma2_obs", "sigma2_level", "sigma2_cycle",
                   "rho", "lambda_c", "period_months", "period_years", "loglik"),
    value = c(sigma2_obs, sigma2_level, sigma2_cycle,
              rho, lambda_c, period_est, period_est / 12, loglik)
)
write.csv(params, "results_stochastic_cycle_params.csv", row.names = FALSE)
cat("\nSaved: results_stochastic_cycle_params.csv\n")

# --- Export profile likelihood (period search) ---
profile <- data.frame(
    period_months = periods,
    period_years = periods / 12,
    loglik = logliks_stage1
)
write.csv(profile, "results_stochastic_cycle_profile.csv", row.names = FALSE)
cat("Saved: results_stochastic_cycle_profile.csv\n")

# --- Export extracted cycle ---
cycle_df <- data.frame(
    date = as.character(dates),
    y = y,
    level = level_smooth,
    cycle = cycle_smooth,
    cycle_star = cycle_star,
    amplitude = amplitude,
    irregular = irregular
)
write.csv(cycle_df, "results_stochastic_cycle_decomposition.csv", row.names = FALSE)
cat("Saved: results_stochastic_cycle_decomposition.csv\n")

# --- Export smoothed states ---
states <- data.frame(
    date = as.character(dates),
    level_filtered = as.numeric(out$a[-1, 1]),
    level_smoothed = level_smooth,
    cycle_filtered = as.numeric(out$a[-1, 2]),
    cycle_smoothed = cycle_smooth,
    cycle_star_smoothed = cycle_star
)
write.csv(states, "states_stochastic_cycle_kfas.csv", row.names = FALSE)
cat("Saved: states_stochastic_cycle_kfas.csv\n")

cat("\nStochastic Cycle validation complete.\n")
