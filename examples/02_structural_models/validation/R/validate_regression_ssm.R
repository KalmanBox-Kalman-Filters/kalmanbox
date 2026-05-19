# =============================================================================
# Validacao: Regression SSM (UK Gas Consumption)
# Pacote: KFAS
#
# Modelo 1 - Fixed coefficients:
#   y_t = X_t * beta + eps_t
#   beta is constant (equivalent to OLS)
#   In KFAS: SSMregression with Q = 0
#
# Modelo 2 - Time-varying coefficients (TVP):
#   y_t = X_t * beta_t + eps_t
#   beta_t = beta_{t-1} + eta_t   (random walk coefficients)
#   In KFAS: SSMregression with Q = diag(NA, ..., NA)
#
# KFAS parametrization:
#   SSMregression(~ X, Q = ...) specifies regression with state-space structure
#   When Q = 0, coefficients are fixed (diffuse initialization recovers OLS)
#   When Q = diag(sigma2_1, ..., sigma2_k), coefficients evolve as random walks
#
# Dados: UK quarterly gas consumption with trend and seasonal regressors
# =============================================================================

library(KFAS)

cat("=== Regression SSM Validation: UK Gas (KFAS) ===\n\n")

# --- Load data ---
gas <- read.csv("../../data/uk_gas.csv")
y <- gas$gas_consumption
n <- length(y)
dates <- as.Date(gas$date)

cat(sprintf("Observations: %d\n", n))
cat(sprintf("Period: %s to %s\n", dates[1], dates[n]))
cat(sprintf("Frequency: quarterly\n\n"))

# --- Construct regressors ---
# Intercept, linear trend, quarterly dummies (Q2, Q3, Q4 vs Q1 baseline)
intercept <- rep(1, n)
trend <- 1:n
quarters <- as.numeric(format(dates, "%m"))
quarter_num <- ceiling(quarters / 3)
q2 <- as.numeric(quarter_num == 2)
q3 <- as.numeric(quarter_num == 3)
q4 <- as.numeric(quarter_num == 4)

X <- cbind(intercept, trend, q2, q3, q4)
k <- ncol(X)
regressor_names <- c("intercept", "trend", "Q2", "Q3", "Q4")

cat(sprintf("Regressors: %s\n", paste(regressor_names, collapse = ", ")))
cat(sprintf("Number of regressors: %d\n\n", k))

# =============================================================================
# Model 1: Fixed Coefficients (equivalent to OLS)
# =============================================================================
cat("--- Model 1: Fixed Coefficients ---\n")

# SSMregression with Q=0 -> fixed coefficients
# Use diffuse initialization (default in KFAS) to recover OLS solution
X_df <- data.frame(X)
colnames(X_df) <- regressor_names

model_fixed <- SSModel(
    y ~ -1 + SSMregression(~ intercept + trend + q2 + q3 + q4,
                            data = X_df,
                            Q = diag(0, k)),
    H = matrix(NA)
)

# For fixed coefficients, only H needs to be estimated
fit_fixed <- fitSSM(model_fixed,
                    inits = c(log(var(y) / 10)),
                    method = "BFGS")

if (fit_fixed$optim.out$convergence != 0) {
    warning("Fixed model: optimization did not converge!")
}

out_fixed <- KFS(fit_fixed$model, filtering = "state", smoothing = "state")

# Extract coefficients (smoothed states at last time point = OLS estimates)
coefs_fixed <- as.numeric(out_fixed$alphahat[n, ])
sigma2_obs_fixed <- as.numeric(fit_fixed$model$H[1])
loglik_fixed <- as.numeric(logLik(fit_fixed$model))

cat("Estimated coefficients (should match OLS):\n")
for (i in 1:k) {
    cat(sprintf("  %-12s = %10.4f\n", regressor_names[i], coefs_fixed[i]))
}
cat(sprintf("sigma2_obs   = %.4f\n", sigma2_obs_fixed))
cat(sprintf("logLik       = %.4f\n\n", loglik_fixed))

# OLS comparison using lm()
ols <- lm(y ~ 0 + X)
cat("OLS comparison (lm):\n")
for (i in 1:k) {
    cat(sprintf("  %-12s = %10.4f\n", regressor_names[i], coef(ols)[i]))
}
cat(sprintf("sigma2_ols   = %.4f\n", summary(ols)$sigma^2))
cat(sprintf("logLik_ols   = %.4f\n\n", as.numeric(logLik(ols))))

# =============================================================================
# Model 2: Time-Varying Parameters (TVP)
# =============================================================================
cat("--- Model 2: Time-Varying Parameters ---\n")

# SSMregression with Q = diag(NA, ..., NA) -> random walk coefficients
model_tvp <- SSModel(
    y ~ -1 + SSMregression(~ intercept + trend + q2 + q3 + q4,
                            data = X_df,
                            Q = diag(NA, k)),
    H = matrix(NA)
)

# Update function for TVP: estimate H and k diagonal elements of Q
updatefn_tvp <- function(pars, model) {
    model$H[1] <- exp(pars[1])
    diag(model$Q[1:k, 1:k, 1]) <- exp(pars[2:(k + 1)])
    model
}

init_tvp <- c(log(var(y) / 10), rep(log(var(y) / 1000), k))

fit_tvp <- fitSSM(model_tvp, inits = init_tvp, updatefn = updatefn_tvp,
                  method = "BFGS", control = list(maxit = 1000))

if (fit_tvp$optim.out$convergence != 0) {
    warning("TVP model: optimization did not converge!")
}

out_tvp <- KFS(fit_tvp$model, filtering = "state", smoothing = "state")

# Extract parameters
sigma2_obs_tvp <- as.numeric(fit_tvp$model$H[1])
sigma2_betas <- diag(fit_tvp$model$Q[1:k, 1:k, 1])
loglik_tvp <- as.numeric(logLik(fit_tvp$model))

cat("Estimated variance parameters:\n")
cat(sprintf("  sigma2_obs       = %.6f\n", sigma2_obs_tvp))
for (i in 1:k) {
    cat(sprintf("  sigma2_%-9s = %.6f\n", regressor_names[i], sigma2_betas[i]))
}
cat(sprintf("logLik             = %.4f\n\n", loglik_tvp))

# Time-varying coefficients (smoothed)
coefs_tvp <- out_tvp$alphahat  # n x k matrix
cat("TVP coefficients at final time point:\n")
for (i in 1:k) {
    cat(sprintf("  %-12s = %10.4f\n", regressor_names[i], coefs_tvp[n, i]))
}

# =============================================================================
# Model Comparison: Fixed vs TVP
# =============================================================================
cat("\n--- Model Comparison ---\n")

n_params_fixed <- 1  # only sigma2_obs (coefficients from diffuse init)
n_params_tvp <- 1 + k  # sigma2_obs + k sigma2_beta

aic_fixed <- -2 * loglik_fixed + 2 * n_params_fixed
bic_fixed <- -2 * loglik_fixed + n_params_fixed * log(n)
aic_tvp <- -2 * loglik_tvp + 2 * n_params_tvp
bic_tvp <- -2 * loglik_tvp + n_params_tvp * log(n)

cat(sprintf("%-25s %12s %12s\n", "Metric", "Fixed", "TVP"))
cat(sprintf("%-25s %12.4f %12.4f\n", "Log-likelihood", loglik_fixed, loglik_tvp))
cat(sprintf("%-25s %12.4f %12.4f\n", "AIC", aic_fixed, aic_tvp))
cat(sprintf("%-25s %12.4f %12.4f\n", "BIC", bic_fixed, bic_tvp))

preferred <- ifelse(bic_tvp < bic_fixed, "TVP", "Fixed")
cat(sprintf("\nPreferred model (BIC): %s\n", preferred))

# Signal-to-noise ratios for coefficient evolution
cat("\nSignal-to-noise ratios (sigma2_beta / sigma2_obs):\n")
for (i in 1:k) {
    snr <- sigma2_betas[i] / sigma2_obs_tvp
    significance <- ifelse(snr > 0.01, "SIGNIFICANT", "negligible")
    cat(sprintf("  %-12s: SNR = %.6f (%s)\n", regressor_names[i], snr, significance))
}

# =============================================================================
# Export results
# =============================================================================

# --- Parameters ---
params_comparison <- data.frame(
    parameter = c(
        paste0("fixed_coef_", regressor_names),
        "fixed_sigma2_obs", "fixed_loglik", "fixed_aic", "fixed_bic",
        paste0("tvp_coef_final_", regressor_names),
        paste0("tvp_sigma2_", regressor_names),
        "tvp_sigma2_obs", "tvp_loglik", "tvp_aic", "tvp_bic"
    ),
    value = c(
        coefs_fixed,
        sigma2_obs_fixed, loglik_fixed, aic_fixed, bic_fixed,
        coefs_tvp[n, ],
        sigma2_betas,
        sigma2_obs_tvp, loglik_tvp, aic_tvp, bic_tvp
    )
)
write.csv(params_comparison, "results_regression_ssm_params.csv", row.names = FALSE)
cat("\nSaved: results_regression_ssm_params.csv\n")

# --- TVP coefficient evolution ---
coefs_evolution <- data.frame(
    date = as.character(dates)
)
for (i in 1:k) {
    coefs_evolution[[paste0("beta_", regressor_names[i])]] <- as.numeric(coefs_tvp[, i])
}
write.csv(coefs_evolution, "results_regression_ssm_tvp_coefs.csv", row.names = FALSE)
cat("Saved: results_regression_ssm_tvp_coefs.csv\n")

# --- Fixed coefficients comparison ---
coefs_compare <- data.frame(
    regressor = regressor_names,
    ssm_fixed = coefs_fixed,
    ols = coef(ols),
    difference = abs(coefs_fixed - coef(ols))
)
write.csv(coefs_compare, "results_regression_ssm_fixed_vs_ols.csv", row.names = FALSE)
cat("Saved: results_regression_ssm_fixed_vs_ols.csv\n")

cat("\nRegression SSM validation complete.\n")
