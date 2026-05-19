# =============================================================================
# Validacao: Gibbs sampler com FFBS para local-level (Nile)
#
# Referencia Bayesian direta para kalmanbox.estimation.bayesian.BayesianSSM.
#
# Modelo (Durbin-Koopman 2012, cap. 2.3; Petris et al. 2009, cap. 4):
#   y_t   = mu_t + eps_t,   eps_t ~ N(0, sigma2_eps)
#   mu_t  = mu_{t-1} + eta_t, eta_t ~ N(0, sigma2_eta)
#
# Priors conjugados (Petris et al. 2009, 4.5.2):
#   sigma2_eps ~ IG(a.y   = 1, b.y   = 1)
#   sigma2_eta ~ IG(a.th  = 1, b.th  = 1)
# Essas hyperparams geram priors weakly-informative (media ~= 1, variancia
# infinita com a=1). A verossimilhanca domina o posterior para T = 100.
#
# Pacote R: dlm::dlmGibbsDIG implementa o Gibbs sampler exato (FFBS de
# Frühwirth-Schnatter 1994 / Carter-Kohn 1994) + updates IG nas variancias.
# =============================================================================

suppressPackageStartupMessages({
    library(dlm)
    library(coda)
})

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
df <- read.csv("../../data/nile.csv")
y     <- df$flow
years <- df$year
T     <- length(y)

cat("=============================================================================\n")
cat("  Bayesian Local-Level (Gibbs + FFBS) - Nile river flow\n")
cat("=============================================================================\n\n")
cat(sprintf("Sample size : T = %d\n", T))
cat(sprintf("Period      : %d - %d\n", min(years), max(years)))
cat(sprintf("Mean, SD    : %.2f, %.2f\n", mean(y), sd(y)))

# ---------------------------------------------------------------------------
# Modelo local-level e configuracao do Gibbs
# ---------------------------------------------------------------------------
# dlmModPoly(order = 1, dV = v, dW = w): polinomial de ordem 1 = local-level.
# Valores iniciais de dV e dW sao apenas sementes para o sampler.
build_mod <- function() dlmModPoly(order = 1, dV = 1, dW = 1)

n_sample <- 10000
burn     <- 2000
thin     <- 1

priors <- list(a.y = 1, b.y = 1, a.theta = 1, b.theta = 1)
cat("\nPriors:\n")
cat(sprintf("  sigma2_eps ~ IG(a=%.2f, b=%.2f)\n", priors$a.y,     priors$b.y))
cat(sprintf("  sigma2_eta ~ IG(a=%.2f, b=%.2f)\n", priors$a.theta, priors$b.theta))
cat(sprintf("\nGibbs config: n.sample=%d, burn=%d, thin=%d\n",
            n_sample, burn, thin))

# ---------------------------------------------------------------------------
# Rodar 2 cadeias (R-hat requer >= 2 cadeias; diagnostico completo em
# validate_mcmc_diagnostics.R)
# ---------------------------------------------------------------------------
run_chain <- function(seed) {
    set.seed(seed)
    dlmGibbsDIG(y,
                mod         = build_mod(),
                a.y         = priors$a.y,
                b.y         = priors$b.y,
                a.theta     = priors$a.theta,
                b.theta     = priors$b.theta,
                n.sample    = n_sample,
                thin        = thin,
                save.states = TRUE,
                progressBar = FALSE)
}

cat("\n--- Rodando cadeia 1 ---\n")
chain1 <- run_chain(seed = 2026)
cat("--- Rodando cadeia 2 ---\n")
chain2 <- run_chain(seed = 42)

# Extrair draws pos-burn-in
# dlmGibbsDIG: $dV e $dW tem shape (n.sample, m); $theta tem shape
# (T+1, p, n.sample). Para local-level m = p = 1.
idx <- (burn + 1):n_sample
dV1 <- as.numeric(chain1$dV[idx])     # sigma2_eps, cadeia 1
dW1 <- as.numeric(chain1$dW[idx])     # sigma2_eta, cadeia 1
dV2 <- as.numeric(chain2$dV[idx])
dW2 <- as.numeric(chain2$dW[idx])

# States: descartar estado t=0 (pre-amostra) -> indices 2:(T+1)
theta1 <- chain1$theta[2:(T + 1), 1, idx]   # (T, n_kept)
theta2 <- chain2$theta[2:(T + 1), 1, idx]
n_kept <- length(idx)

cat(sprintf("\nKept draws per chain: %d\n", n_kept))

# ---------------------------------------------------------------------------
# Posterior summary (cadeia 1 + cadeia 2 combinadas)
# ---------------------------------------------------------------------------
dV_all <- c(dV1, dV2)
dW_all <- c(dW1, dW2)

post_summary <- function(x) {
    c(mean   = mean(x),
      sd     = sd(x),
      q025   = quantile(x, 0.025, names = FALSE),
      q500   = quantile(x, 0.500, names = FALSE),
      q975   = quantile(x, 0.975, names = FALSE))
}

cat("\n--- Posterior das variancias (cadeias combinadas) ---\n")
cat("sigma2_eps (obs):\n")
print(round(post_summary(dV_all), 4))
cat("\nsigma2_eta (level):\n")
print(round(post_summary(dW_all), 4))

# ---------------------------------------------------------------------------
# R-hat (Gelman-Rubin) via coda
# ---------------------------------------------------------------------------
param_list <- mcmc.list(
    mcmc(cbind(sigma2_eps = dV1, sigma2_eta = dW1)),
    mcmc(cbind(sigma2_eps = dV2, sigma2_eta = dW2))
)
gr <- gelman.diag(param_list, autoburnin = FALSE, multivariate = FALSE)
cat("\n--- Gelman-Rubin R-hat (2 cadeias) ---\n")
print(gr$psrf)

rhat_eps <- gr$psrf["sigma2_eps", "Point est."]
rhat_eta <- gr$psrf["sigma2_eta", "Point est."]
if (any(c(rhat_eps, rhat_eta) > 1.1)) {
    cat("[WARN] R-hat > 1.1 -> mixing pode ser ruim. Aumente n.sample.\n")
} else {
    cat("[OK]   R-hat <= 1.1 para ambas as variancias.\n")
}

# ---------------------------------------------------------------------------
# Posterior dos estados (combinar as 2 cadeias)
# ---------------------------------------------------------------------------
theta_all <- cbind(theta1, theta2)   # (T, 2 * n_kept)
mu_mean <- rowMeans(theta_all)
mu_sd   <- apply(theta_all, 1, sd)
mu_lo   <- apply(theta_all, 1, quantile, probs = 0.025)
mu_hi   <- apply(theta_all, 1, quantile, probs = 0.975)

# ---------------------------------------------------------------------------
# Exportar CSVs
# ---------------------------------------------------------------------------
# (1) Draws das variancias (para comparacao com kalmanbox BayesianPosterior)
draws_df <- data.frame(
    iter        = seq_along(dV_all),
    chain       = c(rep(1L, n_kept), rep(2L, n_kept)),
    sigma2_eps  = dV_all,
    sigma2_eta  = dW_all
)
write.csv(draws_df, "results_gibbs_ffbs_draws.csv", row.names = FALSE)
cat("\nSaved: results_gibbs_ffbs_draws.csv  (", nrow(draws_df), "rows)\n")

# (2) Posterior dos estados
states_df <- data.frame(
    year     = years,
    y        = y,
    mu_mean  = mu_mean,
    mu_sd    = mu_sd,
    mu_q025  = as.numeric(mu_lo),
    mu_q975  = as.numeric(mu_hi)
)
write.csv(states_df, "results_gibbs_ffbs_states.csv", row.names = FALSE)
cat("Saved: results_gibbs_ffbs_states.csv\n")

# (3) Summary
summary_df <- data.frame(
    metric = c(
        "n_sample", "burn", "thin", "n_chains", "n_kept_per_chain",
        "prior_a_y", "prior_b_y", "prior_a_theta", "prior_b_theta",
        "sigma2_eps_mean", "sigma2_eps_sd",
        "sigma2_eps_q025", "sigma2_eps_q500", "sigma2_eps_q975",
        "sigma2_eta_mean", "sigma2_eta_sd",
        "sigma2_eta_q025", "sigma2_eta_q500", "sigma2_eta_q975",
        "rhat_sigma2_eps", "rhat_sigma2_eta"
    ),
    value = c(
        n_sample, burn, thin, 2, n_kept,
        priors$a.y, priors$b.y, priors$a.theta, priors$b.theta,
        post_summary(dV_all)[c("mean", "sd", "q025", "q500", "q975")],
        post_summary(dW_all)[c("mean", "sd", "q025", "q500", "q975")],
        rhat_eps, rhat_eta
    )
)
write.csv(summary_df, "results_gibbs_ffbs_summary.csv", row.names = FALSE)
cat("Saved: results_gibbs_ffbs_summary.csv\n")

cat("\nGibbs FFBS validation complete.\n")
