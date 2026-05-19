# =============================================================================
# Validacao: MCMC diagnostics para o sampler Gibbs/FFBS (local-level, Nile)
#
# Executa 4 cadeias independentes com dlm::dlmGibbsDIG e calcula:
#   - Gelman-Rubin R-hat (coda::gelman.diag)
#   - Effective Sample Size (coda::effectiveSize)
#   - Autocorrelacao (coda::autocorr)
# Em seguida compara posterior mean com MLE obtida via KFAS::fitSSM.
#
# Referencias:
#   - Gelman et al. (2014, Bayesian Data Analysis, cap. 11)
#   - Vehtari et al. (2021), 'Rank-normalization, folding, and localization:
#     an improved R-hat for assessing convergence of MCMC'
#   - coda package: Plummer, Best, Cowles, Vines (2006)
#
# Priors: mesmos weakly-informative IG(1,1) usados em validate_gibbs_ffbs.R
# para manter consistencia entre diagnosticos.
# =============================================================================

suppressPackageStartupMessages({
    library(dlm)
    library(coda)
    library(KFAS)
})

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
df    <- read.csv("../../data/nile.csv")
y     <- df$flow
years <- df$year
T     <- length(y)

cat("=============================================================================\n")
cat("  MCMC Diagnostics - Gibbs/FFBS local-level (Nile)\n")
cat("=============================================================================\n\n")
cat(sprintf("Sample size: T = %d\n", T))

# ---------------------------------------------------------------------------
# Configuracao Gibbs
# ---------------------------------------------------------------------------
n_sample <- 10000
burn     <- 2000
thin     <- 1

# Priors IG(1, 1) (ver validate_gibbs_ffbs.R para justificativa)
a_y     <- 1; b_y     <- 1
a_theta <- 1; b_theta <- 1

build_mod <- function() dlmModPoly(order = 1, dV = 1, dW = 1)

run_chain <- function(seed) {
    set.seed(seed)
    dlmGibbsDIG(y,
                mod         = build_mod(),
                a.y         = a_y,     b.y     = b_y,
                a.theta     = a_theta, b.theta = b_theta,
                n.sample    = n_sample,
                thin        = thin,
                save.states = TRUE,
                progressBar = FALSE)
}

# ---------------------------------------------------------------------------
# Rodar 4 cadeias com sementes diferentes
# ---------------------------------------------------------------------------
seeds <- c(2026, 42, 13, 2718)
cat(sprintf("\nRunning %d chains (seeds: %s)...\n",
            length(seeds), paste(seeds, collapse = ", ")))

chains <- vector("list", length(seeds))
for (i in seq_along(seeds)) {
    cat(sprintf("  chain %d (seed=%d)...\n", i, seeds[i]))
    chains[[i]] <- run_chain(seeds[i])
}

idx <- (burn + 1):n_sample
n_kept <- length(idx)

# Extrair draws das variancias
dV_mat <- sapply(chains, function(c) as.numeric(c$dV[idx]))   # (n_kept, 4)
dW_mat <- sapply(chains, function(c) as.numeric(c$dW[idx]))

# ---------------------------------------------------------------------------
# Construir mcmc.list para coda
# ---------------------------------------------------------------------------
mcmc_list <- mcmc.list(lapply(seq_along(seeds), function(i) {
    mcmc(cbind(sigma2_eps = dV_mat[, i],
               sigma2_eta = dW_mat[, i]))
}))

# ---------------------------------------------------------------------------
# Diagnosticos
# ---------------------------------------------------------------------------
cat("\n--- Gelman-Rubin R-hat (4 cadeias) ---\n")
gr <- gelman.diag(mcmc_list, autoburnin = FALSE, multivariate = FALSE)
print(gr$psrf)

cat("\n--- Effective Sample Size ---\n")
ess_per_chain <- sapply(mcmc_list, effectiveSize)
ess_total     <- effectiveSize(mcmc_list)
cat("ESS por cadeia:\n"); print(round(ess_per_chain, 1))
cat("\nESS total (4 cadeias concatenadas):\n"); print(round(ess_total, 1))

cat("\n--- Autocorrelacao (lag-1, lag-5, lag-10) - media sobre cadeias ---\n")
acf_lags <- c(1, 5, 10)
acf_tab <- sapply(mcmc_list, function(x) {
    a <- autocorr(x, lags = acf_lags)   # (length(lags), p, p)
    c(sigma2_eps = a[, "sigma2_eps", "sigma2_eps"],
      sigma2_eta = a[, "sigma2_eta", "sigma2_eta"])
})
acf_mean <- rowMeans(acf_tab)
print(round(acf_mean, 4))

# ---------------------------------------------------------------------------
# Comparacao Bayes (posterior mean) x MLE (KFAS::fitSSM)
# ---------------------------------------------------------------------------
cat("\n--- MLE via KFAS::fitSSM (referencia frequentista) ---\n")
# Local-level com H = sigma2_eps, Q = sigma2_eta (ambos NA para estimar)
model_mle <- SSModel(y ~ SSMtrend(degree = 1, Q = list(NA)), H = NA)
# Inits em log-scale para BFGS (fitSSM expoe parametros via updatefn padrao)
init_pars <- log(c(var(y) * 0.9, var(y) * 0.1))
fit_mle <- fitSSM(model_mle, inits = init_pars, method = "BFGS")
mle_H <- as.numeric(fit_mle$model$H)           # sigma2_eps
mle_Q <- as.numeric(fit_mle$model$Q[, , 1])    # sigma2_eta
cat(sprintf("MLE sigma2_eps = %.2f\n", mle_H))
cat(sprintf("MLE sigma2_eta = %.2f\n", mle_Q))

# Posterior mean combinada (4 cadeias)
post_eps_mean <- mean(dV_mat)
post_eta_mean <- mean(dW_mat)
post_eps_sd   <- sd(as.numeric(dV_mat))
post_eta_sd   <- sd(as.numeric(dW_mat))

cat("\n--- Posterior (Bayes) vs MLE ---\n")
cmp <- data.frame(
    parameter = c("sigma2_eps", "sigma2_eta"),
    MLE       = c(mle_H, mle_Q),
    post_mean = c(post_eps_mean, post_eta_mean),
    post_sd   = c(post_eps_sd,   post_eta_sd),
    rel_diff_pct = c(
        100 * abs(mle_H - post_eps_mean) / abs(mle_H),
        100 * abs(mle_Q - post_eta_mean) / abs(mle_Q)
    )
)
print(cmp, row.names = FALSE)

# ---------------------------------------------------------------------------
# Exportar CSVs
# ---------------------------------------------------------------------------
diag_df <- data.frame(
    parameter = c("sigma2_eps", "sigma2_eta"),
    rhat      = gr$psrf[, "Point est."],
    rhat_upper= gr$psrf[, "Upper C.I."],
    ess_total = as.numeric(ess_total),
    acf_lag1  = c(acf_mean["sigma2_eps.Lag 1"],  acf_mean["sigma2_eta.Lag 1"]),
    acf_lag5  = c(acf_mean["sigma2_eps.Lag 5"],  acf_mean["sigma2_eta.Lag 5"]),
    acf_lag10 = c(acf_mean["sigma2_eps.Lag 10"], acf_mean["sigma2_eta.Lag 10"])
)
write.csv(diag_df, "results_mcmc_diagnostics.csv", row.names = FALSE)
cat("\nSaved: results_mcmc_diagnostics.csv\n")

# Draws das 4 cadeias (longo formato) para comparacao downstream
draws_df <- data.frame(
    iter        = rep(seq_len(n_kept), times = length(seeds)),
    chain       = rep(seq_along(seeds), each = n_kept),
    sigma2_eps  = as.numeric(dV_mat),
    sigma2_eta  = as.numeric(dW_mat)
)
write.csv(draws_df, "results_mcmc_draws.csv", row.names = FALSE)
cat(sprintf("Saved: results_mcmc_draws.csv  (%d rows)\n", nrow(draws_df)))

# Bayes vs MLE
cmp_out <- cmp
cmp_out$ess_total <- ess_total[c("sigma2_eps", "sigma2_eta")]
cmp_out$rhat      <- gr$psrf[c("sigma2_eps", "sigma2_eta"), "Point est."]
write.csv(cmp_out, "results_bayes_vs_mle.csv", row.names = FALSE)
cat("Saved: results_bayes_vs_mle.csv\n")

# Alert se R-hat alto ou ESS baixo
if (any(gr$psrf[, "Point est."] > 1.1)) {
    cat("[WARN] R-hat > 1.1 -> cadeias nao convergiram adequadamente.\n")
} else {
    cat("[OK]   R-hat <= 1.1 para todos os parametros.\n")
}
if (any(ess_total < 400)) {
    cat("[WARN] ESS < 400 -> poucas draws efetivas para inferencia estavel.\n")
} else {
    cat("[OK]   ESS >= 400 para todos os parametros.\n")
}

cat("\nMCMC diagnostics validation complete.\n")
