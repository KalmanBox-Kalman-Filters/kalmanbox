# =============================================================================
# Validacao: Accelerationist Phillips Curve (Staiger-Stock-Watson) via dlm
#
# Modelo:
#   Delta pi_t = alpha_t + beta_t * u_t + eps_t,   eps_t  ~ N(0, V)
#   alpha_t    = alpha_{t-1} + eta_a_t,            eta_a  ~ N(0, w_a)
#   beta_t     = beta_{t-1}  + eta_b_t,            eta_b  ~ N(0, w_b)
#
#   NAIRU_t = -alpha_t / beta_t   (unemployment rate consistent com Delta pi=0)
#
# Especificacao dlm:
#   dlmModReg(U, addInt = TRUE, dV, dW) instancia F_t = [1, u_t] com estado
#   (alpha_t, beta_t) evoluindo como random walk. Essa e a especificacao SSW
#   (Model 3 no notebook kalmanbox): ambos parametros sao time-varying.
#
# bvarsv (Primiceri 2005 TVP-VAR Bayesiano):
#   O pacote bvarsv implementa o modelo TVP-VAR com volatilidade estocastica.
#   Se instalado, ajustamos um modelo VAR(1) bivariado com (Delta pi, u) para
#   obter referencia Bayesiana para a evolucao dos coeficientes de u sobre
#   Delta pi. Caso nao esteja disponivel, pulamos essa parte (apenas dlm).
#
# Dataset: us_inflation_unemployment.csv (quarterly 1960Q1-2023Q4)
# =============================================================================

library(dlm)

HAS_BVARSV <- requireNamespace("bvarsv", quietly = TRUE)
if (HAS_BVARSV) {
    suppressPackageStartupMessages(library(bvarsv))
    cat("[info] bvarsv disponivel - executara TVP-VAR Bayesiano\n")
} else {
    cat("[info] bvarsv nao instalado - pulando validacao TVP-VAR\n")
}

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
df <- read.csv("../../data/us_inflation_unemployment.csv")
df$date <- as.Date(df$date)
dates <- df$date
pi    <- df$inflation
u     <- df$unemployment
n     <- length(pi)

# Primeira diferenca da inflacao (acceleracionista)
dpi     <- c(NA, diff(pi))
mask    <- is.finite(dpi)
dpi_v   <- dpi[mask]
u_v     <- u[mask]
dates_v <- dates[mask]
nv      <- length(dpi_v)

cat("=============================================================================\n")
cat("  Accelerationist Phillips Curve - dlm (+ bvarsv) Validation\n")
cat("=============================================================================\n\n")
cat(sprintf("Observations used: %d (dropping first due to differencing)\n", nv))
cat(sprintf("Period: %s -> %s\n",
            as.character(dates_v[1]), as.character(dates_v[nv])))

# Referencia OLS: Delta pi = alpha + beta * u
ols       <- lm(dpi_v ~ u_v)
alpha_ols <- unname(coef(ols)[1])
beta_ols  <- unname(coef(ols)[2])
nairu_ols <- if (abs(beta_ols) > 1e-8) -alpha_ols / beta_ols else NA
cat(sprintf("\nOLS alpha = %.4f, beta = %.4f\n", alpha_ols, beta_ols))
cat(sprintf("OLS-implied constant NAIRU = %.3f %%\n", nairu_ols))

# ---------------------------------------------------------------------------
# dlm Model 3 (SSW): alpha_t e beta_t ambos random walk
# ---------------------------------------------------------------------------
build_pc <- function(pars) {
    dlmModReg(
        X      = matrix(u_v, ncol = 1),
        addInt = TRUE,
        dV     = exp(pars[1]),
        dW     = exp(pars[2:3])
    )
}

cat("\n--- Estimando dlm SSW Phillips Curve (alpha_t, beta_t random walk) ---\n")

set.seed(42)
init_pc <- c(log(0.5), log(0.01), log(0.001))
fit_pc <- dlmMLE(dpi_v, parm = init_pc, build = build_pc,
                 method = "L-BFGS-B",
                 control = list(maxit = 500, trace = 0))

cat(sprintf("Convergencia: %s\n",
            ifelse(fit_pc$convergence == 0, "OK",
                   paste0("code=", fit_pc$convergence))))

V_pc  <- exp(fit_pc$par[1])
W_pc  <- exp(fit_pc$par[2:3])
ll_pc <- -fit_pc$value
cat(sprintf("V (sigma2_obs) = %.6f\n", V_pc))
cat(sprintf("W_alpha        = %.6f\n", W_pc[1]))
cat(sprintf("W_beta         = %.6f\n", W_pc[2]))
cat(sprintf("loglike (dlm)  = %.4f\n", ll_pc))

mod_pc   <- build_pc(fit_pc$par)
filt_pc  <- dlmFilter(dpi_v, mod_pc)
smooth_pc <- dlmSmooth(filt_pc)

alpha_t <- smooth_pc$s[-1, 1]
beta_t  <- smooth_pc$s[-1, 2]

cov_list_pc <- dlmSvd2var(smooth_pc$U.S, smooth_pc$D.S)[-1]
P00 <- sapply(cov_list_pc, function(M) M[1, 1])
P11 <- sapply(cov_list_pc, function(M) M[2, 2])
P01 <- sapply(cov_list_pc, function(M) M[1, 2])
alpha_se <- sqrt(pmax(P00, 0))
beta_se  <- sqrt(pmax(P11, 0))

# NAIRU com SE via delta method: u* = -alpha/beta
#   d u*/d alpha = -1/beta,  d u*/d beta = alpha/beta^2
beta_safe <- ifelse(abs(beta_t) < 1e-3, sign(beta_t + 1e-12) * 1e-3, beta_t)
nairu_t <- -alpha_t / beta_safe
g1 <- -1 / beta_safe
g2 <-  alpha_t / (beta_safe^2)
var_nairu <- g1^2 * P00 + g2^2 * P11 + 2 * g1 * g2 * P01
var_nairu <- pmax(var_nairu, 0)
nairu_se  <- sqrt(var_nairu)

frac_in_band <- mean(nairu_t >= 3 & nairu_t <= 8)
frac_neg_beta <- mean(beta_t < 0)

cat(sprintf("\nNAIRU (dlm): mean=%.3f, min=%.3f, max=%.3f\n",
            mean(nairu_t), min(nairu_t), max(nairu_t)))
cat(sprintf("Fracao em [3, 8]: %.3f\n", frac_in_band))
cat(sprintf("Fracao com beta_t < 0: %.3f\n", frac_neg_beta))

# Sanity: pre vs pos 1990 (flattening literatura)
pre_1990  <- dates_v < as.Date("1990-01-01")
cat(sprintf("Slope pre-1990  mean: %.4f\n", mean(beta_t[pre_1990])))
cat(sprintf("Slope post-1990 mean: %.4f\n", mean(beta_t[!pre_1990])))

# ---------------------------------------------------------------------------
# Exportacao principal (dlm)
# ---------------------------------------------------------------------------
out_pc <- data.frame(
    date       = dates_v,
    alpha      = alpha_t,
    alpha_se   = alpha_se,
    beta       = beta_t,
    beta_se    = beta_se,
    beta_lo    = beta_t  - 1.96 * beta_se,
    beta_hi    = beta_t  + 1.96 * beta_se,
    nairu      = nairu_t,
    nairu_se   = nairu_se,
    nairu_lo   = nairu_t - 1.96 * nairu_se,
    nairu_hi   = nairu_t + 1.96 * nairu_se
)
write.csv(out_pc, "results_phillips_curve_dlm.csv", row.names = FALSE)
cat("\nSaved: results_phillips_curve_dlm.csv\n")

summary_pc <- data.frame(
    metric = c("logLik", "V_sigma2_obs", "W_alpha", "W_beta",
               "ols_alpha", "ols_beta", "ols_nairu",
               "mean_nairu", "min_nairu", "max_nairu",
               "frac_nairu_in_3_8", "frac_neg_beta",
               "mean_beta_pre_1990", "mean_beta_post_1990",
               "n_obs", "convergence"),
    value = c(ll_pc, V_pc, W_pc[1], W_pc[2],
              alpha_ols, beta_ols, nairu_ols,
              mean(nairu_t), min(nairu_t), max(nairu_t),
              frac_in_band, frac_neg_beta,
              mean(beta_t[pre_1990]), mean(beta_t[!pre_1990]),
              nv, fit_pc$convergence)
)
write.csv(summary_pc, "results_phillips_curve_summary.csv", row.names = FALSE)
cat("Saved: results_phillips_curve_summary.csv\n")

# ---------------------------------------------------------------------------
# bvarsv: TVP-VAR Bayesiano (opcional)
# ---------------------------------------------------------------------------
if (HAS_BVARSV) {
    cat("\n--- Estimando bvarsv TVP-VAR (Primiceri 2005) ---\n")
    Y_var <- cbind(Delta_pi = dpi_v, u = u_v)
    set.seed(42)
    # nrep/nburn pequenos para rodar rapido no runner. Defaults recomendados
    # pelo pacote sao >= 5000; ajuste se quiser producao.
    fit_bvarsv <- tryCatch(
        bvar.sv.tvp(Y_var, p = 1, nrep = 2000, nburn = 1000, thinfac = 1),
        error = function(e) {
            cat(sprintf("[warn] bvarsv falhou: %s\n", conditionMessage(e)))
            NULL
        }
    )

    if (!is.null(fit_bvarsv)) {
        # Extrai coeficientes medios posteriores em cada t
        # Beta.postmean tem dimensao (K x K*p+1 x T), mas API exata depende
        # da versao do pacote - usamos acessor seguro.
        beta_draws <- fit_bvarsv$Beta.postmean
        cat(sprintf("  dim(Beta.postmean): %s\n",
                    paste(dim(beta_draws), collapse = " x ")))
        # coef de u na equacao de Delta_pi: posicao (1, 3, t) em VAR(1) bivariado
        # equacao: Delta_pi_t = c + a11*Delta_pi_{t-1} + a12*u_{t-1}
        # layout Primiceri: [intercept, lag1_var1, lag1_var2]
        a12_t <- tryCatch(beta_draws[1, 3, ], error = function(e) NULL)
        if (!is.null(a12_t) && length(a12_t) == nv) {
            cat(sprintf("  Coef u_{t-1} na eq Delta_pi: mean=%.4f, sd=%.4f\n",
                        mean(a12_t), sd(a12_t)))
            out_bvarsv <- data.frame(
                date              = dates_v,
                coef_u_on_dpi_bvar = a12_t
            )
            write.csv(out_bvarsv, "results_phillips_curve_bvarsv.csv",
                      row.names = FALSE)
            cat("Saved: results_phillips_curve_bvarsv.csv\n")
        } else {
            cat("[warn] Nao foi possivel extrair Beta.postmean compatively\n")
        }
    }
} else {
    cat("\n[info] bvarsv nao executado (instale com install.packages('bvarsv'))\n")
}

cat("\ndlm Phillips curve validation complete.\n")
