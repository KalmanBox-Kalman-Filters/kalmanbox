# =============================================================================
# Validacao: Dynamic Factor Model (K=2) via MARSS
#
# Modelo:
#   x_t = B * x_{t-1} + w_t,   w_t ~ N(0, Q)
#   y_t = Z * x_t + v_t,       v_t ~ N(0, R)
#
# Especificacao DFM:
#   - K = 2 fatores latentes (AR(1) independentes)
#   - B = "diagonal and unequal" (phi_1, phi_2)
#   - Q = identidade (identificacao da escala dos fatores)
#   - R = "diagonal and unequal" (variancias idiossincraticas)
#   - x0 = zero, tinitx = 0
#
# Identificacao das cargas (Lambda = Z):
#   Estrutura triangular nas K primeiras linhas de Z:
#     Z[1,2] = 0
#   As demais entradas sao livres. Sem essa restricao, Lambda*F nao e
#   identificado (ambiguidade de rotacao ortogonal).
#
# Pacote: MARSS
# Dataset: us_macro_panel.csv (T=288 meses, N=15 series)
# =============================================================================

library(MARSS)

# ---- Dados ----
panel <- read.csv("../../data/us_macro_panel.csv")
dates <- as.Date(panel$date)
y_mat <- as.matrix(panel[, -1])  # remove coluna de data
series_names <- colnames(y_mat)
N <- ncol(y_mat)
T_obs <- nrow(y_mat)

cat("=============================================================================\n")
cat("  Dynamic Factor Model (K=2) - MARSS Validation\n")
cat("=============================================================================\n\n")

cat(sprintf("Panel dimensions: T=%d months x N=%d series\n", T_obs, N))
cat(sprintf("Date range: %s -> %s\n", as.character(dates[1]),
            as.character(dates[T_obs])))
cat(sprintf("Series: %s\n", paste(series_names, collapse = ", ")))

# MARSS exige Y na forma (N x T) -> series nas linhas
Y <- t(y_mat)
rownames(Y) <- series_names

# =============================================================================
# Especificacao do modelo DFM K=2
# =============================================================================

K <- 2

# --- Z (Lambda): matriz N x K com estrutura triangular para identificacao ---
# Z[i, j] = "zij" (livre)  exceto Z[1, 2] = 0  (restricao de identificacao)
Z_vals <- vector("list", N * K)
dim(Z_vals) <- c(N, K)
for (i in seq_len(N)) {
    for (j in seq_len(K)) {
        if (i == 1 && j == 2) {
            Z_vals[i, j] <- 0
        } else {
            Z_vals[i, j] <- paste0("z", i, "_", j)
        }
    }
}

# --- B (transicao): AR(1) diagonal, parametros distintos ---
B_spec <- "diagonal and unequal"

# --- Q (variancia dos fatores): identidade para fixar escala ---
Q_spec <- "identity"

# --- R (idiossincratica): diagonal com variancias distintas ---
R_spec <- "diagonal and unequal"

# --- x0 e estado inicial ---
x0_spec <- "zero"

# Sem termos lineares (a, U, A, D, C, c)
model_list <- list(
    B = B_spec,
    U = "zero",
    Q = Q_spec,
    Z = Z_vals,
    A = "zero",
    R = R_spec,
    x0 = x0_spec,
    tinitx = 0
)

cat("\n--- Estimando MARSS DFM (K=2) ---\n")
cat("Algoritmo: EM (default), maxit=500\n\n")

# Padroniza series para estabilidade numerica (escala comparavel)
Y_std <- (Y - rowMeans(Y, na.rm = TRUE)) /
    apply(Y, 1, sd, na.rm = TRUE)

set.seed(42)
fit <- MARSS(
    Y_std,
    model = model_list,
    control = list(maxit = 500, conv.test.slope.tol = 0.5),
    method = "kem",
    silent = TRUE
)

cat(sprintf("Convergencia: %s\n",
            ifelse(fit$convergence == 0, "OK", paste0("code=", fit$convergence))))
cat(sprintf("Log-likelihood: %.4f\n", fit$logLik))
cat(sprintf("AIC:            %.4f\n", fit$AIC))
cat(sprintf("AICc:           %.4f\n", fit$AICc))
cat(sprintf("# parameters:   %d\n", fit$num.params))

# =============================================================================
# Extracao de fatores e cargas
# =============================================================================

# Estados suavizados (fatores) -> matriz K x T
factors_smoothed <- fit$states  # K x T
factors_se <- fit$states.se     # K x T

# Cargas (Lambda) reconstruidas em N x K
Lambda_hat <- coef(fit, type = "matrix")$Z  # N x K

# AR(1) coeficientes dos fatores
phi_hat <- diag(coef(fit, type = "matrix")$B)

cat("\n--- Cargas estimadas (Lambda, N x K) ---\n")
loadings_df <- data.frame(
    series = series_names,
    f1 = Lambda_hat[, 1],
    f2 = Lambda_hat[, 2]
)
print(loadings_df, row.names = FALSE, digits = 4)

cat("\n--- Coeficientes AR(1) dos fatores ---\n")
for (j in seq_len(K)) {
    cat(sprintf("  phi_%d = %.4f\n", j, phi_hat[j]))
}

# Variancias idiossincraticas
R_diag <- diag(coef(fit, type = "matrix")$R)
cat("\n--- Variancias idiossincraticas (diag R) ---\n")
for (i in seq_len(N)) {
    cat(sprintf("  %-25s sigma2_eps = %.4f\n", series_names[i], R_diag[i]))
}

# =============================================================================
# Exportacao para CSV
# =============================================================================

# 1. Fatores suavizados com datas
factors_out <- data.frame(
    date = dates,
    f1 = factors_smoothed[1, ],
    f2 = factors_smoothed[2, ],
    f1_se = factors_se[1, ],
    f2_se = factors_se[2, ]
)
write.csv(factors_out, "results_dfm_factors.csv", row.names = FALSE)
cat("\nSaved: results_dfm_factors.csv\n")

# 2. Cargas fatoriais
write.csv(loadings_df, "results_dfm_loadings.csv", row.names = FALSE)
cat("Saved: results_dfm_loadings.csv\n")

# 3. Sumario do ajuste
fit_summary <- data.frame(
    metric = c("logLik", "AIC", "AICc", "n_params", "T", "N", "K",
               "phi_1", "phi_2", "convergence"),
    value = c(fit$logLik, fit$AIC, fit$AICc, fit$num.params,
              T_obs, N, K, phi_hat[1], phi_hat[2], fit$convergence)
)
write.csv(fit_summary, "results_dfm_fit_summary.csv", row.names = FALSE)
cat("Saved: results_dfm_fit_summary.csv\n")

# 4. Variancias idiossincraticas
idio_df <- data.frame(series = series_names, sigma2_eps = R_diag)
write.csv(idio_df, "results_dfm_idio_var.csv", row.names = FALSE)
cat("Saved: results_dfm_idio_var.csv\n")

cat("\nMARSS DFM validation complete.\n")
