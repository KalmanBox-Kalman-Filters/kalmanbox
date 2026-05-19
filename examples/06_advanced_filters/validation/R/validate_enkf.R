# =============================================================================
# Validacao: Ensemble Kalman Filter (EnKF) para Lorenz 63
#
# Modelo:
#   Estado (3D, Lorenz 1963):
#     dx/dt = sigma * (y - x)
#     dy/dt = x * (rho - z) - y
#     dz/dt = x * y - beta * z
#   Observacao (2D): y_obs = [x, z] + v,  v ~ N(0, sigma_obs^2 * I)
#
# Pacotes R: NAO HA pacote CRAN maduro para EnKF (o antigo 'ENsembleKalman'
# foi removido; 'EnKF' nao existe). Por isso implementamos manualmente o
# Stochastic EnKF (Evensen 1994/2009), que e o baseline classico:
#
#   Predicao  : x_pred^{i} = f(x^{i}) + w^{i},   w^{i} ~ N(0, Q)
#   Update    : usa observacoes perturbadas y^{i} = y + v^{i}, v^{i} ~ N(0, R)
#               K_ens = P_xy * (P_yy)^{-1}
#               x^{i} <- x^{i} + K_ens * (y^{i} - h(x^{i}))
#
# Como comparativo adicional rodamos o EKF (mesmo kernel do validate_ekf_ukf.R)
# estendido para Lorenz 63, para mostrar que EnKF com N grande se aproxima do
# EKF (apos convergencia transitoria).
#
# EXPORTACAO:
#   - estados EnKF para N = 10, 20, 50, 100, 200, 500
#   - RMSE vs N do ensemble
#   - estados do EKF como referencia
# =============================================================================

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
df <- read.csv("../../data/lorenz63.csv")
t_grid <- df$t
x_true <- cbind(df$x_x, df$x_y, df$x_z)
y_obs  <- cbind(df$y_x_obs, df$y_z_obs)
n  <- nrow(df)
dt <- t_grid[2] - t_grid[1]

# parametros do gerador
sigma_p <- 10.0
rho     <- 28.0
beta    <- 8.0 / 3.0
sigma_obs <- 2.0
R_obs <- diag(c(sigma_obs^2, sigma_obs^2))
# Ruido de processo pequeno injetado para estabilidade do EnKF (tipico)
Q_mat <- diag(c(1e-2, 1e-2, 1e-2))

cat("=============================================================================\n")
cat("  EnKF - Lorenz 63 Validation (R)\n")
cat("=============================================================================\n\n")
cat(sprintf("Sample size n = %d, dt = %.4f\n", n, dt))
cat(sprintf("Observation : y = [x, z] + N(0, %.2f^2)\n", sigma_obs))

# ---------------------------------------------------------------------------
# Dinamica (RK4, identica ao gerador Python)
# ---------------------------------------------------------------------------
lorenz_rhs <- function(s) {
    c(sigma_p * (s[2] - s[1]),
      s[1] * (rho - s[3]) - s[2],
      s[1] * s[2] - beta * s[3])
}
f_dyn <- function(s) {
    k1 <- lorenz_rhs(s)
    k2 <- lorenz_rhs(s + 0.5 * dt * k1)
    k3 <- lorenz_rhs(s + 0.5 * dt * k2)
    k4 <- lorenz_rhs(s + dt * k3)
    s + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
}
H_mat <- matrix(c(1, 0, 0,
                  0, 0, 1), nrow = 2, byrow = TRUE)

# Jacobiano da dinamica (RK4 -> usamos EKF no sistema continuo aproximado)
F_jac <- function(s) {
    # Usa Jacobiano de Euler para simplicidade (EKF como baseline, nao como
    # referencia exata). Para Lorenz, a aproximacao basta para definir escala.
    x <- s[1]; y <- s[2]; z <- s[3]
    A <- matrix(c(-sigma_p, sigma_p, 0,
                  rho - z, -1,     -x,
                  y,        x,     -beta), nrow = 3, byrow = TRUE)
    diag(3) + dt * A
}

# ---------------------------------------------------------------------------
# EKF para referencia
# ---------------------------------------------------------------------------
run_ekf <- function() {
    x_filt <- matrix(0, n, 3)
    s <- c(1, 1, 1)
    P <- diag(3) * 1.0
    for (k in seq_len(n)) {
        if (k > 1) {
            Fk <- F_jac(s)
            s  <- f_dyn(s)
            P  <- Fk %*% P %*% t(Fk) + Q_mat
        }
        v <- y_obs[k, ] - as.numeric(H_mat %*% s)
        S <- H_mat %*% P %*% t(H_mat) + R_obs
        K <- P %*% t(H_mat) %*% solve(S)
        s <- s + K %*% v
        P <- (diag(3) - K %*% H_mat) %*% P
        x_filt[k, ] <- as.numeric(s)
    }
    x_filt
}

cat("\n--- EKF (referencia) ---\n")
ekf_states <- run_ekf()
rmse_ekf <- sqrt(colMeans((ekf_states - x_true)^2))
cat(sprintf("RMSE EKF (x, y, z): (%.4f, %.4f, %.4f)\n",
            rmse_ekf[1], rmse_ekf[2], rmse_ekf[3]))

# ---------------------------------------------------------------------------
# Stochastic EnKF (Evensen)
# ---------------------------------------------------------------------------
run_enkf <- function(N, seed = 0) {
    set.seed(seed)
    d <- 3
    # ensemble inicial: N(1,1,1) com alguma dispersao
    ens <- matrix(rep(c(1, 1, 1), each = N), nrow = N, ncol = d) +
        matrix(rnorm(N * d, sd = 1.0), nrow = N, ncol = d)
    # cholesky do ruido
    LQ <- t(chol(Q_mat))
    LR <- t(chol(R_obs))
    x_filt <- matrix(0, n, d)
    for (k in seq_len(n)) {
        if (k > 1) {
            # propagacao ensemble com ruido de processo
            for (i in seq_len(N)) {
                w <- LQ %*% rnorm(d)
                ens[i, ] <- f_dyn(ens[i, ]) + as.numeric(w)
            }
        }
        # media e anomalias
        mu <- colMeans(ens)
        A  <- sweep(ens, 2, mu)           # N x d
        Y  <- A %*% t(H_mat)              # N x p  (h(x) linear aqui)
        # covariancias amostrais (1/(N-1))
        Pxy <- (t(A) %*% Y) / (N - 1)     # d x p
        Pyy <- (t(Y) %*% Y) / (N - 1) + R_obs
        K_ens <- Pxy %*% solve(Pyy)
        # update com observacoes perturbadas
        for (i in seq_len(N)) {
            v <- LR %*% rnorm(2)
            y_pert <- y_obs[k, ] + as.numeric(v)
            hi <- as.numeric(H_mat %*% ens[i, ])
            ens[i, ] <- ens[i, ] + as.numeric(K_ens %*% (y_pert - hi))
        }
        x_filt[k, ] <- colMeans(ens)
    }
    x_filt
}

# ---------------------------------------------------------------------------
# Rodar para varios N e computar RMSE
# ---------------------------------------------------------------------------
N_grid <- c(10, 20, 50, 100, 200, 500)
cat("\n--- EnKF por tamanho de ensemble ---\n")
enkf_results <- list()
rmse_mat <- matrix(0, length(N_grid), 3)
colnames(rmse_mat) <- c("x", "y", "z")
for (i in seq_along(N_grid)) {
    N <- N_grid[i]
    t0 <- Sys.time()
    states <- run_enkf(N, seed = 100 + N)
    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    rmse <- sqrt(colMeans((states - x_true)^2))
    rmse_mat[i, ] <- rmse
    enkf_results[[as.character(N)]] <- states
    cat(sprintf("  N = %3d  RMSE = (%.4f, %.4f, %.4f)  time = %.2fs\n",
                N, rmse[1], rmse[2], rmse[3], elapsed))
}

# ---------------------------------------------------------------------------
# Exportacao CSV
# ---------------------------------------------------------------------------
# 1. Estados: EKF + EnKF para cada N
states_df <- data.frame(
    t = t_grid,
    x_true = x_true[, 1], y_true = x_true[, 2], z_true = x_true[, 3],
    ekf_x = ekf_states[, 1], ekf_y = ekf_states[, 2], ekf_z = ekf_states[, 3]
)
for (N in N_grid) {
    s <- enkf_results[[as.character(N)]]
    states_df[[sprintf("enkf_N%d_x", N)]] <- s[, 1]
    states_df[[sprintf("enkf_N%d_y", N)]] <- s[, 2]
    states_df[[sprintf("enkf_N%d_z", N)]] <- s[, 3]
}
write.csv(states_df, "results_enkf_states.csv", row.names = FALSE)
cat("\nSaved: results_enkf_states.csv\n")

# 2. RMSE vs N
rmse_df <- data.frame(
    N        = N_grid,
    rmse_x   = rmse_mat[, 1],
    rmse_y   = rmse_mat[, 2],
    rmse_z   = rmse_mat[, 3],
    rmse_total = sqrt(rowMeans(rmse_mat^2))
)
# acrescenta linha com EKF para comparacao
ref_row <- data.frame(
    N        = NA,
    rmse_x   = rmse_ekf[1],
    rmse_y   = rmse_ekf[2],
    rmse_z   = rmse_ekf[3],
    rmse_total = sqrt(mean(rmse_ekf^2))
)
rmse_df$source <- "EnKF"
ref_row$source <- "EKF"
rmse_df <- rbind(rmse_df, ref_row)
write.csv(rmse_df, "results_enkf_rmse_vs_N.csv", row.names = FALSE)
cat("Saved: results_enkf_rmse_vs_N.csv\n")

cat(sprintf("\nNota: EnKF com N grande converge para performance proxima do EKF.\n"))
cat(sprintf("      Para N pequeno (<=20) ha underestimation de covariancia e\n"))
cat(sprintf("      degradacao esperada do RMSE.\n"))
cat("\nEnKF validation complete.\n")
