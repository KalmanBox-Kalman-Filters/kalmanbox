# =============================================================================
# Validacao: EKF / UKF para pendulo nao-linear
#
# Modelo (Sarkka 2013, Cap. 5):
#   theta_dot_{k+1} = theta_dot_k - dt * (g/L) * sin(theta_k) + w_k
#   theta_{k+1}     = theta_k + dt * theta_dot_{k+1}
#   y_k             = sin(theta_k) + v_k
#
# Pacotes R para filtros NAO-LINEARES sao escassos. Esta validacao:
#   1. Implementa EKF manualmente em R (loop de predicao/atualizacao)
#      - referencia para o EKF do kalmanbox
#   2. Implementa UKF manualmente (unscented transform) para comparar
#   3. Aplica KFAS ao modelo LINEARIZADO em torno do ponto inicial
#      - referencia limitada, so e valida para oscilacoes pequenas
#
# LIMITACOES DO R PARA FILTROS NAO-LINEARES:
#   - Nao existe pacote CRAN completo para EKF/UKF/EnKF (circa 2024)
#   - KFAS, FKF e dlm sao restritos a modelos Gaussianos lineares
#   - Implementacoes manuais sao pedagogicas mas lentas (sem vetorizacao C)
# =============================================================================

suppressPackageStartupMessages({
    library(KFAS)
})

# ---------------------------------------------------------------------------
# Carregar dados
# ---------------------------------------------------------------------------
df <- read.csv("../../data/pendulum.csv")
t_grid <- df$t
x_true <- cbind(df$x_theta, df$x_theta_dot)   # (n, 2)
y      <- matrix(df$y_sin_theta, ncol = 1)    # (n, 1)
n      <- nrow(df)
dt     <- t_grid[2] - t_grid[1]

# Parametros do gerador (devem bater com generate_nonlinear.py)
g       <- 9.81
length_ <- 1.0
omega2  <- g / length_
sigma_proc <- 0.01
sigma_obs  <- 0.3

Q <- matrix(c(0, 0,
              0, (dt * sigma_proc)^2),
            nrow = 2, byrow = TRUE)
R_obs <- matrix(sigma_obs^2, 1, 1)

cat("=============================================================================\n")
cat("  EKF / UKF - Nonlinear Pendulum Validation (R)\n")
cat("=============================================================================\n\n")
cat(sprintf("Sample size      : n  = %d\n", n))
cat(sprintf("Step size        : dt = %.3f s\n", dt))
cat(sprintf("Process noise    : sigma_proc = %.4f\n", sigma_proc))
cat(sprintf("Observation noise: sigma_obs  = %.4f\n", sigma_obs))

# ---------------------------------------------------------------------------
# Dinamica (semi-implicita, igual ao gerador Python)
# ---------------------------------------------------------------------------
f_dyn <- function(x) {
    theta <- x[1]; td <- x[2]
    td_next    <- td - dt * omega2 * sin(theta)
    theta_next <- theta + dt * td_next
    c(theta_next, td_next)
}
F_jac <- function(x) {
    theta <- x[1]
    # df/dx (semi-implicit): theta' = theta + dt*(td - dt*omega2*sin(theta))
    #                         td'    = td - dt*omega2*sin(theta)
    matrix(c(1 - dt^2 * omega2 * cos(theta), dt,
             -dt * omega2 * cos(theta),      1),
           nrow = 2, byrow = TRUE)
}
h_obs <- function(x) sin(x[1])
H_jac <- function(x) matrix(c(cos(x[1]), 0), nrow = 1)

# ---------------------------------------------------------------------------
# 1) EKF manual
# ---------------------------------------------------------------------------
run_ekf <- function() {
    x_filt <- matrix(0, n, 2)
    P_filt <- array(0, c(n, 2, 2))
    x <- c(1.2, 0.0)
    P <- diag(c(0.1, 0.1))
    loglik <- 0

    for (k in seq_len(n)) {
        if (k > 1) {
            x <- f_dyn(x)
            Fk <- F_jac(x_filt[k - 1, ])
            P  <- Fk %*% P %*% t(Fk) + Q
        }
        Hk <- H_jac(x)
        innov <- y[k, 1] - h_obs(x)
        S  <- Hk %*% P %*% t(Hk) + R_obs
        K  <- P %*% t(Hk) %*% solve(S)
        x  <- x + as.numeric(K * innov)
        P  <- (diag(2) - K %*% Hk) %*% P

        x_filt[k, ] <- x
        P_filt[k, , ] <- P
        loglik <- loglik - 0.5 * (log(2 * pi * as.numeric(S)) +
                                  (innov^2) / as.numeric(S))
    }
    list(x = x_filt, P = P_filt, loglik = loglik)
}

cat("\n--- Rodando EKF manual ---\n")
ekf <- run_ekf()
rmse_ekf <- sqrt(colMeans((ekf$x - x_true)^2))
cat(sprintf("RMSE theta      : %.6f\n", rmse_ekf[1]))
cat(sprintf("RMSE theta_dot  : %.6f\n", rmse_ekf[2]))
cat(sprintf("logLik (EKF)    : %.4f\n", ekf$loglik))

# ---------------------------------------------------------------------------
# 2) UKF manual (unscented transform padrao)
# ---------------------------------------------------------------------------
run_ukf <- function(alpha = 1e-3, beta = 2, kappa = 0) {
    d <- 2
    lambda <- alpha^2 * (d + kappa) - d
    c_sig  <- d + lambda
    Wm <- c(lambda / c_sig, rep(1 / (2 * c_sig), 2 * d))
    Wc <- Wm
    Wc[1] <- Wc[1] + (1 - alpha^2 + beta)

    sigma_points <- function(mu, P) {
        S <- tryCatch(t(chol(c_sig * P)),
                      error = function(e) t(chol(c_sig * (P + 1e-9 * diag(d)))))
        X <- matrix(0, 2 * d + 1, d)
        X[1, ] <- mu
        for (i in seq_len(d)) {
            X[1 + i, ]     <- mu + S[, i]
            X[1 + d + i, ] <- mu - S[, i]
        }
        X
    }

    x_filt <- matrix(0, n, 2)
    P_filt <- array(0, c(n, 2, 2))
    mu <- c(1.2, 0.0)
    P  <- diag(c(0.1, 0.1))
    loglik <- 0

    for (k in seq_len(n)) {
        if (k > 1) {
            X <- sigma_points(mu, P)
            Xp <- t(apply(X, 1, f_dyn))
            mu <- as.numeric(Wm %*% Xp)
            diffs <- sweep(Xp, 2, mu)
            P <- matrix(0, d, d)
            for (i in seq_len(2 * d + 1)) {
                P <- P + Wc[i] * tcrossprod(diffs[i, ])
            }
            P <- P + Q
        }
        X  <- sigma_points(mu, P)
        Yp <- apply(X, 1, h_obs)  # (2d+1,)
        y_mean <- sum(Wm * Yp)
        dy <- Yp - y_mean
        Pyy <- sum(Wc * dy^2) + as.numeric(R_obs)
        dX  <- sweep(X, 2, mu)
        Pxy <- colSums(Wc * dy * dX) # (d,)
        K <- Pxy / Pyy
        innov <- y[k, 1] - y_mean
        mu <- mu + K * innov
        P  <- P - outer(K, K) * Pyy

        x_filt[k, ] <- mu
        P_filt[k, , ] <- P
        loglik <- loglik - 0.5 * (log(2 * pi * Pyy) + innov^2 / Pyy)
    }
    list(x = x_filt, P = P_filt, loglik = loglik)
}

cat("\n--- Rodando UKF manual ---\n")
ukf <- run_ukf()
rmse_ukf <- sqrt(colMeans((ukf$x - x_true)^2))
cat(sprintf("RMSE theta      : %.6f\n", rmse_ukf[1]))
cat(sprintf("RMSE theta_dot  : %.6f\n", rmse_ukf[2]))
cat(sprintf("logLik (UKF)    : %.4f\n", ukf$loglik))

# ---------------------------------------------------------------------------
# 3) KFAS aplicado ao modelo LINEARIZADO em torno de theta = 0
#
#    Expansao: sin(theta) ~ theta,  (g/L) * sin(theta) ~ omega2 * theta
#    Dinamica linear:
#       [theta, td]' = F_lin [theta, td]
#       y = [1, 0] [theta, td] + v
#    Essa aproximacao so e razoavel para |theta| pequeno.
#    Inclusa apenas como REFERENCIA do que KFAS consegue fazer
#    para esse problema (ou seja: muito pouco).
# ---------------------------------------------------------------------------
cat("\n--- KFAS (modelo linearizado) ---\n")
F_lin <- matrix(c(1 - dt^2 * omega2, dt,
                  -dt * omega2,      1), nrow = 2, byrow = TRUE)
Z_lin <- matrix(c(1, 0), nrow = 1)
T_lin <- F_lin
R_mat <- diag(2)
Q_lin <- Q
a1 <- matrix(c(1.2, 0.0), ncol = 1)
P1 <- diag(c(0.1, 0.1))

# KFAS trabalha com y = Z * alpha + eps; usamos y_t = sin(theta) aproximado
# pelo theta linearizado. Ainda e uma aproximacao ruim longe de theta=0.
model_lin <- SSModel(y ~ -1 + SSMcustom(Z = Z_lin, T = T_lin, R = R_mat,
                                        Q = Q_lin, a1 = a1, P1 = P1),
                     H = R_obs)
out_lin <- KFS(model_lin, filtering = "state", smoothing = "state")
state_lin <- as.matrix(out_lin$alphahat)   # (n, 2)
rmse_lin  <- sqrt(colMeans((state_lin - x_true)^2))
cat(sprintf("KFAS linearizado RMSE theta     : %.6f\n", rmse_lin[1]))
cat(sprintf("KFAS linearizado RMSE theta_dot : %.6f\n", rmse_lin[2]))
cat("Nota: RMSE elevado em relacao ao EKF/UKF confirma a necessidade de\n")
cat("      filtros nao-lineares. KFAS e apenas referencia parcial.\n")

# ---------------------------------------------------------------------------
# Exportacao CSV
# ---------------------------------------------------------------------------
states_df <- data.frame(
    t               = t_grid,
    x_theta_true    = x_true[, 1],
    x_theta_dot_true = x_true[, 2],
    ekf_theta       = ekf$x[, 1],
    ekf_theta_dot   = ekf$x[, 2],
    ukf_theta       = ukf$x[, 1],
    ukf_theta_dot   = ukf$x[, 2],
    kfas_lin_theta     = state_lin[, 1],
    kfas_lin_theta_dot = state_lin[, 2]
)
write.csv(states_df, "results_ekf_ukf_states.csv", row.names = FALSE)
cat("\nSaved: results_ekf_ukf_states.csv\n")

summary_df <- data.frame(
    metric = c("loglik_ekf", "loglik_ukf",
               "rmse_ekf_theta", "rmse_ekf_theta_dot",
               "rmse_ukf_theta", "rmse_ukf_theta_dot",
               "rmse_kfas_lin_theta", "rmse_kfas_lin_theta_dot",
               "n_obs", "dt"),
    value = c(ekf$loglik, ukf$loglik,
              rmse_ekf[1], rmse_ekf[2],
              rmse_ukf[1], rmse_ukf[2],
              rmse_lin[1], rmse_lin[2],
              n, dt)
)
write.csv(summary_df, "results_ekf_ukf_summary.csv", row.names = FALSE)
cat("Saved: results_ekf_ukf_summary.csv\n")

cat("\nEKF/UKF validation complete.\n")
