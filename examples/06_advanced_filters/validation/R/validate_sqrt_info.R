# =============================================================================
# Validacao: Square-Root e Information filters para KF LINEAR
#
# Modelo linear usado como benchmark (2-D constant velocity, obs. linear
# em posicao). Baseado no dataset target_tracking.csv, substituindo a
# observacao nao-linear range/bearing por observacao linear (x, y):
#
#   state x = [px, vx, py, vy]'
#   F = [[1,dt,0,0],[0,1,0,0],[0,0,1,dt],[0,0,0,1]]
#   H = [[1,0,0,0],[0,0,1,0]]      (posicao apenas)
#   Q = bloco diagonal CWNA, R diagonal
#
# OBJETIVO: mostrar que
#   KF padrao == SQRT filter == Information filter (ate tol ~ 1e-10)
# no caso linear / gaussiano.
#
# Comparacoes:
#   1. FKF::fkf                       (KF padrao em C, referencia)
#   2. SQRT filter manual (triangulacao de Householder-like via QR)
#   3. Information filter manual       (propagacao de (Y, y))
#
# PACOTE: FKF (Luethi, Erb, Otziger). Referencia para KF rapido em R.
# =============================================================================

suppressPackageStartupMessages({
    library(FKF)
})

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
df <- read.csv("../../data/target_tracking.csv")
t_grid <- df$t
x_true <- cbind(df$x_x, df$x_vx, df$x_y, df$x_vy)
# Observacao LINEAR sintetica: usamos a posicao verdadeira + ruido, para
# que o cenario seja estritamente Gaussiano linear e as tres formulacoes
# devam coincidir. Isto NAO usa range/bearing.
set.seed(42)
sigma_pos <- 1.0
R_obs <- diag(c(sigma_pos^2, sigma_pos^2))
y_obs <- cbind(x_true[, 1], x_true[, 3]) +
    matrix(rnorm(2 * nrow(df), sd = sigma_pos), ncol = 2)
n  <- nrow(df)
dt <- t_grid[2] - t_grid[1]

# Sistema (mesmo do gerador)
sigma_proc <- 0.1
q <- sigma_proc^2
F_mat <- matrix(c(1, dt, 0, 0,
                  0,  1, 0, 0,
                  0,  0, 1, dt,
                  0,  0, 0, 1), nrow = 4, byrow = TRUE)
Q_mat <- q * matrix(c(dt^3/3, dt^2/2, 0, 0,
                      dt^2/2, dt,     0, 0,
                      0, 0, dt^3/3, dt^2/2,
                      0, 0, dt^2/2, dt),
                    nrow = 4, byrow = TRUE)
H_mat <- matrix(c(1, 0, 0, 0,
                  0, 0, 1, 0), nrow = 2, byrow = TRUE)
a0 <- c(0, 1, 100, 0.5)
P0 <- diag(c(10, 1, 10, 1))

cat("=============================================================================\n")
cat("  SQRT / Information filters - Linear KF (R)\n")
cat("=============================================================================\n\n")
cat(sprintf("Sample size n = %d, dt = %.3f\n", n, dt))

# ---------------------------------------------------------------------------
# 1) FKF (referencia)
# ---------------------------------------------------------------------------
cat("\n--- FKF (referencia) ---\n")
# FKF usa t(y) (d_y x n); parametros podem ser time-invariant (slice 1).
yt <- t(y_obs)                   # (2, n)
dt_ <- array(0, c(4, 1))
ct <- array(0, c(2, 1))
Tt <- array(F_mat, dim = c(4, 4, 1))
Zt <- array(H_mat, dim = c(2, 4, 1))
HHt <- array(Q_mat, dim = c(4, 4, 1))
GGt <- array(R_obs, dim = c(2, 2, 1))

fkf_out <- fkf(a0 = a0, P0 = P0,
               dt = dt_, ct = ct,
               Tt = Tt, Zt = Zt,
               HHt = HHt, GGt = GGt,
               yt = yt)
a_fkf <- t(fkf_out$att)          # (n, 4) filtered state
loglik_fkf <- fkf_out$logLik
cat(sprintf("logLik (FKF): %.6f\n", loglik_fkf))

# ---------------------------------------------------------------------------
# 2) KF padrao manual (para termos acesso a P filtered)
# ---------------------------------------------------------------------------
kf_standard <- function() {
    a_filt <- matrix(0, n, 4)
    P_filt <- array(0, c(n, 4, 4))
    a <- a0
    P <- P0
    ll <- 0
    for (k in seq_len(n)) {
        if (k > 1) {
            a <- F_mat %*% a
            P <- F_mat %*% P %*% t(F_mat) + Q_mat
        }
        v <- y_obs[k, ] - as.numeric(H_mat %*% a)
        S <- H_mat %*% P %*% t(H_mat) + R_obs
        K <- P %*% t(H_mat) %*% solve(S)
        a <- a + K %*% v
        P <- (diag(4) - K %*% H_mat) %*% P
        a_filt[k, ] <- as.numeric(a)
        P_filt[k, , ] <- P
        ll <- ll - 0.5 * (length(v) * log(2 * pi) +
                          log(det(S)) +
                          t(v) %*% solve(S, v))
    }
    list(a = a_filt, P = P_filt, loglik = as.numeric(ll))
}

kf <- kf_standard()

# ---------------------------------------------------------------------------
# 3) Square-Root filter (Cholesky, triangulacao via QR)
#
#    Carry S tal que P = S %*% t(S). Updates conservam a triangularidade
#    construindo matrizes empilhadas e aplicando decomposicao QR.
# ---------------------------------------------------------------------------
sqrt_filter <- function() {
    a_filt <- matrix(0, n, 4)
    P_filt <- array(0, c(n, 4, 4))
    a <- a0
    S <- t(chol(P0))              # lower-triangular tal que P = S %*% t(S)
    SQ <- t(chol(Q_mat))
    SR <- t(chol(R_obs))
    ll <- 0
    I4 <- diag(4)
    for (k in seq_len(n)) {
        if (k > 1) {
            a <- F_mat %*% a
            # Predicao: [S, SQ]' is (n_state + n_state) x n_state before F;
            # precompose with F via Householder/QR triangulation.
            M <- rbind(t(F_mat %*% S), t(SQ))
            Qr <- qr(M)
            R_ <- qr.R(Qr)[seq_len(4), seq_len(4), drop = FALSE]
            S <- t(R_)
            # Enforce lower-triangular convention with positive diagonal
            signs <- sign(diag(S)); signs[signs == 0] <- 1
            S <- S * rep(signs, each = 4)
        }
        # Update: forma matriz de bloco
        # [SR,      0      ]
        # [H%*%S,   S       ]
        M <- rbind(
            cbind(t(SR), matrix(0, 2, 4)),
            cbind(t(H_mat %*% S), t(S))
        )
        Qr <- qr(M)
        R_ <- qr.R(Qr)
        # Apos QR: bloco superior 2x2 = sqrt(S_inn); bloco 2x4 a direita
        # e -t(K) %*% sqrt(S_inn) (segundo parametrizacao); bloco inferior
        # 4x4 = sqrt(P_post)
        S_inn_T <- R_[1:2, 1:2, drop = FALSE]
        KT      <- R_[1:2, 3:6, drop = FALSE]
        Spost_T <- R_[3:6, 3:6, drop = FALSE]
        # ajustar sinais para diagonais positivas
        sig <- sign(diag(S_inn_T)); sig[sig == 0] <- 1
        S_inn_T <- S_inn_T * sig
        KT      <- KT * sig
        S_inn <- t(S_inn_T)
        K <- t(KT) %*% solve(S_inn)
        v <- y_obs[k, ] - as.numeric(H_mat %*% a)
        a <- a + K %*% v
        S <- t(Spost_T)
        sig <- sign(diag(S)); sig[sig == 0] <- 1
        S <- S * rep(sig, each = 4)

        P <- S %*% t(S)
        a_filt[k, ] <- as.numeric(a)
        P_filt[k, , ] <- P

        S_mat <- S_inn %*% t(S_inn)
        ll <- ll - 0.5 * (length(v) * log(2 * pi) +
                          log(det(S_mat)) +
                          t(v) %*% solve(S_mat, v))
    }
    list(a = a_filt, P = P_filt, loglik = as.numeric(ll))
}

cat("\n--- SQRT filter manual ---\n")
sq <- sqrt_filter()
cat(sprintf("logLik (SQRT): %.6f\n", sq$loglik))

# ---------------------------------------------------------------------------
# 4) Information filter (forma dual do KF linear)
# ---------------------------------------------------------------------------
info_filter <- function() {
    a_filt <- matrix(0, n, 4)
    P_filt <- array(0, c(n, 4, 4))
    a <- a0
    P <- P0
    Yk <- solve(P)
    yk <- Yk %*% a
    ll <- 0
    R_inv <- solve(R_obs)
    for (k in seq_len(n)) {
        if (k > 1) {
            # Predicao em forma de momento (eficiente para este tamanho)
            P <- F_mat %*% solve(Yk) %*% t(F_mat) + Q_mat
            a <- F_mat %*% solve(Yk, yk)
            Yk <- solve(P)
            yk <- Yk %*% a
        }
        # Update da forma de informacao
        Y_post <- Yk + t(H_mat) %*% R_inv %*% H_mat
        y_post <- yk + t(H_mat) %*% R_inv %*% y_obs[k, ]
        Yk <- Y_post
        yk <- y_post
        P_post <- solve(Yk)
        a_post <- P_post %*% yk
        a <- a_post
        P <- P_post
        a_filt[k, ] <- as.numeric(a_post)
        P_filt[k, , ] <- P_post

        # Loglik calculado via forma momento para precisao numerica
        # usando predicao antes do update.
        v <- y_obs[k, ] - as.numeric(H_mat %*% (if (k == 1) a0 else
                                                 F_mat %*% a_filt[k - 1, ]))
        if (k == 1) {
            P_pred <- P0
        } else {
            P_pred <- F_mat %*% P_filt[k - 1, , ] %*% t(F_mat) + Q_mat
        }
        S <- H_mat %*% P_pred %*% t(H_mat) + R_obs
        ll <- ll - 0.5 * (length(v) * log(2 * pi) +
                          log(det(S)) +
                          t(v) %*% solve(S, v))
    }
    list(a = a_filt, P = P_filt, loglik = as.numeric(ll))
}

cat("\n--- Information filter manual ---\n")
inf <- info_filter()
cat(sprintf("logLik (Info): %.6f\n", inf$loglik))

# ---------------------------------------------------------------------------
# Comparacao numerica
# ---------------------------------------------------------------------------
cat("\n--- Comparacao de equivalencia ---\n")
diff_fkf_kf  <- max(abs(a_fkf - kf$a))
diff_sq_kf   <- max(abs(sq$a  - kf$a))
diff_inf_kf  <- max(abs(inf$a - kf$a))
diff_ll_fkf  <- abs(loglik_fkf - kf$loglik)
diff_ll_sq   <- abs(sq$loglik  - kf$loglik)
diff_ll_inf  <- abs(inf$loglik - kf$loglik)
cat(sprintf("max |a_FKF - a_KF| = %.3e\n", diff_fkf_kf))
cat(sprintf("max |a_SQRT - a_KF|= %.3e\n", diff_sq_kf))
cat(sprintf("max |a_INF - a_KF| = %.3e\n", diff_inf_kf))
cat(sprintf("|logLik FKF - KF| = %.3e\n", diff_ll_fkf))
cat(sprintf("|logLik SQRT - KF|= %.3e\n", diff_ll_sq))
cat(sprintf("|logLik INF  - KF|= %.3e\n", diff_ll_inf))

tol <- 1e-6
ok <- max(diff_fkf_kf, diff_sq_kf, diff_inf_kf) < tol &&
      max(diff_ll_fkf, diff_ll_sq, diff_ll_inf) < tol
cat(sprintf("\nEquivalencia numerica (tol=%g): %s\n", tol,
            ifelse(ok, "OK", "DIVERGENCIA - inspecionar")))

# ---------------------------------------------------------------------------
# Exportacao CSV
# ---------------------------------------------------------------------------
states_df <- data.frame(
    t            = t_grid,
    fkf_px  = a_fkf[, 1], fkf_vx  = a_fkf[, 2],
    fkf_py  = a_fkf[, 3], fkf_vy  = a_fkf[, 4],
    kf_px   = kf$a[, 1],  kf_vx   = kf$a[, 2],
    kf_py   = kf$a[, 3],  kf_vy   = kf$a[, 4],
    sqrt_px = sq$a[, 1],  sqrt_vx = sq$a[, 2],
    sqrt_py = sq$a[, 3],  sqrt_vy = sq$a[, 4],
    info_px = inf$a[, 1], info_vx = inf$a[, 2],
    info_py = inf$a[, 3], info_vy = inf$a[, 4]
)
write.csv(states_df, "results_sqrt_info_states.csv", row.names = FALSE)
cat("\nSaved: results_sqrt_info_states.csv\n")

summary_df <- data.frame(
    metric = c("loglik_fkf", "loglik_kf", "loglik_sqrt", "loglik_info",
               "max_diff_fkf_kf", "max_diff_sqrt_kf", "max_diff_info_kf",
               "tol", "equivalence_ok"),
    value = c(loglik_fkf, kf$loglik, sq$loglik, inf$loglik,
              diff_fkf_kf, diff_sq_kf, diff_inf_kf,
              tol, as.numeric(ok))
)
write.csv(summary_df, "results_sqrt_info_summary.csv", row.names = FALSE)
cat("Saved: results_sqrt_info_summary.csv\n")

cat("\nSQRT/Information validation complete.\n")
