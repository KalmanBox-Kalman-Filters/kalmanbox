# =============================================================================
# Validacao: ARIMA State-Space via KFAS
#
# Equivalencia SSM-ARIMA:
#   Um ARIMA(p,d,q) pode ser representado como modelo de espaco de estados.
#   Para ARIMA(0,1,1): (1-B)y_t = (1 + theta*B)*eps_t
#   Isso e equivalente a um local level model onde:
#     y_t = mu_t + e_t, mu_t = mu_{t-1} + eta_t
#   A relacao entre theta (MA coef) e as variancias SSM e:
#     Cov(Dy_t, Dy_{t-1}) = sigma2_a * theta = -sigma2_eps
#     Var(Dy_t) = sigma2_a * (1 + theta^2) = 2*sigma2_eps + sigma2_eta
#   Para o Nile, theta ~ -0.73 (R convention: 1 + theta*B).
#
#   Nota: Algumas referencias citam theta ~ -0.37. Isso pode decorrer de
#   convencao diferente de sinal ou parametrizacao. Com R arima() e a
#   convencao (1 + theta*B), o resultado padrao e theta ~ -0.73.
#
# Modelos:
#   1. ARIMA(0,1,1) no Nile
#   2. Airline model ARIMA(0,1,1)(0,1,1)_12 no AirPassengers
#
# Abordagem: Estimar via arima(), depois construir SSM em KFAS com
# parametros estimados e verificar que a log-verossimilhanca coincide.
# Isso valida a equivalencia da representacao state-space.
#
# Pacotes: KFAS
# =============================================================================

library(KFAS)

# ---- Dados ----
nile_data <- read.csv("../../data/nile.csv")
y_nile <- ts(nile_data$flow, start = nile_data$year[1], frequency = 1)

airline_data <- read.csv("../../data/airline.csv")
airline_dates <- as.Date(airline_data$date)
airline_start_year <- as.integer(format(airline_dates[1], "%Y"))
airline_start_month <- as.integer(format(airline_dates[1], "%m"))
y_airline <- ts(airline_data$passengers,
                start = c(airline_start_year, airline_start_month),
                frequency = 12)
# Log-transform for airline model (multiplicative -> additive)
y_airline_log <- log(y_airline)

cat("=============================================================================\n")
cat("  ARIMA State-Space Validation via KFAS\n")
cat("=============================================================================\n\n")

# =============================================================================
# PARTE 1: ARIMA(0,1,1) no Nile
# =============================================================================
# Na representacao SSM de um ARIMA(0,1,1), temos:
#   Estado: alpha_t = [mu_t, theta*a_t]'  (companion form)
#   Transicao: alpha_{t+1} = T * alpha_t + R * eta_t
#   Observacao: y_t = Z * alpha_t
#
# Com T = [[1, 1], [0, 0]], Z = [1, 1], R = [1, theta]', Q = sigma2_a
# Equivale a: SSMarima(d=1, ma=theta, Q=sigma2_a), H=0
# =============================================================================

cat("=== PARTE 1: ARIMA(0,1,1) - Nile ===\n")
cat(sprintf("Observations: %d\n", length(y_nile)))
cat(sprintf("Period: %d - %d\n", start(y_nile)[1], end(y_nile)[1]))

# --- stats::arima (referencia) ---
cat("\n--- stats::arima (ML) ---\n")
fit_arima_nile <- arima(y_nile, order = c(0, 1, 1), method = "ML")

theta_nile <- as.numeric(coef(fit_arima_nile)["ma1"])
sigma2_nile <- fit_arima_nile$sigma2
loglik_arima_nile <- as.numeric(fit_arima_nile$loglik)

cat(sprintf("theta (MA1) = %.6f\n", theta_nile))
cat(sprintf("sigma2      = %.4f\n", sigma2_nile))
cat(sprintf("logLik      = %.4f\n", loglik_arima_nile))

# Verificar theta (R convention: 1+theta*B)
# Durbin & Koopman: theta ~ -0.73 com a convencao R
# Nota: a spec menciona theta ~ -0.37, que pode referir-se ao
# signal-to-noise ratio q = sigma2_eta/sigma2_eps ~ 0.097
# ou a uma convencao diferente.
cat(sprintf("\ntheta (MA1) = %.4f\n", theta_nile))
cat("  (Nota: Com convencao R (1+theta*B), theta ~ -0.73 para Nile)\n")
cat("  (Equivalente local level: q = sigma2_eta/sigma2_eps ~ 0.097)\n")

# --- KFAS: SSMarima com parametros estimados ---
# Plugar os parametros do arima() no SSM para validar equivalencia
cat("\n--- KFAS (SSMarima com params do arima) ---\n")
model_nile_kfas <- SSModel(y_nile ~
    SSMarima(d = 1, ma = theta_nile, Q = sigma2_nile), H = 0)
loglik_kfas_nile <- as.numeric(logLik(model_nile_kfas))
cat(sprintf("logLik (KFAS) = %.4f\n", loglik_kfas_nile))
cat(sprintf("Diff logLik (arima - KFAS): %.6f\n",
            loglik_arima_nile - loglik_kfas_nile))

# --- KFAS: fitSSM com updatefn para estimar MA e Q ---
cat("\n--- KFAS (fitSSM com updatefn) ---\n")
update_arima011 <- function(pars, model) {
    theta <- pars[1]
    Q <- exp(pars[2])
    model$T[2, 2, 1] <- 0
    model$R[1, 1, 1] <- 1
    model$R[2, 1, 1] <- theta
    model$Q[1, 1, 1] <- Q
    model
}

model_nile_init <- SSModel(y_nile ~
    SSMarima(d = 1, ma = -0.5, Q = var(diff(y_nile))), H = 0)

fit_nile_kfas <- fitSSM(model_nile_init,
                         inits = c(-0.5, log(var(diff(y_nile)))),
                         updatefn = update_arima011,
                         method = "BFGS")

if (fit_nile_kfas$optim.out$convergence == 0) {
    theta_kfas_nile <- fit_nile_kfas$model$R[2, 1, 1]
    Q_kfas_nile <- as.numeric(fit_nile_kfas$model$Q[1, 1, 1])
    loglik_fit_kfas_nile <- as.numeric(logLik(fit_nile_kfas$model))

    cat(sprintf("theta (MA1) = %.6f\n", theta_kfas_nile))
    cat(sprintf("Q (innov)   = %.4f\n", Q_kfas_nile))
    cat(sprintf("logLik      = %.4f\n", loglik_fit_kfas_nile))
    cat(sprintf("Diff logLik (arima - KFAS fit): %.6f\n",
                loglik_arima_nile - loglik_fit_kfas_nile))
} else {
    cat("  [WARN] KFAS fitSSM did not converge\n")
    theta_kfas_nile <- NA
    Q_kfas_nile <- NA
    loglik_fit_kfas_nile <- NA
}

# =============================================================================
# PARTE 2: Airline Model ARIMA(0,1,1)(0,1,1)_12
# =============================================================================
# O airline model de Box-Jenkins e um ARIMA(0,1,1)(0,1,1)_12
# aplicado ao log dos passageiros:
#   (1-B)(1-B^12) log(y_t) = (1 + theta1*B)(1 + Theta1*B^12)*eps_t
#
# SSMarima do KFAS nao suporta ARIMA sazonal diretamente.
# Abordagem: Construir SSM manualmente com as matrizes do ARIMA sazonal.
#
# A expansao do MA polynomial:
#   (1 + theta1*B)(1 + Theta1*B^12) = 1 + theta1*B + Theta1*B^12 + theta1*Theta1*B^13
# O operador de diferenciacao (1-B)(1-B^12) tem ordem total d=1, D=1.
# =============================================================================

cat("\n\n=== PARTE 2: Airline Model ARIMA(0,1,1)(0,1,1)_12 ===\n")
cat(sprintf("Observations: %d\n", length(y_airline_log)))
cat(sprintf("Period: %d/%d - %d/%d\n",
            start(y_airline_log)[1], start(y_airline_log)[2],
            end(y_airline_log)[1], end(y_airline_log)[2]))

# --- stats::arima (referencia) ---
cat("\n--- stats::arima (ML) ---\n")
fit_airline <- arima(y_airline_log,
                     order = c(0, 1, 1),
                     seasonal = list(order = c(0, 1, 1), period = 12),
                     method = "ML")

theta1_airline <- as.numeric(coef(fit_airline)["ma1"])
Theta1_airline <- as.numeric(coef(fit_airline)["sma1"])
sigma2_airline <- fit_airline$sigma2
loglik_arima_airline <- as.numeric(fit_airline$loglik)

cat(sprintf("theta1 (MA1)  = %.6f\n", theta1_airline))
cat(sprintf("Theta1 (SMA1) = %.6f\n", Theta1_airline))
cat(sprintf("sigma2        = %.8f\n", sigma2_airline))
cat(sprintf("logLik        = %.4f\n", loglik_arima_airline))

# --- KFAS: Construcao manual do SSM para ARIMA sazonal ---
# Para ARIMA(0,1,1)(0,1,1)_12, aplicamos diferencas regular e sazonal
# na serie, obtendo w_t = (1-B)(1-B^12)y_t, e modelamos w_t como MA(13)
# com restricoes nos coeficientes.
# Alternativamente, usamos o resultado do arima() e construimos o SSM
# para verificar a log-likelihood.
cat("\n--- KFAS (SSM manual com params do arima) ---\n")

# Abordagem: diferenciar sazonalmente e usar SSMarima(d=1, ma=...)
# A serie duplamente diferenciada segue MA com:
# coefs nas posicoes 1, 12, 13: theta1, Theta1, theta1*Theta1
n_airline <- length(y_airline_log)

# Construir vetor MA expandido: posicoes 1..13
# MA poly = (1 + theta1*B)(1 + Theta1*B^12)
# = 1 + theta1*B + Theta1*B^12 + theta1*Theta1*B^13
ma_expanded <- rep(0, 13)
ma_expanded[1] <- theta1_airline
ma_expanded[12] <- Theta1_airline
ma_expanded[13] <- theta1_airline * Theta1_airline

# Diferenciar sazonalmente primeiro
y_sdiff <- diff(y_airline_log, lag = 12)

# Agora aplicar SSMarima com d=1 e os coeficientes MA expandidos
model_airline_kfas <- SSModel(y_sdiff ~
    SSMarima(d = 1, ma = ma_expanded, Q = sigma2_airline), H = 0)
loglik_kfas_airline <- as.numeric(logLik(model_airline_kfas))

cat(sprintf("logLik (KFAS SSM) = %.4f\n", loglik_kfas_airline))
cat(sprintf("logLik (arima)    = %.4f\n", loglik_arima_airline))
cat(sprintf("Diff logLik:        %.6f\n",
            loglik_arima_airline - loglik_kfas_airline))
cat("  (Nota: diferenca pode existir por tratamento diferente dos\n")
cat("   valores iniciais na diferenciacao sazonal)\n")

# =============================================================================
# Exportar resultados
# =============================================================================

# Tabela comparativa Nile ARIMA(0,1,1)
results_nile <- data.frame(
    source = c("arima_ML", "KFAS_SSMarima_plugin", "KFAS_fitSSM"),
    theta_ma1 = c(theta_nile, theta_nile, theta_kfas_nile),
    sigma2_or_Q = c(sigma2_nile, sigma2_nile, Q_kfas_nile),
    loglik = c(loglik_arima_nile, loglik_kfas_nile, loglik_fit_kfas_nile)
)
write.csv(results_nile, "results_arima_nile.csv", row.names = FALSE)
cat("\nSaved: results_arima_nile.csv\n")

# Tabela comparativa airline model
results_airline <- data.frame(
    source = c("arima_ML", "KFAS_SSM_manual"),
    theta1_ma1 = c(theta1_airline, theta1_airline),
    Theta1_sma1 = c(Theta1_airline, Theta1_airline),
    sigma2 = c(sigma2_airline, sigma2_airline),
    loglik = c(loglik_arima_airline, loglik_kfas_airline)
)
write.csv(results_airline, "results_arima_airline.csv", row.names = FALSE)
cat("Saved: results_arima_airline.csv\n")

# Parametros equivalentes do local level model
# Da relacao ARIMA(0,1,1) <-> local level:
#   sigma2_a * theta = -sigma2_eps
#   sigma2_a * (1 + theta^2) = 2*sigma2_eps + sigma2_eta
sigma2_eps_equiv <- -sigma2_nile * theta_nile
sigma2_eta_equiv <- sigma2_nile * (1 + theta_nile^2) - 2 * sigma2_eps_equiv
q_ratio <- sigma2_eta_equiv / sigma2_eps_equiv

cat(sprintf("\nEquivalencia local level para Nile:\n"))
cat(sprintf("  sigma2_eps = %.2f\n", sigma2_eps_equiv))
cat(sprintf("  sigma2_eta = %.2f\n", sigma2_eta_equiv))
cat(sprintf("  q = sigma2_eta/sigma2_eps = %.4f\n", q_ratio))
cat("  (Durbin & Koopman 2012: sigma2_eps=15099, sigma2_eta=1469.1, q=0.097)\n")

cat("\nARIMA SSM validation complete.\n")
