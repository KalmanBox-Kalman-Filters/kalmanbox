# =============================================================================
# Validacao: Comparacao MLE - CSS vs ML para ARIMA
#
# Equivalencia SSM-ARIMA e metodos de estimacao:
#   - ML (Maximum Likelihood): Usa o filtro de Kalman para calcular a
#     verossimilhanca exata. Equivalente a KFS no KFAS.
#   - CSS (Conditional Sum of Squares): Minimiza a soma de quadrados dos
#     residuos condicionada nos valores iniciais. Nao usa filtro de Kalman.
#   - CSS-ML: Usa CSS para obter valores iniciais, depois refina com ML.
#
# O KFAS sempre calcula a verossimilhanca exata via filtro de Kalman,
# portanto seus resultados devem coincidir com method="ML" do arima().
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
y_airline_log <- log(y_airline)

cat("=============================================================================\n")
cat("  MLE Comparison: CSS vs ML vs KFAS\n")
cat("=============================================================================\n\n")

# =============================================================================
# PARTE 1: ARIMA(0,1,1) - Nile - CSS vs ML vs KFAS
# =============================================================================
# CSS condiciona nos valores iniciais e minimiza sum of squares.
# ML calcula a verossimilhanca exata via filtro de Kalman.
# Para series longas, CSS e ML convergem. Para series curtas,
# podem haver diferencas notaveis nos parametros estimados.
# =============================================================================

cat("=== ARIMA(0,1,1) - Nile ===\n\n")

# stats::arima com method="ML"
fit_nile_ml <- arima(y_nile, order = c(0, 1, 1), method = "ML")
theta_nile_ml <- as.numeric(coef(fit_nile_ml)["ma1"])
sigma2_nile_ml <- fit_nile_ml$sigma2
loglik_nile_ml <- as.numeric(fit_nile_ml$loglik)

cat("--- arima(method='ML') ---\n")
cat(sprintf("theta (MA1) = %.6f\n", theta_nile_ml))
cat(sprintf("sigma2      = %.4f\n", sigma2_nile_ml))
cat(sprintf("logLik      = %.4f\n", loglik_nile_ml))

# stats::arima com method="CSS"
fit_nile_css <- arima(y_nile, order = c(0, 1, 1), method = "CSS")
theta_nile_css <- as.numeric(coef(fit_nile_css)["ma1"])
sigma2_nile_css <- fit_nile_css$sigma2
loglik_nile_css <- as.numeric(fit_nile_css$loglik)

cat("\n--- arima(method='CSS') ---\n")
cat(sprintf("theta (MA1) = %.6f\n", theta_nile_css))
cat(sprintf("sigma2      = %.4f\n", sigma2_nile_css))
cat(sprintf("logLik (CSS)= %.4f (nao comparavel diretamente com ML)\n", loglik_nile_css))

# stats::arima com method="CSS-ML"
fit_nile_cssml <- arima(y_nile, order = c(0, 1, 1), method = "CSS-ML")
theta_nile_cssml <- as.numeric(coef(fit_nile_cssml)["ma1"])
sigma2_nile_cssml <- fit_nile_cssml$sigma2
loglik_nile_cssml <- as.numeric(fit_nile_cssml$loglik)

cat("\n--- arima(method='CSS-ML') ---\n")
cat(sprintf("theta (MA1) = %.6f\n", theta_nile_cssml))
cat(sprintf("sigma2      = %.4f\n", sigma2_nile_cssml))
cat(sprintf("logLik      = %.4f\n", loglik_nile_cssml))

# KFAS - verossimilhanca exata via KFS
# Usar parametros do ML para construir o SSM
model_nile_kfas <- SSModel(y_nile ~
    SSMarima(d = 1, ma = theta_nile_ml, Q = sigma2_nile_ml), H = 0)
loglik_nile_kfas <- as.numeric(logLik(model_nile_kfas))

cat("\n--- KFAS (exact likelihood via KFS, params from ML) ---\n")
cat(sprintf("logLik      = %.4f\n", loglik_nile_kfas))

cat(sprintf("\nComparacoes Nile:\n"))
cat(sprintf("  ML vs CSS     (theta): %.6f\n", theta_nile_ml - theta_nile_css))
cat(sprintf("  ML vs CSS-ML  (theta): %.6f\n", theta_nile_ml - theta_nile_cssml))
cat(sprintf("  ML vs KFAS   (logLik): %.6f\n", loglik_nile_ml - loglik_nile_kfas))

# =============================================================================
# PARTE 2: Airline ARIMA(0,1,1)(0,1,1)_12 - CSS vs ML vs KFAS
# =============================================================================
# Para o airline model, a diferenca entre CSS e ML pode ser mais
# pronunciada por causa da diferenciacao sazonal que consome 12
# observacoes iniciais no CSS.
# =============================================================================

cat("\n\n=== Airline ARIMA(0,1,1)(0,1,1)_12 ===\n\n")

# ML
fit_air_ml <- arima(y_airline_log,
                    order = c(0, 1, 1),
                    seasonal = list(order = c(0, 1, 1), period = 12),
                    method = "ML")
theta1_air_ml <- as.numeric(coef(fit_air_ml)["ma1"])
Theta1_air_ml <- as.numeric(coef(fit_air_ml)["sma1"])
sigma2_air_ml <- fit_air_ml$sigma2
loglik_air_ml <- as.numeric(fit_air_ml$loglik)

cat("--- arima(method='ML') ---\n")
cat(sprintf("theta1 (MA1)  = %.6f\n", theta1_air_ml))
cat(sprintf("Theta1 (SMA1) = %.6f\n", Theta1_air_ml))
cat(sprintf("sigma2        = %.8f\n", sigma2_air_ml))
cat(sprintf("logLik        = %.4f\n", loglik_air_ml))

# CSS
fit_air_css <- arima(y_airline_log,
                     order = c(0, 1, 1),
                     seasonal = list(order = c(0, 1, 1), period = 12),
                     method = "CSS")
theta1_air_css <- as.numeric(coef(fit_air_css)["ma1"])
Theta1_air_css <- as.numeric(coef(fit_air_css)["sma1"])
sigma2_air_css <- fit_air_css$sigma2
loglik_air_css <- as.numeric(fit_air_css$loglik)

cat("\n--- arima(method='CSS') ---\n")
cat(sprintf("theta1 (MA1)  = %.6f\n", theta1_air_css))
cat(sprintf("Theta1 (SMA1) = %.6f\n", Theta1_air_css))
cat(sprintf("sigma2        = %.8f\n", sigma2_air_css))
cat(sprintf("logLik (CSS)  = %.4f (nao comparavel diretamente com ML)\n", loglik_air_css))

# CSS-ML
fit_air_cssml <- arima(y_airline_log,
                       order = c(0, 1, 1),
                       seasonal = list(order = c(0, 1, 1), period = 12),
                       method = "CSS-ML")
theta1_air_cssml <- as.numeric(coef(fit_air_cssml)["ma1"])
Theta1_air_cssml <- as.numeric(coef(fit_air_cssml)["sma1"])
sigma2_air_cssml <- fit_air_cssml$sigma2
loglik_air_cssml <- as.numeric(fit_air_cssml$loglik)

cat("\n--- arima(method='CSS-ML') ---\n")
cat(sprintf("theta1 (MA1)  = %.6f\n", theta1_air_cssml))
cat(sprintf("Theta1 (SMA1) = %.6f\n", Theta1_air_cssml))
cat(sprintf("sigma2        = %.8f\n", sigma2_air_cssml))
cat(sprintf("logLik        = %.4f\n", loglik_air_cssml))

# KFAS - construir SSM com parametros do ML
# Diferenciar sazonalmente e usar SSMarima
ma_expanded <- rep(0, 13)
ma_expanded[1] <- theta1_air_ml
ma_expanded[12] <- Theta1_air_ml
ma_expanded[13] <- theta1_air_ml * Theta1_air_ml

y_sdiff <- diff(y_airline_log, lag = 12)
model_air_kfas <- SSModel(y_sdiff ~
    SSMarima(d = 1, ma = ma_expanded, Q = sigma2_air_ml), H = 0)
loglik_air_kfas <- as.numeric(logLik(model_air_kfas))

cat("\n--- KFAS (exact likelihood via KFS, params from ML) ---\n")
cat(sprintf("logLik        = %.4f\n", loglik_air_kfas))

cat(sprintf("\nComparacoes Airline:\n"))
cat(sprintf("  ML vs CSS    (theta1): %.6f\n", theta1_air_ml - theta1_air_css))
cat(sprintf("  ML vs CSS    (Theta1): %.6f\n", Theta1_air_ml - Theta1_air_css))
cat(sprintf("  ML vs CSS-ML (theta1): %.6f\n", theta1_air_ml - theta1_air_cssml))

# =============================================================================
# Exportar tabela comparativa
# =============================================================================

# Tabela Nile
comparison_nile <- data.frame(
    model = rep("ARIMA(0,1,1)_Nile", 4),
    method = c("ML", "CSS", "CSS-ML", "KFAS"),
    theta_ma1 = c(theta_nile_ml, theta_nile_css, theta_nile_cssml, NA),
    sigma2 = c(sigma2_nile_ml, sigma2_nile_css, sigma2_nile_cssml, NA),
    loglik = c(loglik_nile_ml, loglik_nile_css, loglik_nile_cssml, loglik_nile_kfas),
    note = c("exact_ML", "conditional_SS", "CSS_then_ML", "exact_via_KFS")
)

# Tabela Airline
comparison_airline <- data.frame(
    model = rep("ARIMA(0,1,1)(0,1,1)_12_Airline", 4),
    method = c("ML", "CSS", "CSS-ML", "KFAS"),
    theta1_ma1 = c(theta1_air_ml, theta1_air_css, theta1_air_cssml, NA),
    Theta1_sma1 = c(Theta1_air_ml, Theta1_air_css, Theta1_air_cssml, NA),
    sigma2 = c(sigma2_air_ml, sigma2_air_css, sigma2_air_cssml, NA),
    loglik = c(loglik_air_ml, loglik_air_css, loglik_air_cssml, loglik_air_kfas),
    note = c("exact_ML", "conditional_SS", "CSS_then_ML", "exact_via_KFS")
)

write.csv(comparison_nile, "results_comparison_mle_nile.csv", row.names = FALSE)
cat("\nSaved: results_comparison_mle_nile.csv\n")

write.csv(comparison_airline, "results_comparison_mle_airline.csv", row.names = FALSE)
cat("Saved: results_comparison_mle_airline.csv\n")

# Tabela consolidada de log-likelihoods
comparison_all <- rbind(
    data.frame(model = comparison_nile$model,
               method = comparison_nile$method,
               loglik = comparison_nile$loglik,
               note = comparison_nile$note),
    data.frame(model = comparison_airline$model,
               method = comparison_airline$method,
               loglik = comparison_airline$loglik,
               note = comparison_airline$note)
)
write.csv(comparison_all, "results_comparison_mle_all.csv", row.names = FALSE)
cat("Saved: results_comparison_mle_all.csv\n")

cat("\nMLE comparison validation complete.\n")
