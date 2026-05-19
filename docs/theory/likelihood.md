# Likelihood

This page collects the likelihood expressions that underpin MLE, EM and
Bayesian inference in `kalmanbox`.

## Prediction-error decomposition

Let $\theta$ be the parameter vector. By the chain rule

$$
p(y_{1:n} \mid \theta) = \prod_{t=1}^{n} p(y_t \mid y_{1:t-1}, \theta),
$$

and each conditional is Gaussian with mean $Z_t a_{t|t-1}$ and
covariance $F_t = Z_t P_{t|t-1} Z_t' + H_t$. Hence

$$
\boxed{\;
\log p(y_{1:n} \mid \theta) =
-\tfrac{1}{2}\sum_{t=1}^{n}
\big[ p\log 2\pi + \log\det F_t + v_t' F_t^{-1} v_t \big].
\;}
$$

This is the **log-likelihood as a by-product of the Kalman filter** —
no extra computation needed.

## Diffuse log-likelihood

When some state components are diffuse (rank $d$ of $P_\infty$), the
limiting log-likelihood is

$$
\log p(y \mid \theta) =
-\tfrac{1}{2}\sum_{t=1}^{d}
\big[\log\det F_t^{(\infty)} + (\text{stationary correction at }t)\big]
\;-\;
\tfrac{1}{2}\sum_{t=d+1}^{n}
\big[ p\log 2\pi + \log\det F_t + v_t' F_t^{-1} v_t \big].
$$

`kalmanbox` accumulates this exactly via the auxiliary diffuse
quantities (see Koopman & Durbin 2003).

## EM auxiliary function

In EM, the M-step maximises
$\mathcal{Q}(\theta \mid \theta^{(s)}) = E_{\alpha \mid y, \theta^{(s)}}
[\log p(\alpha, y \mid \theta)]$. Closed-form updates exist when:

- $Q$, $H$ are unrestricted positive-definite (or diagonal).
- $T$, $Z$ enter linearly.

Let

$$
S_{aa} = \sum_t E[\alpha_t \alpha_t'],\quad
S_{aa^-} = \sum_t E[\alpha_t \alpha_{t-1}'],\quad
S_{yy} = \sum_t y_t y_t',\quad
S_{ya} = \sum_t y_t E[\alpha_t]',
$$

(all expectations under the smoothed posterior). The standard updates
are

$$
\hat T = S_{aa^-}\,(S_{a^-a^-})^{-1},\qquad
\hat Z = S_{ya}\,S_{aa}^{-1},\qquad
\hat Q = \tfrac{1}{n}\!\sum_t E[\eta_t \eta_t'],\qquad
\hat H = \tfrac{1}{n}\!\sum_t E[\varepsilon_t \varepsilon_t'].
$$

Detailed derivations: Shumway & Stoffer (2017), Ch. 6.

## Score and information matrix

For MLE standard errors, `kalmanbox` evaluates the **observed**
information matrix by numerical differentiation of the log-likelihood
at $\hat\theta$. Analytic scores are available for select models via
the Koopman (1993) recursion.

## Related

- [User guide: Kalman filter](../user-guide/kalman/kalman-filter.md)
- [API: estimation](../api/estimation.md)
- [Identifiability](identifiability.md)
