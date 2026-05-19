# References

This page collects the primary literature on which `kalmanbox` is built, organized by topic.
Entries include full bibliographic details, journal volumes, page ranges, ISBNs where applicable,
and notes indicating how each work connects to specific features of the library.

!!! note "Convention alignment"
    Throughout `kalmanbox`, matrix naming follows **Durbin & Koopman (2012)** — the system
    matrices are denoted `Z`, `T`, `R`, `Q`, `H`, `c`, `d`. When reading other references, map
    their notation accordingly (Harvey uses `H`, `F`, `G`; Anderson & Moore use `A`, `C`, `Γ`).

---

## 1. Foundational Books

These are the primary textbooks on which `kalmanbox` theory and implementation are based.
Each section note links the book to specific modules and methods in the library.

### Durbin & Koopman (2012)

> **Durbin, J. & Koopman, S. J. (2012).**
> *Time Series Analysis by State Space Methods*, 2nd edition.
> Oxford Statistical Science Series, Vol. 38.
> Oxford University Press, Oxford.
> ISBN 978-0-19-964117-8.

**Key chapters for `kalmanbox` users:**

| Chapter | Topic | Relevant `kalmanbox` module |
|---------|-------|----------------------------|
| 2 | The Kalman filter | `kalmanbox.filters.KalmanFilter` |
| 3 | Filtering, smoothing and forecasting | `kalmanbox.smoothers.RTSSmoother` |
| 4 | State space models in practice | `kalmanbox.models.*` |
| 5 | Diffuse initialisation | `kalmanbox.filters.DiffuseKalmanFilter` |
| 6 | Further computational aspects | `kalmanbox.filters.SquareRootFilter` |
| 7 | Maximum likelihood estimation | `kalmanbox.estimation.MLE` |
| 8 | Bayesian estimation | `kalmanbox.estimation.EMMixin` |
| 11 | Unobserved components models | `kalmanbox.models.UCM` |
| 12 | Multivariate models | `kalmanbox.models.DFM` |

!!! note "Primary reference"
    This book is the canonical reference for `kalmanbox`. The filter recursions in
    `KalmanFilter.filter()`, the diffuse initialisation in `DiffuseKalmanFilter`, and the
    prediction-error likelihood function all follow Durbin & Koopman (2012) exactly.

---

### Harvey (1989)

> **Harvey, A. C. (1989).**
> *Forecasting, Structural Time Series Models and the Kalman Filter*.
> Cambridge University Press, Cambridge.
> ISBN 978-0-521-32196-2.

Harvey's book is the origin of the structural time series framework that underlies
`kalmanbox.models.BSM`, `LocalLevel`, and `LocalLinearTrend`. The decomposition of a series
into trend, seasonal, cycle, and irregular components, and the concept of signal-to-noise
ratios as key parameters, were introduced and formalised here.

**Key contributions relevant to `kalmanbox`:**

- Definition of the Basic Structural Model (BSM)
- Trigonometric seasonal representation
- Cycle component specification
- Diagnostic checking via auxiliary residuals
- Prediction-error decomposition for likelihood computation (Chapter 3)

---

### Anderson & Moore (1979)

> **Anderson, B. D. O. & Moore, J. B. (1979).**
> *Optimal Filtering*.
> Prentice-Hall, Englewood Cliffs, NJ.
> ISBN 978-0-13-638122-8.
> (Reprinted by Dover Publications, 2005. ISBN 978-0-486-43938-2.)

The classic engineering-oriented treatment of the Kalman filter. Covers the continuous-time
filter, discrete-time filter, duality with LQR control, steady-state behaviour, and algebraic
Riccati equations. The convergence proofs and stability conditions referenced in
`kalmanbox` documentation are drawn from this book.

---

### Shumway & Stoffer (2017)

> **Shumway, R. H. & Stoffer, D. S. (2017).**
> *Time Series Analysis and Its Applications: With R Examples*, 4th edition.
> Springer Texts in Statistics.
> Springer, New York.
> ISBN 978-3-319-52451-1 (hardcover); 978-3-319-52452-8 (eBook).

The EM algorithm derivations for state-space models (the E-step via fixed-interval smoothing,
the M-step closed-form updates) implemented in `kalmanbox.estimation.EMMixin` follow
Shumway & Stoffer (2017, Chapter 6) closely, which itself builds on Shumway & Stoffer (1982).

---

### West & Harrison (1997)

> **West, M. & Harrison, J. (1997).**
> *Bayesian Forecasting and Dynamic Models*, 2nd edition.
> Springer Series in Statistics.
> Springer, New York.
> ISBN 978-0-387-94725-6.

The foundational Bayesian treatment of dynamic linear models (DLMs). Covers conjugate
updating, discount factors, mixture models, and intervention. The Gibbs sampling framework
in `kalmanbox.estimation.GibbsSampler` is informed by the DLM structure developed here, and
the concept of variance discount factors follows West & Harrison's Chapter 6.

---

### Brockwell & Davis (1991)

> **Brockwell, P. J. & Davis, R. A. (1991).**
> *Time Series: Theory and Methods*, 2nd edition.
> Springer Series in Statistics.
> Springer, New York.
> ISBN 978-0-387-97429-3.

Background reference for ARIMA representations and the connection between ARIMA models and
state-space form. The ARIMA-to-state-space conversion implemented in
`kalmanbox.models.ARIMASSM` follows the innovation form derived in Brockwell & Davis (1991,
Chapter 12).

---

### Kim & Nelson (1999)

> **Kim, C.-J. & Nelson, C. R. (1999).**
> *State-Space Models with Regime Switching: Classical and Gibbs-Sampling Approaches with Applications*.
> MIT Press, Cambridge, MA.
> ISBN 978-0-262-11245-2.

The primary reference for Markov-switching state-space models, Kim's collapsing algorithm
for approximate filtering under regime switching, and Gibbs sampling approaches for
regime-switching parameters. Referenced in the advanced topics section of the `kalmanbox`
documentation.

---

### Commandeur & Koopman (2007)

> **Commandeur, J. J. F. & Koopman, S. J. (2007).**
> *An Introduction to State Space Time Series Analysis*.
> Oxford University Press, Oxford.
> ISBN 978-0-19-922887-4.

An accessible, applied introduction to state-space modelling. Recommended as the entry-point
reading for `kalmanbox` users who are new to the topic — particularly before approaching
Durbin & Koopman (2012). Examples use the same structural model conventions as `kalmanbox`.

---

### Petris, Petrone & Campagnoli (2009)

> **Petris, G., Petrone, S. & Campagnoli, P. (2009).**
> *Dynamic Linear Models with R*.
> Use R! Series.
> Springer, New York.
> ISBN 978-0-387-77237-0.

Covers both Bayesian and frequentist approaches to dynamic linear models with worked R
examples. The `dlm` R package accompanies this book. The Bayesian estimation exposition and
the FFBS implementation described here parallel the approach in `kalmanbox.estimation`.

---

## 2. Kalman Filtering — Seminal Papers

### Kalman (1960)

> **Kalman, R. E. (1960).**
> A new approach to linear filtering and prediction problems.
> *Journal of Basic Engineering — Transactions of the ASME, Series D*,
> **82**(1), 35–45.
> doi: 10.1115/1.3662552

The paper that introduced what is now called the Kalman filter. Kalman derived the optimal
linear recursive estimator for discrete-time linear dynamic systems with Gaussian noise by
minimising the mean squared error of state estimates. This recursion, implemented in
`KalmanFilter.filter()`, has been unchanged in its core form since 1960.

!!! note "Historical note"
    Kalman's 1960 paper was submitted to ASME, not to a signal processing journal, because
    Kalman's motivating application was control engineering. The paper is often cited alongside
    Kalman & Bucy (1961) (*Journal of Basic Engineering*, 83, 95–108) which extended the
    results to the continuous-time case.

---

### Rauch, Tung & Striebel (1965)

> **Rauch, H. E., Tung, F. & Striebel, C. T. (1965).**
> Maximum likelihood estimates of linear dynamic systems.
> *AIAA Journal*, **3**(8), 1445–1450.
> doi: 10.2514/3.3166

Introduced the Rauch–Tung–Striebel (RTS) smoother: a two-pass algorithm that runs the
Kalman filter forward and then applies a backward smoothing recursion to compute the posterior
distribution of all states simultaneously given all observations. Implemented directly in
`kalmanbox.smoothers.RTSSmoother`.

---

### Bryson & Ho (1969)

> **Bryson, A. E. & Ho, Y.-C. (1969).**
> *Applied Optimal Control: Optimization, Estimation, and Control*.
> Blaisdell, Waltham, MA.
> (Revised edition: Taylor & Francis, 1975. ISBN 978-0-891-16228-5.)

Early textbook treatment connecting optimal control theory (LQR) to optimal filtering
(Kalman filter) through the principle of duality. Referenced in `kalmanbox` for the
theoretical underpinning of the Riccati recursion and its steady-state properties.

---

### Bucy & Joseph (1968)

> **Bucy, R. S. & Joseph, P. D. (1968).**
> *Filtering for Stochastic Processes with Applications to Guidance*.
> Wiley-Interscience, New York.
> (2nd edition: Chelsea Publishing, 1987. ISBN 978-0-8284-0318-5.)

An early and mathematically rigorous treatment of optimal filtering, covering both continuous
and discrete time. Influential in establishing the theoretical foundations that justify the
Kalman recursion as optimal among all unbiased estimators in the linear Gaussian case.

---

## 3. Diffuse Initialisation

Diffuse initialisation is required when the initial state is non-stationary or otherwise
cannot be assigned a finite prior covariance. `kalmanbox.filters.DiffuseKalmanFilter`
implements the exact diffuse filter following the literature below.

### De Jong (1991)

> **De Jong, P. (1991).**
> The diffuse Kalman filter.
> *The Annals of Statistics*, **19**(2), 1073–1083.
> doi: 10.1214/aos/1176348139

Established a rigorous statistical treatment of diffuse initialisation. De Jong showed that
the likelihood of a state-space model with non-stationary initial state can be computed
exactly by a modified filter recursion that tracks the "diffuse" and "stationary" components
of the covariance separately. This is the paper that mathematically underpins the diffuse
filter in `kalmanbox`.

---

### De Jong (1991) — Biometrika

> **De Jong, P. (1991).**
> Stable algorithms for the state space model.
> *Journal of Time Series Analysis*, **12**(2), 143–157.
> doi: 10.1111/j.1467-9892.1991.tb00075.x

---

### Koopman (1997)

> **Koopman, S. J. (1997).**
> Exact initial Kalman filtering and smoothing for nonstationary time series models.
> *Journal of the American Statistical Association*, **92**(440), 1630–1638.
> doi: 10.1080/01621459.1997.10473685

Extended De Jong's results to cover smoothing under diffuse initialisation, providing the
complete framework for inference in non-stationary models. Key reference for the diffuse
smoother equations in `kalmanbox`.

---

### Koopman & Durbin (2003)

> **Koopman, S. J. & Durbin, J. (2003).**
> Filtering and smoothing of state vector for diffuse state-space models.
> *Journal of Time Series Analysis*, **24**(1), 85–98.
> doi: 10.1111/1467-9892.00294

A unified, compact presentation of the exact diffuse Kalman filter and smoother, with
efficient recursions and derivation of the marginal likelihood. This paper is the single most
directly implemented reference for `DiffuseKalmanFilter` in `kalmanbox`.

---

### Harvey & Phillips (1979)

> **Harvey, A. C. & Phillips, G. D. A. (1979).**
> Maximum likelihood estimation of regression models with autoregressive-moving average
> disturbances.
> *Biometrika*, **66**(1), 49–58.
> doi: 10.1093/biomet/66.1.49

An early treatment of the initialisation problem for models with unit-root components.
Introduced the idea of concentrating the likelihood over diffuse parameters, anticipating the
exact diffuse approaches that followed.

---

### Francke, Koopman & de Vos (2010)

> **Francke, M. K., Koopman, S. J. & de Vos, A. F. (2010).**
> Likelihood functions for state space models with diffuse initial conditions.
> *Journal of Time Series Analysis*, **31**(6), 407–414.
> doi: 10.1111/j.1467-9892.2010.00673.x

Clarified the relationship between different definitions of the diffuse likelihood and showed
when they are numerically equivalent. Relevant to the likelihood computation in
`kalmanbox.estimation.MLE` for models with integrated components.

---

## 4. Maximum Likelihood and EM Algorithm

### Schweppe (1965)

> **Schweppe, F. (1965).**
> Evaluation of likelihood functions for Gaussian signals.
> *IEEE Transactions on Information Theory*, **11**(1), 61–70.
> doi: 10.1109/TIT.1965.1053737

Derived the prediction-error decomposition of the Gaussian likelihood for a state-space
model — expressing the log-likelihood as a sum of squared standardised innovations and their
log-determinants. This is the form computed by `kalmanbox.estimation.MLE.log_likelihood()`.

---

### Dempster, Laird & Rubin (1977)

> **Dempster, A. P., Laird, N. M. & Rubin, D. B. (1977).**
> Maximum likelihood from incomplete data via the EM algorithm.
> *Journal of the Royal Statistical Society, Series B (Methodological)*, **39**(1), 1–38.
> doi: 10.1111/j.2517-6161.1977.tb01600.x

The original EM paper. Introduced the expectation-maximisation algorithm as a general
framework for MLE when data are incomplete or when latent variables are present. In the
state-space context, the latent states are the "missing data", and the E-step corresponds to
running the Kalman smoother.

---

### Shumway & Stoffer (1982)

> **Shumway, R. H. & Stoffer, D. S. (1982).**
> An approach to time series smoothing and forecasting using the EM algorithm.
> *Journal of Time Series Analysis*, **3**(4), 253–264.
> doi: 10.1111/j.1467-9892.1982.tb00349.x

Applied the EM algorithm specifically to state-space models. Showed that the E-step is
exactly the Kalman smoother, and derived closed-form M-step updates for the system matrices
`Q`, `H`, and `T`. This paper is the direct basis for `kalmanbox.estimation.EMMixin`.

---

### Watson & Engle (1983)

> **Watson, M. W. & Engle, R. F. (1983).**
> Alternative algorithms for the estimation of dynamic factor, MIMIC and varying coefficient
> regression models.
> *Journal of Econometrics*, **23**(3), 385–400.
> doi: 10.1016/0304-4076(83)90066-0

Extended the Shumway–Stoffer EM approach to time-varying parameter models and dynamic factor
models. The EM updates for the DFM loading matrix implemented in `kalmanbox.models.DFM` trace
to this paper.

---

### Hamilton (1994)

> **Hamilton, J. D. (1994).**
> *Time Series Analysis*.
> Princeton University Press, Princeton, NJ.
> ISBN 978-0-691-04289-3.

Widely used graduate-level econometrics textbook with thorough coverage of state-space
representations, the Kalman filter (Chapter 13), and the connection to ARIMA and VAR models.
Hamilton's notation is common in applied economics and is cross-referenced in the
`kalmanbox` documentation where it differs from Durbin & Koopman.

---

## 5. Structural Time Series

### Harvey (1985)

> **Harvey, A. C. (1985).**
> Trends and cycles in macroeconomic time series.
> *Journal of Business and Economic Statistics*, **3**(3), 216–227.
> doi: 10.1080/07350015.1985.10509453

Introduced the decomposition of macroeconomic series into stochastic trend, cycle, and
irregular components using state-space methods, demonstrating that many classical business
cycle stylised facts emerge naturally from this framework. This paper is the empirical
motivation for `kalmanbox.models.BSM` and `UCM`.

---

### Harvey & Todd (1983)

> **Harvey, A. C. & Todd, P. H. J. (1983).**
> Forecasting economic time series with structural and Box–Jenkins models: A case study.
> *Journal of Business and Economic Statistics*, **1**(4), 299–307.
> doi: 10.1080/07350015.1983.10509358

Comparative forecasting study showing that structural time series models perform at least as
well as Box–Jenkins ARIMA models, often better out-of-sample, while providing interpretable
components. Referenced in the `kalmanbox` documentation as motivation for the structural
model approach.

---

### Harvey & Jaeger (1993)

> **Harvey, A. C. & Jaeger, A. (1993).**
> Detrending, stylized facts and the business cycle.
> *Journal of Applied Econometrics*, **8**(3), 231–247.
> doi: 10.1002/jae.3950080302

Compared Hodrick–Prescott filtering with structural time series decomposition, arguing that
the BSM-based decomposition avoids the spurious cyclicality introduced by HP filtering. The
cycle component in `kalmanbox.models.UCM` is specified following this paper.

---

### Maravall & Planas (1999)

> **Maravall, A. & Planas, C. (1999).**
> Estimation error and the specification of unobserved component models.
> *Journal of Econometrics*, **92**(2), 325–353.
> doi: 10.1016/S0304-4076(98)00094-4

Analysed identification issues in unobserved components models — specifically when different
parameterisations of the structural model can produce identical reduced-form ARIMA
representations. Relevant to the identifiability warnings issued by `kalmanbox` for
under-determined UCM specifications.

---

## 6. Dynamic Factor Models

### Stock & Watson (2002)

> **Stock, J. H. & Watson, M. W. (2002).**
> Macroeconomic forecasting using diffusion indexes.
> *Journal of Business and Economic Statistics*, **20**(2), 147–162.
> doi: 10.1198/073500102317351921

Introduced diffusion index forecasting: using the first few principal components of a large
panel of economic indicators as predictors. This paper established the empirical case for
approximate factor models in macroeconomics, motivating the DFM implementation in
`kalmanbox.models.DFM`.

---

### Bai & Ng (2002)

> **Bai, J. & Ng, S. (2002).**
> Determining the number of factors in approximate factor models.
> *Econometrica*, **70**(1), 191–221.
> doi: 10.1111/1468-0262.00273

Derived information criteria (`IC_p1`, `IC_p2`, `IC_p3`) for consistently selecting the
number of factors in large approximate factor models. The factor number selection routines
in `kalmanbox.models.DFM.select_n_factors()` are based on these criteria.

---

### Bai (2003)

> **Bai, J. (2003).**
> Inferential theory for factor models of large dimensions.
> *Econometrica*, **71**(1), 135–171.
> doi: 10.1111/1468-0262.00392

Established asymptotic theory for estimated factors and loadings when both the cross-section
dimension N and time dimension T grow. Provides the theoretical justification for treating
estimated common factors as observed in a second-stage regression.

---

### Forni, Hallin, Lippi & Reichlin (2000)

> **Forni, M., Hallin, M., Lippi, M. & Reichlin, L. (2000).**
> The generalized dynamic factor model: identification and estimation.
> *The Review of Economics and Statistics*, **82**(4), 540–554.
> doi: 10.1162/003465300559037

Proposed the Generalized Dynamic Factor Model (GDFM) in which factors are loaded with lags,
allowing richer cross-spectral structure than the static factor model. Referenced in the
`kalmanbox` documentation as context for the simpler static DFM implementation.

---

### Giannone, Reichlin & Small (2008)

> **Giannone, D., Reichlin, L. & Small, D. (2008).**
> Nowcasting: the real-time informational content of macroeconomic data.
> *Journal of Monetary Economics*, **55**(4), 665–676.
> doi: 10.1016/j.jmoneco.2008.05.010

Combined the DFM with the Kalman filter to handle ragged-edge data (panels where different
series are released at different lags) for GDP nowcasting. This paper is the key reference
for missing-data handling in `kalmanbox.models.DFM` when applied to unbalanced panels.

---

### Doz, Giannone & Reichlin (2012)

> **Doz, C., Giannone, D. & Reichlin, L. (2012).**
> A quasi-maximum likelihood approach for large approximate dynamic factor models.
> *The Review of Economics and Statistics*, **94**(4), 1014–1024.
> doi: 10.1162/REST_a_00225

Showed that a two-step estimator — PCA followed by Kalman filter/smoother with fixed
parameters — is consistent and asymptotically normal for large N. This is the default
estimation strategy for `kalmanbox.models.DFM` when the panel dimension is large.

---

### Banbura, Giannone & Reichlin (2010)

> **Banbura, M., Giannone, D. & Reichlin, L. (2010).**
> Large Bayesian vector auto regressions.
> *Journal of Applied Econometrics*, **25**(1), 71–92.
> doi: 10.1002/jae.1137

Demonstrated that Bayesian shrinkage (Minnesota-type priors) allows consistent estimation of
large VAR models, matching or exceeding DFM forecasting performance in many settings. The
Bayesian DFM extensions in `kalmanbox` are informed by the shrinkage ideas developed here.

---

### Onatski (2010)

> **Onatski, A. (2010).**
> Determining the number of factors from empirical distribution of eigenvalues.
> *The Review of Economics and Statistics*, **92**(4), 1004–1016.
> doi: 10.1162/REST_a_00043

Proposed a test for the number of factors based on the spacing of adjacent eigenvalues of the
sample covariance matrix, using results from random matrix theory. An alternative to the Bai &
Ng (2002) criteria available in `DFM.select_n_factors()`.

---

### Ahn & Horenstein (2013)

> **Ahn, S. C. & Horenstein, A. R. (2013).**
> Eigenvalue ratio test for the number of factors.
> *Econometrica*, **81**(3), 1203–1227.
> doi: 10.3982/ECTA8968

Introduced the eigenvalue ratio (ER) and growth ratio (GR) criteria as simple, consistent
estimators for the number of factors. The ER criterion is particularly easy to compute and
is included as a default option in `DFM.select_n_factors()`.

---

## 7. Nonlinear and Non-Gaussian Filters

### Julier & Uhlmann (1997)

> **Julier, S. J. & Uhlmann, J. K. (1997).**
> A new extension of the Kalman filter to nonlinear systems.
> In *Proceedings of SPIE 3068, Signal Processing, Sensor Fusion, and Target Recognition VI*,
> pp. 182–193. SPIE, Orlando, FL.
> doi: 10.1117/12.280797

The original conference paper introducing the Unscented Transform and the Unscented Kalman
Filter (UKF). Proposed approximating a Gaussian distribution by a deterministic set of sigma
points and propagating them through the nonlinear function, rather than linearising the
function as in the EKF. Implemented in `kalmanbox.filters.UKF`.

---

### Julier & Uhlmann (2004)

> **Julier, S. J. & Uhlmann, J. K. (2004).**
> Unscented filtering and nonlinear estimation.
> *Proceedings of the IEEE*, **92**(3), 401–422.
> doi: 10.1109/JPROC.2003.823141

Comprehensive review of the UKF, including the generalised unscented transform, the scaled
sigma-point algorithm, and extensions to non-additive noise. This is the primary reference
for the sigma-point selection rules and weight computation in `kalmanbox.filters.UKF`.

---

### Wan & van der Merwe (2000)

> **Wan, E. A. & van der Merwe, R. (2000).**
> The unscented Kalman filter for nonlinear estimation.
> In *Proceedings of the IEEE Adaptive Systems for Signal Processing, Communications, and
> Control Symposium (AS-SPCC)*, pp. 153–158. IEEE, Lake Louise, Canada.
> doi: 10.1109/ASSPCC.2000.882463

Demonstrated the UKF's empirical superiority over the EKF in parameter estimation and neural
network training tasks. Introduced the `alpha`, `beta`, `kappa` parameterisation of the
scaled unscented transform used in `kalmanbox.filters.UKF`.

---

### Evensen (2003)

> **Evensen, G. (2003).**
> The Ensemble Kalman Filter: theoretical formulation and practical implementation.
> *Ocean Dynamics*, **53**(4), 343–367.
> doi: 10.1007/s10236-003-0036-9

The definitive review paper on the Ensemble Kalman Filter (EnKF). Covers the analysis step
with perturbed observations, covariance localisation, inflation, and practical implementation
details. This is the primary reference for `kalmanbox.filters.EnsembleKF`.

---

### Burgers, van Leeuwen & Evensen (1998)

> **Burgers, G., van Leeuwen, P. J. & Evensen, G. (1998).**
> Analysis scheme in the ensemble Kalman filter.
> *Monthly Weather Review*, **126**(6), 1719–1724.
> doi: 10.1175/1520-0493(1998)126<1719:ASITEK>2.0.CO;2

Showed that perturbing the observations in the EnKF analysis step is necessary to obtain
consistent ensemble spread, and derived the stochastic analysis scheme. The observation
perturbation in `EnsembleKF` follows this paper.

---

### Gordon, Salmond & Smith (1993)

> **Gordon, N. J., Salmond, D. J. & Smith, A. F. M. (1993).**
> Novel approach to nonlinear/non-Gaussian Bayesian state estimation.
> *IEE Proceedings F — Radar and Signal Processing*, **140**(2), 107–113.
> doi: 10.1049/ip-f-2.1993.0015

Introduced the Bootstrap Particle Filter: a sequential importance resampling algorithm that
approximates the filtering distribution by a weighted set of random samples (particles).
Referenced in `kalmanbox` as the theoretical link between the Kalman filter and the
`particlefilterbox` library.

---

### Arulampalam et al. (2002)

> **Arulampalam, M. S., Maskell, S., Gordon, N. & Clapp, T. (2002).**
> A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking.
> *IEEE Transactions on Signal Processing*, **50**(2), 174–188.
> doi: 10.1109/78.978374

Comprehensive tutorial on sequential Monte Carlo methods. Covers importance sampling,
resampling schemes (multinomial, stratified, systematic), auxiliary PF, and regularised PF.
Primary reference for the `particlefilterbox` library that extends `kalmanbox` to the
non-Gaussian setting.

---

## 8. Bayesian State-Space Methods

### Carter & Kohn (1994)

> **Carter, C. K. & Kohn, R. (1994).**
> On Gibbs sampling for state space models.
> *Biometrika*, **81**(3), 541–553.
> doi: 10.1093/biomet/81.3.541

Introduced the Forward Filtering Backward Sampling (FFBS) algorithm for drawing the entire
state trajectory jointly from its posterior distribution conditional on parameters and
observations. This is the key building block of `kalmanbox.estimation.GibbsSampler` and is
used in the `kalmanbox.estimation.FFBS` smoother.

---

### Frühwirth-Schnatter (1994)

> **Frühwirth-Schnatter, S. (1994).**
> Data augmentation and dynamic linear models.
> *Journal of Time Series Analysis*, **15**(2), 183–202.
> doi: 10.1111/j.1467-9892.1994.tb00184.x

Independently derived the FFBS algorithm (simultaneously with Carter & Kohn) and embedded it
in a full data-augmentation Gibbs sampler for dynamic linear models, deriving the full
conditional distributions for the variance parameters. The Gibbs sampler structure in
`kalmanbox.estimation.GibbsSampler` mirrors this paper.

---

### Gelman, Carlin, Stern, Dunson, Vehtari & Rubin (2013)

> **Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. & Rubin, D. B. (2013).**
> *Bayesian Data Analysis*, 3rd edition.
> Chapman & Hall/CRC Texts in Statistical Science.
> CRC Press, Boca Raton, FL.
> ISBN 978-1-439-84095-5.

Standard graduate-level text on Bayesian analysis. Chapters 11–12 cover MCMC methods
(Gibbs sampling, Metropolis–Hastings); Chapter 17 covers hierarchical linear models that
generalise dynamic linear models. Background reference for prior specification in
`kalmanbox.estimation`.

---

### Gilks, Richardson & Spiegelhalter (1996)

> **Gilks, W. R., Richardson, S. & Spiegelhalter, D. J. (Eds.) (1996).**
> *Markov Chain Monte Carlo in Practice*.
> Chapman & Hall, London.
> ISBN 978-0-412-05551-5.

Edited volume covering the theory and practice of MCMC methods. Chapters on Gibbs sampling,
convergence diagnostics (Gelman–Rubin statistic), and practical implementation. Referenced
for the convergence diagnostics in `kalmanbox.estimation.PosteriorDiagnostics`.

---

### Robert & Casella (2004)

> **Robert, C. P. & Casella, G. (2004).**
> *Monte Carlo Statistical Methods*, 2nd edition.
> Springer Texts in Statistics.
> Springer, New York.
> ISBN 978-0-387-21239-5.

Comprehensive treatment of Monte Carlo methods including importance sampling, rejection
sampling, MCMC, and variance reduction techniques. Background reference for the simulation
methods used in `kalmanbox.estimation` and extended in `particlefilterbox`.

---

### Chib (1996)

> **Chib, S. (1996).**
> Calculating posterior distributions and modal estimates in Markov mixture models.
> *Journal of Econometrics*, **75**(1), 79–97.
> doi: 10.1016/0304-4076(95)01770-4

Derived the Gibbs sampler for hidden Markov models and Markov-switching regression models,
including efficient sampling of the discrete latent state sequence. Referenced in the
`kalmanbox` documentation on regime-switching extensions.

---

## 9. Numerical Methods and Stability

### Bierman (1977)

> **Bierman, G. J. (1977).**
> *Factorization Methods for Discrete Sequential Estimation*.
> Mathematics in Science and Engineering, Vol. 128.
> Academic Press, New York.
> ISBN 978-0-120-97350-9.
> (Reprinted by Dover Publications, 2006. ISBN 978-0-486-44981-7.)

The classic reference for square-root and U-D factorisation methods for the Kalman filter.
Showed that propagating the Cholesky factor (or U-D factor) of the covariance matrix, rather
than the covariance matrix itself, yields numerically stable filter recursions that do not
suffer from loss of positive-definiteness. Directly implemented in
`kalmanbox.filters.SquareRootFilter`.

---

### Kailath, Sayed & Hassibi (2000)

> **Kailath, T., Sayed, A. H. & Hassibi, B. (2000).**
> *Linear Estimation*.
> Prentice-Hall Information and Systems Sciences Series.
> Prentice-Hall, Upper Saddle River, NJ.
> ISBN 978-0-130-22464-4.

Advanced treatment of linear estimation covering Wiener filtering, Kalman filtering, H-infinity
filtering, and array algorithms. Chapters 12–13 on the information filter and array forms are
the theoretical basis for `kalmanbox.filters.InformationFilter`.

---

### Golub & van Loan (2013)

> **Golub, G. H. & van Loan, C. F. (2013).**
> *Matrix Computations*, 4th edition.
> Johns Hopkins Studies in the Mathematical Sciences.
> Johns Hopkins University Press, Baltimore, MD.
> ISBN 978-1-421-40859-0.

The standard reference for numerical linear algebra. QR decomposition, Cholesky factorisation,
and singular value decomposition algorithms used in `kalmanbox` follow the numerically stable
implementations described in this book, particularly for the square-root filter and the
information filter.

---

### Watkins (2002)

> **Watkins, D. S. (2002).**
> *Fundamentals of Matrix Computations*, 2nd edition.
> Pure and Applied Mathematics.
> Wiley-Interscience, New York.
> ISBN 978-0-471-21394-4.

Accessible numerical linear algebra text covering Gaussian elimination, Cholesky
factorisation, QR algorithms, and eigenvalue problems. Background reference for the
computational methods underlying `kalmanbox`'s matrix operations.

---

## 10. Information Criteria and Model Selection

Model selection is relevant to choosing the number of structural components, the lag order in
ARIMA-SSM, and the number of factors in DFM.

### Akaike (1974)

> **Akaike, H. (1974).**
> A new look at the statistical model identification.
> *IEEE Transactions on Automatic Control*, **19**(6), 716–723.
> doi: 10.1109/TAC.1974.1100705

Introduced the Akaike Information Criterion (AIC = -2 log L + 2k). Used by
`kalmanbox.estimation.MLE` for model comparison via `result.aic`.

---

### Schwarz (1978)

> **Schwarz, G. (1978).**
> Estimating the dimension of a model.
> *The Annals of Statistics*, **6**(2), 461–464.
> doi: 10.1214/aos/1176344136

Introduced the Bayesian Information Criterion (BIC = -2 log L + k log n). Available via
`result.bic`. BIC imposes a heavier penalty than AIC and is preferred for large samples
when the goal is model identification rather than prediction.

---

### Hannan & Quinn (1979)

> **Hannan, E. J. & Quinn, B. G. (1979).**
> The determination of the order of an autoregression.
> *Journal of the Royal Statistical Society, Series B (Methodological)*, **41**(2), 190–195.
> doi: 10.1111/j.2517-6161.1979.tb01072.x

Introduced the Hannan–Quinn Criterion (HQC = -2 log L + 2k log log n), which is strongly
consistent for lag order selection and penalises more than AIC but less than BIC. Available
via `result.hqc`.

---

### Burnham & Anderson (2002)

> **Burnham, K. P. & Anderson, D. R. (2002).**
> *Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach*,
> 2nd edition.
> Springer, New York.
> ISBN 978-0-387-95364-9.

Comprehensive practical guide to information-theoretic model selection, including AIC
corrected for small samples (AICc), model averaging, and evidence ratios. The `result.aicc`
attribute in `kalmanbox` follows the small-sample correction derived in this book.

---

## 11. Diagnostics and Residual Analysis

### Harvey & Koopman (1992)

> **Harvey, A. C. & Koopman, S. J. (1992).**
> Diagnostic checking of unobserved-components time series models.
> *Journal of Business and Economic Statistics*, **10**(4), 377–389.
> doi: 10.1080/07350015.1992.10509913

Introduced the framework of auxiliary residuals for structural time series models: computing
the smoothed disturbances for each unobserved component (trend, seasonal, irregular) and
using them to identify outliers, level shifts, and slope breaks. The `kalmanbox` diagnostic
module follows this framework.

---

### Koopman, Harvey, Doornik & Shephard — STAMP

> **Koopman, S. J., Harvey, A. C., Doornik, J. A. & Shephard, N. (2009).**
> *STAMP 8.3: Structural Time Series Analyser, Modeller and Predictor*.
> Timberlake Consultants, London.

The STAMP software manual is a practical guide to structural time series modelling and
diagnostics, written by the authors of the underlying theory. Diagnostics implemented in
`kalmanbox` (heteroscedasticity test, normality test, autocorrelation test for standardised
residuals) follow STAMP conventions.

---

### Ljung & Box (1978)

> **Ljung, G. M. & Box, G. E. P. (1978).**
> On a measure of lack of fit in time series models.
> *Biometrika*, **65**(2), 297–303.
> doi: 10.1093/biomet/65.2.297

Introduced the Ljung–Box portmanteau test for autocorrelation in residuals, a finite-sample
corrected version of the Box–Pierce statistic. Applied in `kalmanbox` via
`result.test_autocorrelation(lags=...)`.

---

### Jarque & Bera (1980)

> **Jarque, C. M. & Bera, A. K. (1980).**
> Efficient tests for normality, homoscedasticity and serial independence of regression
> residuals.
> *Economics Letters*, **6**(3), 255–259.
> doi: 10.1016/0165-1765(80)90024-5

Introduced the Jarque–Bera test for normality of residuals, based on the sample skewness and
excess kurtosis. Used in `kalmanbox` via `result.test_normality()` to verify the Gaussian
innovations assumption underlying the Kalman filter.

---

## 12. Related Software

The table below situates `kalmanbox` within the broader ecosystem of state-space and Kalman
filter software.

| Package | Language | Primary domain | Relationship to `kalmanbox` |
|---------|----------|----------------|------------------------------|
| `statsmodels.tsa.statespace` | Python | Econometrics | Closest relative; informed several `kalmanbox` APIs |
| `KFAS` | R | Statistics | Comprehensive; shares Durbin & Koopman conventions |
| `dlm` | R | Bayesian DLMs | Companion to Petris et al. (2009) |
| `MARSS` | R | Ecology/multivariate | Multivariate state-space with EM |
| `pykalman` | Python | General | Lightweight; basic KF and EM |
| `simdkalman` | Python | General | Vectorised batch Kalman filter |
| `filterpy` | Python | Engineering | Engineering-oriented; EKF, UKF, PF |
| `JAGS` / `Stan` | R / Python | General Bayesian | Full probabilistic programming; SSMs as special case |

---

### statsmodels (`statsmodels.tsa.statespace`)

> **Fulton, C. (2015–present).**
> `statsmodels.tsa.statespace` — State space models in statsmodels.
> Part of the `statsmodels` Python library.
> Source: [github.com/statsmodels/statsmodels](https://github.com/statsmodels/statsmodels)
> Documentation: [www.statsmodels.org/stable/statespace.html](https://www.statsmodels.org/stable/statespace.html)

The most comprehensive state-space implementation in Python. Covers SARIMA, VARMAX, DFM, UCM,
and local linear trend. Uses Durbin & Koopman (2012) notation and implements the exact diffuse
filter. `kalmanbox` draws on `statsmodels.tsa.statespace` as a reference implementation and
offers a complementary API focused on modularity and extensibility.

!!! note "Key differences from `kalmanbox`"
    `statsmodels` integrates deeply with `pandas` and provides `summary()` tables for econometric
    output. `kalmanbox` prioritises composable filter/model/estimator objects and is designed as
    a foundation for the NodesEcon ecosystem rather than a standalone end-user package.

---

### KFAS (R)

> **Helske, J. (2017).**
> KFAS: Exponential Family State Space Models in R.
> *Journal of Statistical Software*, **78**(10), 1–39.
> doi: 10.18637/jss.v078.i10
> Source: [CRAN — KFAS](https://cran.r-project.org/package=KFAS)

Implements the Kalman filter and smoother for linear Gaussian and exponential family
(Poisson, binomial, gamma, negative binomial) state-space models in R. Uses the exact diffuse
initialisation following Koopman & Durbin (2003). A reference implementation against which
`kalmanbox` results can be validated.

---

### dlm (R)

> **Petris, G. (2010).**
> An R package for dynamic linear models.
> *Journal of Statistical Software*, **36**(12), 1–16.
> doi: 10.18637/jss.v036.i12
> Source: [CRAN — dlm](https://cran.r-project.org/package=dlm)

Companion package to Petris, Petrone & Campagnoli (2009). Supports Bayesian analysis of DLMs
via FFBS and MCMC, as well as MLE. Model specification follows the West & Harrison (1997)
framework.

---

### MARSS (R)

> **Holmes, E. E., Ward, E. J. & Wills, K. (2012).**
> MARSS: Multivariate autoregressive state-space models for analyzing time-series data.
> *The R Journal*, **4**(1), 11–19.
> Source: [CRAN — MARSS](https://cran.r-project.org/package=MARSS)

Fits multivariate autoregressive state-space (MARSS) models using EM. Common in ecology for
analysing population dynamics. The multivariate model structure and the EM algorithm
implementation in `kalmanbox.models.DFM` were cross-referenced against MARSS.

---

### pykalman (Python)

> **Duckworth, D. & contributors (2012–present).**
> `pykalman` — Kalman Filter, Smoother, and EM Algorithm for Python.
> Source: [github.com/pykalman/pykalman](https://github.com/pykalman/pykalman)

Lightweight Kalman filter implementation with scikit-learn-compatible API. Supports basic KF,
UKF, and EM parameter estimation. `kalmanbox` provides a superset of `pykalman`'s
functionality with additional structural models, diffuse initialisation, and Bayesian
estimation.

---

### simdkalman (Python)

> **Solin, A. & contributors (2016–present).**
> `simdkalman` — Vectorised Kalman filter for Python.
> Source: [github.com/oseiskar/simdkalman](https://github.com/oseiskar/simdkalman)

Implements the Kalman filter as fully vectorised NumPy operations for batch processing of
many independent time series simultaneously. Relevant for high-throughput applications.
`kalmanbox` provides batch-mode filtering via `KalmanFilter.filter_batch()` with comparable
performance.

---

### filterpy (Python)

> **Labbe, R. (2014–present).**
> `filterpy` — Kalman and Bayesian Filters in Python.
> Source: [github.com/rlabbe/filterpy](https://github.com/rlabbe/filterpy)
> Companion book: *Kalman and Bayesian Filters in Python* (open access).

Engineering-oriented implementation covering EKF, UKF, particle filter, and several
adaptive filter variants. The companion Jupyter notebook book is an excellent introduction
to the intuition behind the filters implemented in `kalmanbox`.

---

### JAGS and Stan

> **Plummer, M. (2003).**
> JAGS: A program for analysis of Bayesian graphical models using Gibbs sampling.
> In *Proceedings of the 3rd International Workshop on Distributed Statistical Computing
> (DSC 2003)*, Vienna.
> Source: [mcmc-jags.sourceforge.io](http://mcmc-jags.sourceforge.io)

> **Carpenter, B. et al. (2017).**
> Stan: A probabilistic programming language.
> *Journal of Statistical Software*, **76**(1), 1–32.
> doi: 10.18637/jss.v076.i01
> Source: [mc-stan.org](https://mc-stan.org)

General probabilistic programming languages in which state-space models can be expressed and
fitted via MCMC or variational inference. `kalmanbox` is specialised and faster for linear
Gaussian models, but JAGS and Stan are more flexible for non-standard observation equations
or non-Gaussian noise distributions.

---

## 13. NodesEcon Ecosystem

`kalmanbox` is one module in the **NodesEcon** collection of composable Python libraries for
quantitative economics and time series analysis. The packages are designed so that each builds
on the interfaces defined by the one below it.

```
chronobox          ← high-level time series toolbox
    │
forecastbox        ← forecasting framework (point & probabilistic)
    │
kalmanbox          ← state-space models and Kalman filtering   ← you are here
    │
particlefilterbox  ← sequential Monte Carlo (non-Gaussian / nonlinear)
```

---

### chronobox

> **NodesEcon (2024–present).**
> `chronobox` — Time series toolbox built on `kalmanbox`.
> Source: [github.com/NodesEcon/chronobox](https://github.com/NodesEcon/chronobox)

`chronobox` provides high-level time series utilities — data loading, preprocessing, seasonal
adjustment, and visualisation — that call `kalmanbox` models internally. Users who want a
batteries-included analysis environment should start with `chronobox`; users who need direct
control over filter parameters and likelihood functions should use `kalmanbox` directly.

**Key `kalmanbox` features used by `chronobox`:**

- `BSM` for seasonal adjustment
- `LocalLinearTrend` for trend extraction
- `KalmanFilter` for signal extraction
- `MLE` for automatic parameter estimation

---

### forecastbox

> **NodesEcon (2024–present).**
> `forecastbox` — Probabilistic forecasting framework built on `kalmanbox`.
> Source: [github.com/NodesEcon/forecastbox](https://github.com/NodesEcon/forecastbox)

`forecastbox` provides point and probabilistic forecasting utilities including forecast
evaluation (CRPS, quantile score, interval coverage), model combination, and calibration.
Internally it uses `kalmanbox` for state-space-based forecast distributions.

**Key `kalmanbox` interfaces consumed by `forecastbox`:**

- `KalmanFilter.forecast(h)` — returns mean and covariance of h-step-ahead forecast
- `RTSSmoother.smooth()` — for backcasting and interpolation
- `MLE.fit()` — for estimation before forecasting

---

### particlefilterbox

> **NodesEcon (2024–present).**
> `particlefilterbox` — Sequential Monte Carlo extending `kalmanbox` to non-Gaussian and
> nonlinear models.
> Source: [github.com/NodesEcon/particlefilterbox](https://github.com/NodesEcon/particlefilterbox)

`particlefilterbox` implements bootstrap particle filter, auxiliary particle filter, and
Rao–Blackwellised particle filter. It extends `kalmanbox`'s linear Gaussian filter classes
to handle non-Gaussian observation equations and nonlinear state transitions. The Rao–
Blackwellised PF uses `kalmanbox.filters.KalmanFilter` analytically for the linear
substructure and sequential Monte Carlo for the nonlinear part.

!!! note "When to use `particlefilterbox` vs `kalmanbox`"
    Use `kalmanbox` when your model is linear and Gaussian (or approximately so). Use
    `particlefilterbox` when observations follow a non-Gaussian distribution (e.g., count data,
    heavy-tailed errors) or when the state transition involves threshold or regime-switching
    nonlinearity that cannot be adequately handled by the EKF or UKF.

---

## Alphabetical index

For quick look-up, the table below lists every first author cited in this page.

| Author | Year | Section |
|--------|------|---------|
| Ahn & Horenstein | 2013 | Dynamic Factor Models |
| Akaike | 1974 | Information Criteria |
| Anderson & Moore | 1979 | Foundational Books |
| Arulampalam et al. | 2002 | Nonlinear Filters |
| Bai | 2003 | Dynamic Factor Models |
| Bai & Ng | 2002 | Dynamic Factor Models |
| Banbura, Giannone & Reichlin | 2010 | Dynamic Factor Models |
| Bierman | 1977 | Numerical Methods |
| Brockwell & Davis | 1991 | Foundational Books |
| Bryson & Ho | 1969 | Seminal Papers |
| Bucy & Joseph | 1968 | Seminal Papers |
| Burgers, van Leeuwen & Evensen | 1998 | Nonlinear Filters |
| Burnham & Anderson | 2002 | Information Criteria |
| Carter & Kohn | 1994 | Bayesian Methods |
| Carpenter et al. (Stan) | 2017 | Related Software |
| Chib | 1996 | Bayesian Methods |
| Commandeur & Koopman | 2007 | Foundational Books |
| De Jong | 1991 | Diffuse Initialisation |
| Dempster, Laird & Rubin | 1977 | MLE and EM |
| Doz, Giannone & Reichlin | 2012 | Dynamic Factor Models |
| Durbin & Koopman | 2012 | Foundational Books |
| Evensen | 2003 | Nonlinear Filters |
| Forni, Hallin, Lippi & Reichlin | 2000 | Dynamic Factor Models |
| Francke, Koopman & de Vos | 2010 | Diffuse Initialisation |
| Frühwirth-Schnatter | 1994 | Bayesian Methods |
| Gelman et al. | 2013 | Bayesian Methods |
| Giannone, Reichlin & Small | 2008 | Dynamic Factor Models |
| Gilks, Richardson & Spiegelhalter | 1996 | Bayesian Methods |
| Golub & van Loan | 2013 | Numerical Methods |
| Gordon, Salmond & Smith | 1993 | Nonlinear Filters |
| Hamilton | 1994 | MLE and EM |
| Hannan & Quinn | 1979 | Information Criteria |
| Harvey | 1989 | Foundational Books |
| Harvey & Jaeger | 1993 | Structural Models |
| Harvey & Koopman | 1992 | Diagnostics |
| Harvey & Phillips | 1979 | Diffuse Initialisation |
| Harvey & Todd | 1983 | Structural Models |
| Harvey (1985) | 1985 | Structural Models |
| Holmes, Ward & Wills (MARSS) | 2012 | Related Software |
| Jarque & Bera | 1980 | Diagnostics |
| Julier & Uhlmann | 1997 | Nonlinear Filters |
| Julier & Uhlmann | 2004 | Nonlinear Filters |
| Kailath, Sayed & Hassibi | 2000 | Numerical Methods |
| Kalman | 1960 | Seminal Papers |
| Kim & Nelson | 1999 | Foundational Books |
| Koopman | 1997 | Diffuse Initialisation |
| Koopman & Durbin | 2003 | Diffuse Initialisation |
| Koopman, Harvey, Doornik & Shephard | 2009 | Diagnostics |
| Labbe (filterpy) | 2014 | Related Software |
| Ljung & Box | 1978 | Diagnostics |
| Maravall & Planas | 1999 | Structural Models |
| Onatski | 2010 | Dynamic Factor Models |
| Petris (dlm) | 2010 | Related Software |
| Petris, Petrone & Campagnoli | 2009 | Foundational Books |
| Plummer (JAGS) | 2003 | Related Software |
| Rauch, Tung & Striebel | 1965 | Seminal Papers |
| Robert & Casella | 2004 | Bayesian Methods |
| Schwarz | 1978 | Information Criteria |
| Schweppe | 1965 | MLE and EM |
| Shumway & Stoffer | 1982 | MLE and EM |
| Shumway & Stoffer | 2017 | Foundational Books |
| Stock & Watson | 2002 | Dynamic Factor Models |
| Wan & van der Merwe | 2000 | Nonlinear Filters |
| Watkins | 2002 | Numerical Methods |
| Watson & Engle | 1983 | MLE and EM |
| West & Harrison | 1997 | Foundational Books |
