# Simulation API

`kalmanbox.simulation` provides two tools for working with state-space
models in the generative direction: forward simulation of synthetic
datasets and bootstrap (particle) filtering for non-Gaussian or
nonlinear models.

## simulate

`simulate_ssm` draws state trajectories and observations from a fully
specified state-space model:

$$
\alpha_{t+1} = T_t\,\alpha_t + R_t\,\eta_t, \quad \eta_t \sim \mathcal{N}(0, Q_t)
$$
$$
y_t = Z_t\,\alpha_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, H_t)
$$

```python
from kalmanbox import LocalLevel
from kalmanbox.simulation import simulate_ssm

model = LocalLevel.__new__(LocalLevel)
ssr   = model._build_ssr(sigma2_eta=1200.0, sigma2_eps=14000.0, n=200)

sim = simulate_ssm(ssr, n=200, a0=np.array([900.0]), seed=7)
# sim.states  — shape (200, 1)
# sim.observations — shape (200, 1)
```

::: kalmanbox.simulation.simulate

## bootstrap

`bootstrap_filter` implements the Sequential Importance
Resampling (SIR) particle filter — the simplest member of the
sequential Monte Carlo family. It is suitable for non-Gaussian noise or
mild nonlinearities when the proposal is the prior transition density.

```python
from kalmanbox.simulation import bootstrap_filter

pf_out = bootstrap_filter(
    y,
    transition_fn=lambda particles, t, rng: T @ particles.T + rng.multivariate_normal(
        np.zeros(4), Q, size=particles.shape[0]
    ).T,
    observation_fn=lambda particles, t: h_vectorised(particles),
    obs_noise_cov=H,
    n_particles=2000,
    a0=a0,
    P0=P0,
    seed=0,
)
# pf_out.state_mean   — shape (n, k)
# pf_out.state_cov    — shape (n, k, k)
# pf_out.log_weights  — shape (n, n_particles)
```

!!! warning "Particle degeneracy"

    The bootstrap filter degrades rapidly for high state dimensions or
    very informative likelihoods. For production use with complex models
    consider [particlefilterbox](../getting-started/ecosystem.md), which
    provides auxiliary particle filters, twisted proposals, and SMC
    samplers.

::: kalmanbox.simulation.bootstrap
