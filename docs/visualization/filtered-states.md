# Plotting filtered states

The filter produces $a_{t|t}$ and $P_{t|t}$ — the **online** estimate of
the latent state at each $t$. Plotting them shows how the model "tracks"
the data in real time.

## Quick plot

```python
from kalmanbox.visualization import plot_filtered

fig = plot_filtered(results, components="all", alpha=0.05)
fig.savefig("filtered.png", dpi=150)
```

`plot_filtered` does:

- One subplot per state component, or per requested component name.
- Filtered mean as a solid line.
- Shaded 95% band $\pm 1.96\sqrt{P_{t|t}}$.

## Layered with the data

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(y, "k.", label="data")
ax.plot(results.filtered_states[:, 0], color="C0", label=r"$a_{t|t}$")
band = 1.96 * results.filtered_states_se[:, 0]
ax.fill_between(range(len(y)),
                results.filtered_states[:, 0] - band,
                results.filtered_states[:, 0] + band,
                color="C0", alpha=0.2)
ax.legend(); ax.set_title("Local Level — filtered estimate")
```

## Filter vs. smoother in one figure

```python
from kalmanbox.visualization import plot_filtered_vs_smoothed

plot_filtered_vs_smoothed(results)
```

The two-sided smoothed estimate is **always at least as tight** as the
filtered estimate — that gap is the value of seeing the future.

## Related

- [Smoothed states](smoothed-states.md)
- [User guide: Kalman filter](../user-guide/kalman/kalman-filter.md)
