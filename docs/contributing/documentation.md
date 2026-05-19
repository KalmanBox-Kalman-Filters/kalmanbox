# Documentation Contributions

The kalmanbox docs are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
and auto-generated API reference pages from
[mkdocstrings](https://mkdocstrings.github.io/).

## Where docs live

```
docs/
├── index.md                   # landing page
├── getting-started/
├── user-guide/
├── tutorials/
├── theory/
├── diagnostics/
├── visualization/
├── benchmarks/
├── faq/
├── contributing/
└── api/                       # mostly auto-generated from docstrings
mkdocs.yml                     # site configuration
```

New narrative pages (guides, tutorials, FAQ entries) go under the
appropriate subdirectory. API pages are generated automatically from
docstrings using `mkdocstrings` — you generally do not need to edit
files in `docs/api/` directly.

## Building locally

```bash
pip install -e ".[docs]"   # if not already installed
mkdocs serve               # live-reload server at http://127.0.0.1:8000
```

The server watches all `.md` files and `mkdocs.yml` for changes and
rebuilds automatically. For a one-shot static build:

```bash
mkdocs build               # output goes to site/
```

## Writing docstrings (NumPy style)

The `docs/api/` pages are generated from inline docstrings. Write
them in **NumPy style** so that mkdocstrings renders them correctly:

```python
def smooth(self) -> SmootherResult:
    """Run the RTS backward smoother.

    Returns the smoothed state means and covariances
    $a_{t|n}$ and $P_{t|n}$ for all $t$.

    Returns
    -------
    SmootherResult
        Contains ``a_smoothed`` of shape ``(n_obs, n_states)``
        and ``P_smoothed`` of shape ``(n_obs, n_states, n_states)``.

    Notes
    -----
    The smoother requires the full filtered state history.
    Calling this method after ``filter(store_history=False)``
    raises ``RuntimeError``.

    References
    ----------
    Rauch, Tung & Striebel (1965), "Maximum likelihood estimates of
    linear dynamic systems", AIAA Journal 3(8), pp. 1445–1450.

    Examples
    --------
    >>> result = model.fit()
    >>> sm = result.smooth()
    >>> sm.a_smoothed.shape
    (100, 1)
    """
```

mkdocstrings will render the `Returns`, `Notes`, `References`, and
`Examples` sections with proper formatting.

## Adding math

Use standard LaTeX syntax. MkDocs Material renders math via MathJax.

**Display math** (centred, on its own line):

```markdown
$$
P_{t|t} = (I - K_t Z_t)\, P_{t|t-1}
$$
```

**Inline math** (within a sentence):

```markdown
The Kalman gain is $K_t = P_{t|t-1} Z_t' F_t^{-1}$.
```

Do not use `\[...\]` or `$$..$$ ` with extra spaces — MathJax may
not render them. Always use `$$` on its own line for display math.

## Adding admonitions

MkDocs Material supports admonitions with the `!!!` syntax:

```markdown
!!! note "Optional custom title"
    Content of the note. Indent with 4 spaces.

!!! tip
    A helpful tip without a custom title.

!!! warning
    Something the reader should be careful about.

!!! example
    A brief code example inline.

!!! danger "Breaking change"
    A warning about a breaking API change.
```

Supported types: `note`, `tip`, `info`, `warning`, `danger`,
`example`, `abstract`, `question`, `success`, `failure`, `bug`, `quote`.

Use admonitions sparingly — only when the content genuinely warrants
special emphasis.

## Adding grid cards

Grid cards are used on index pages to provide a visual table of
contents:

```markdown
<div class="grid cards" markdown>

-   :material-filter:{ .lg .middle } **Card title**

    ---

    One or two sentences describing what this section covers.

    [:octicons-arrow-right-24: Link text](relative-path.md)

</div>
```

Browse [Material for MkDocs icons](https://squidfunk.github.io/mkdocs-material/reference/icons-emojis/)
to find appropriate `:material-xxx:` icons.

## Cross-linking

Link to other pages using relative Markdown paths:

```markdown
See [Kalman filter](../user-guide/kalman/kalman-filter.md).
```

For auto-cross-references to API symbols (powered by the `autorefs`
plugin), use:

```markdown
[`MLEstimator`][kalmanbox.estimation.mle.MLEstimator]
```

This generates a hyperlink to the `MLEstimator` API page, with the
class name as the link text. If the symbol is not found, a build
warning is emitted and the link degrades gracefully.

## Code blocks

Specify the language for syntax highlighting:

````markdown
```python
from kalmanbox import LocalLevel
model = LocalLevel(y)
```
````

Use `bash` for shell commands, `python` for Python, `text` for
plain output or config files.

## Adding a new page to the navigation

Edit `mkdocs.yml` and add the page path under the appropriate `nav`
section:

```yaml
nav:
  - Tutorials:
    - tutorials/index.md
    - tutorials/nile-local-level.md
    - tutorials/airline-bsm.md
    - tutorials/us-macro-dfm.md     # add new page here
```

Run `mkdocs serve` after editing `mkdocs.yml` to verify the page
appears in the sidebar.
