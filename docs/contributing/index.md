# Contributing

kalmanbox welcomes contributions of all kinds: bug fixes, new models,
performance improvements, documentation, and test coverage. This
section describes the full contributor workflow.

## Contribution workflow

The standard path for a contribution is:

1. **Fork** the repository on GitHub to your own namespace.
2. Create a **feature branch** from `main`:
   `git checkout -b feat/my-feature`.
3. Make your changes following the project's code style and testing
   standards.
4. Open a **Pull Request** against `main` with a clear title and
   description.
5. Address review feedback; a maintainer will merge once CI passes
   and at least one approval is given.

For large features or API changes, open a **GitHub Discussion** first
to align on design before writing code.

<div class="grid cards" markdown>

-   :material-wrench:{ .lg .middle } **Development Setup**

    ---

    Fork, clone, virtual environment, pre-commit hooks, and verifying
    the test suite runs.

    [:octicons-arrow-right-24: Setup](setup.md)

-   :material-format-letter-case:{ .lg .middle } **Code Style**

    ---

    Line length, matrix naming, type hints, docstrings, and ruff / pyright
    configuration.

    [:octicons-arrow-right-24: Style](style.md)

-   :material-test-tube:{ .lg .middle } **Testing**

    ---

    pytest layout, numerical tolerances, property-based tests with
    Hypothesis, and mutation testing.

    [:octicons-arrow-right-24: Testing](testing.md)

-   :material-book-edit:{ .lg .middle } **Documentation**

    ---

    MkDocs Material syntax, docstrings, math, admonitions, and running
    a local doc server.

    [:octicons-arrow-right-24: Documentation](documentation.md)

-   :material-tag-outline:{ .lg .middle } **Release Process**

    ---

    Version bumping, changelog, PyPI upload, doc deployment, and GitHub
    releases (for maintainers).

    [:octicons-arrow-right-24: Release](release.md)

</div>

!!! note "Code of Conduct"
    All contributors are expected to follow the
    [Contributor Covenant Code of Conduct](https://github.com/nodesecon/kalmanbox/blob/main/CODE_OF_CONDUCT.md).
    Be kind and constructive.
