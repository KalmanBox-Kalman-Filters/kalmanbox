# Release Process

This page is for **maintainers** with PyPI upload credentials and
write access to the repository. Contributors do not need to follow
this process.

kalmanbox uses [Semantic Versioning](https://semver.org/):
`MAJOR.MINOR.PATCH`.

| Increment | When to use                                              |
|-----------|----------------------------------------------------------|
| PATCH     | Bug fixes, documentation corrections, performance tweaks |
| MINOR     | New features, new models, new public API (backwards compatible) |
| MAJOR     | Breaking API changes (rare)                              |

## Step 1 — Bump the version

Edit `kalmanbox/__version__.py`:

```python
__version__ = "0.5.1"   # was "0.5.0"
```

Do **not** add `v` prefix here; that belongs only on the git tag.

## Step 2 — Update CHANGELOG.md

Move entries from the `[Unreleased]` section to a new versioned
section following [Keep a Changelog](https://keepachangelog.com/)
format:

```markdown
## [0.5.1] — 2026-05-15

### Fixed
- Square-Root filter no longer loses symmetry after 10,000+ steps
  on ill-conditioned covariances (#412).

### Changed
- `DFM.fit(method="em")` now uses the Louis (1982) correction for
  standard errors by default.

[0.5.1]: https://github.com/nodesecon/kalmanbox/compare/v0.5.0...v0.5.1
```

## Step 3 — Run the full test suite

```bash
pytest tests/ -x --tb=short
```

All tests must pass. Also run the pre-commit hooks over the entire
codebase:

```bash
pre-commit run --all-files
```

## Step 4 — Tag the release

```bash
git add kalmanbox/__version__.py CHANGELOG.md
git commit -m "Release v0.5.1"
git tag v0.5.1
git push origin main --tags
```

Pushing the tag triggers the CI release workflow, which:

1. Runs the test suite on all supported Python versions.
2. Builds the distribution packages.
3. Uploads to PyPI (requires the `PYPI_TOKEN` secret in GitHub Actions).

## Step 5 — Build and upload to PyPI (manual fallback)

If the CI workflow is unavailable, upload manually:

```bash
pip install hatch twine
hatch build                          # produces dist/kalmanbox-X.Y.Z.*
twine check dist/*                   # verify metadata
twine upload dist/*                  # prompts for PyPI credentials
```

Use a PyPI API token stored in `~/.pypirc` or passed via the
`TWINE_PASSWORD` environment variable. Never commit credentials.

## Step 6 — Documentation deployment

Documentation is deployed automatically via the
[Read the Docs](https://readthedocs.org/) webhook configured on the
repository. Pushing the tag causes RTD to build and publish the
versioned docs at `https://kalmanbox.readthedocs.io/en/vX.Y.Z/`.

Version aliases are managed with
[`mike`](https://github.com/jimporter/mike):

```bash
pip install mike
mike deploy --push --update-aliases 0.5.1 latest
mike set-default --push latest
```

This updates the `latest` alias to point to the new version and keeps
older docs accessible at their version-specific URL.

The `stable` alias points to the latest non-pre-release version. Update
it manually:

```bash
mike alias --push 0.5.1 stable --update
```

## Step 7 — Create a GitHub Release

1. Go to **Releases → Draft a new release** on GitHub.
2. Select the tag `v0.5.1`.
3. Set the title to `v0.5.1`.
4. Paste the relevant `CHANGELOG.md` section into the description.
5. Attach the built `dist/` artifacts if needed.
6. Publish the release.

The release triggers the GitHub Discussions auto-announcement bot.

## Pre-release versions

For release candidates, use the `rcN` suffix:

```
0.6.0rc1
0.6.0rc2
0.6.0
```

Pre-release tags (`v0.6.0rc1`) cause RTD to build a `vX.Y.ZrcN` doc
version but do not update `latest` or `stable`.

Upload pre-releases to PyPI's test instance first:

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ kalmanbox==0.6.0rc1
```

Verify the install and basic imports before promoting to the
production PyPI.
