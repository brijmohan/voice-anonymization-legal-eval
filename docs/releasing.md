# Releasing

How a version reaches PyPI.

## One-time setup

Publishing uses PyPI **Trusted Publishing**, so no API token is stored in the
repository. GitHub's OIDC identity is what PyPI authenticates.

1. Create the project on PyPI by uploading the first release manually (below),
   or reserve it through [PyPI's pending publisher](https://pypi.org/manage/account/publishing/)
   flow before any upload exists.
2. On PyPI, under the project's **Publishing** settings, add a GitHub publisher:

   | Field | Value |
   |---|---|
   | Owner | `brijmohan` |
   | Repository | `voice-anonymization-legal-eval` |
   | Workflow | `publish.yml` |
   | Environment | `pypi` |

3. In the GitHub repository settings, create an environment named `pypi`. Adding
   a required reviewer there means a release cannot go out without a human
   approving it.

## Cutting a release

```sh
# 1. Bump the single source of truth. pyproject reads the version from here.
$EDITOR legal_eval/__about__.py

# 2. Record what changed.
$EDITOR CHANGELOG.md

# 3. Land it on main and let CI pass.
git commit -am "Release 2.1.0" && git push

# 4. Tag. The tag drives the publish workflow.
git tag v2.1.0 && git push origin v2.1.0
```

The workflow builds the sdist and wheel, runs `twine check`, asserts the tag
matches `__about__.__version__`, installs the wheel into a throwaway virtualenv
and confirms the published results load from it, then uploads to PyPI after the
`pypi` environment approval.

## Publishing by hand

Only needed for the very first upload, if you are not using the pending
publisher flow.

```sh
python -m pip install --upgrade build twine
rm -rf dist build
python -m build
python -m twine check dist/*

# Rehearse on TestPyPI first.
python -m twine upload --repository testpypi dist/*
python -m pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple voice-anonymization-legal-eval

python -m twine upload dist/*
```

## Pre-release checklist

- [ ] `pytest -q` passes.
- [ ] `ruff check legal_eval tests examples scripts` passes.
- [ ] `python examples/02_reproduce_paper_figures.py` reports every paper value
      reproducing. This is the regression test on the shipped data.
- [ ] `python -m build && python -m twine check dist/*` passes.
- [ ] A wheel installed into a clean virtualenv can
      `from legal_eval.paper import load_paper_results` and get all nine panels.
      The published results live **inside** the package for this reason; moving
      them back out would break every installed user.
- [ ] `legal-eval demo` runs from that clean virtualenv.
- [ ] `CHANGELOG.md` has an entry.
- [ ] No em or en dashes in the documentation:
      `git ls-files -z | xargs -0 grep -nP '[\x{2014}\x{2013}]'` returns nothing.

## Versioning

Semantic versioning, where the public contract is **the numbers the metrics
produce**, not only the Python API.

- **Major**: a metric's value changes for the same input, or a default that
  affects published results changes.
- **Minor**: new metrics, new estimators, new CLI commands, additional shipped
  data.
- **Patch**: fixes that leave every computed number identical.

Anything that moves a number needs an entry in
[`differences.md`](differences.md) as well as the changelog, so that a reader
comparing against the paper knows which version produced which value.
