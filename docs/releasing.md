# Releasing

How a version reaches PyPI.

## One-time setup

Publishing uses **Trusted Publishing**, so no API token is ever stored in the
repository. PyPI authenticates GitHub's OIDC identity for this workflow.

Do it in this order. No manual upload is needed at any point.

### 1. Register a pending publisher on TestPyPI

At <https://test.pypi.org/manage/account/publishing/>, add:

| Field | Value |
|---|---|
| PyPI Project Name | `voice-anonymization-legal-eval` |
| Owner | `brijmohan` |
| Repository name | `voice-anonymization-legal-eval` |
| Workflow name | `publish.yml` |
| Environment name | `testpypi` |

A *pending* publisher is how you claim a name that does not exist yet. The first
successful run creates the project.

### 2. Register the same on PyPI

Identical, at <https://pypi.org/manage/account/publishing/>, except the
environment name is `pypi`.

### 3. Create both GitHub environments

Repository **Settings, Environments**, then **New environment**, twice:
`testpypi` and `pypi`.

On `pypi`, add yourself under **Required reviewers**. That turns a release into
something you approve rather than something a tag does behind your back.

### 4. Rehearse on TestPyPI

Trigger the workflow. Nothing is uploaded until this runs:

```sh
gh workflow run publish.yml -f target=testpypi
gh run watch "$(gh run list --workflow=publish.yml --limit 1 --json databaseId -q '.[0].databaseId')"
```

Or in the browser: **Actions, Publish to PyPI, Run workflow**, target `testpypi`.

Check it actually uploaded before moving on. A version number can only be used
once on an index, so a failed run that uploaded nothing is easy to retry, while
a successful one means the next attempt needs a new version:

```sh
curl -s -o /dev/null -w "%{http_code}\n" \
    https://test.pypi.org/pypi/voice-anonymization-legal-eval/json   # 200 once published
```

Then confirm the artifact is real:

```sh
python -m venv /tmp/rehearsal
/tmp/rehearsal/bin/pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple voice-anonymization-legal-eval
/tmp/rehearsal/bin/python -c "
from legal_eval.paper import load_paper_results
print(len(load_paper_results()), 'metrics loaded from the published wheel')"
/tmp/rehearsal/bin/legal-eval demo --output-dir /tmp/rehearsal-demo
```

The `--extra-index-url` is needed because TestPyPI does not mirror NumPy.

### 5. Release for real

```sh
git tag v2.0.0 && git push origin v2.0.0
gh run watch "$(gh run list --workflow=publish.yml --limit 1 --json databaseId -q '.[0].databaseId')"
```

Approve the `pypi` environment when GitHub asks. Then verify:

```sh
python -m venv /tmp/live && /tmp/live/bin/pip install voice-anonymization-legal-eval
/tmp/live/bin/legal-eval demo --output-dir /tmp/live-demo
```

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

Not needed if you followed the setup above. Kept for the case where Trusted
Publishing is unavailable.

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

## When something goes wrong

**`No matching distribution found` during the rehearsal.** The package is not on
the index, which almost always means the publish workflow never ran. Check:

```sh
gh run list --workflow=publish.yml --limit 5
```

An empty list means no run exists, so trigger it as in step 4. A failed run
means the logs will say why, usually a missing pending publisher or an
environment name that does not match the one registered on the index.

**`Trusted publishing exchange failure`.** The pending publisher's four fields
must match the workflow exactly: owner, repository, workflow filename
(`publish.yml`, not a path), and environment name. The environment is the field
that is most often wrong, since it differs between the two indexes: `testpypi`
and `pypi`.

**`File already exists`.** That version was already uploaded. Indexes never allow
reusing a version number, even after deleting the release. Bump
`legal_eval/__about__.py` and try again.

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
