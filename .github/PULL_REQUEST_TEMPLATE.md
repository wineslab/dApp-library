<!--
Thank you for contributing to dapps. Please fill in every section.
PRs that leave the mandatory boxes unchecked will not be reviewed.
-->

## Summary

<!-- 1–3 bullets describing what this PR changes and *why*. -->

-
-

## Type of change

- [ ] Bug fix
- [ ] New feature / enhancement
- [ ] New Service Model
- [ ] Refactor (no behavior change)
- [ ] Documentation
- [ ] Test / CI
- [ ] Other (explain):

## Linked issue

<!-- Required. Use "Closes #N" so the issue is auto-closed on merge. -->

Closes #

## Mandatory test checklist

These mirror what CI (`.github/workflows/test.yml`) enforces. **All boxes must be ticked before review.**

- [ ] `hatch run pytest tests/ -v` passes locally on Python 3.12
- [ ] `hatch build` succeeds (verifies packaging metadata)
- [ ] `VERSION` file bumped per [SemVer](https://semver.org/) if the public API or wire protocol changed
- [ ] `README.md` updated if interface, environment variables, or workflow changed
- [ ] `CONTRIBUTING.md` updated if contributor-facing rules changed
- [ ] If new dependencies were added, they appear in `pyproject.toml` (and the right optional-extras group)

## CI checklist

- [ ] `Run tests` workflow is green on the latest commit of this PR

## Twin-repo coordination

dapps is paired with [`libe3`](https://github.com/wineslab/dApp-libe3) and [`dApp-openairinterface5g`](https://github.com/wineslab/dApp-openairinterface5g). Breaking changes to the twin repos are not accepted.

- [ ] This PR does not break the E3 wire protocol, OR a paired PR exists in the affected twin repo (link below).

Paired PR(s):

## Workflow confirmation

- [ ] This PR was opened against the **internal** repository (private). The public mirror `wineslab/dApp-library` is updated automatically by `.github/workflows/mirror.yml`.
- [ ] Commits will be **squashed** at merge time. Updates to this PR will be applied via **rebase** only — no merge commits, no duplicated history. (See `CONTRIBUTING.md` § Pull Request Process.)
- [ ] I have read and followed `CONTRIBUTING.md`.
