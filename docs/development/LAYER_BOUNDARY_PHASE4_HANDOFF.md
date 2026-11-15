# Phase 4 handoff prompt

Paste the prompt below into a new agent session in this repository to continue the layer boundary work. Update it when the state it describes changes.

---

You are continuing the layer boundary refactor of Theta-IDE on the branch `implement/layer-boundaries`. The branch exists on `origin` (ThetaIDE/ThetaIDE), but the local branch is ahead of it: the phase 4 commits are not pushed. It has no PR and must not be merged to `main` until phase 4 and the Hub republish are ready together.

**Read first, in this order:**
1. `AGENTS.md` and `docs/README.md`.
2. `docs/plans/LAYER_BOUNDARY_PHASE4_PLAN.md`: the plan you are implementing.
3. `docs/plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md`, especially "Confirmed decisions", which win over anything else.
4. `docs/development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md`: the progress outline and test-failure categories.
5. `docs/reference/LAYER_BOUNDARIES.md` (contract and exception register) and `docs/reference/PLUGIN_STORAGE.md`.

**Where things stand (2026-11-14):**
- **Phases 0–3** are complete locally, with SDK 2.2 (`src.app.api`). The only open items are external: a real cluster smoke run and an interactive check of the installed desktop. Both need the user.
- **Phase 4 units 4a–4g have landed locally.** Every decision is settled: no legacy support of any kind (v3 only, no `in/` handling, every compatibility adapter retired including checkpoint aliases, historical results untouched, no readers or forwarders in phases 5 and 6). Details, commits and evidence are in the status doc's 4g section.
- **4g prepared the Hub releases without publishing them.** Worktrees `../theta-plugins-v3` and `../theta-hub-v3` (branch `v3-republish`) hold the v3 sources and 39 release archives with registry entries and feeds. The user's own `../theta-plugins` and `../theta-hub` checkouts are untouched. Tests that install Hub releases read `THETA_HUB_ROOT` and `THETA_PLUGINS_ROOT`.
- **Next: user decisions, then publication.**
  - Whether to port `ai_assistant` and `theta_mcp` (they need a large SDK service surface), drop them from the feed, or ask their author (ccast44).
  - Phase 6 says `src.app.api` becomes `src.ide.api` "with forwarding", but no forwarders are allowed, so the rename would force another republish of every package. Settle this before phase 6.
  - When the user authorizes publication, follow `docs/development/THETA_HUB_RELEASE_WORKFLOW.md`: push the Hub and plugins branches, verify the client feed, and set `published_at`. Then remove the last exception-register row in `docs/reference/LAYER_BOUNDARIES.md`.

  Do not reintroduce compatibility code, deprecation paths or messages about old formats.

**Then continue** with the user's decisions, committing separately with the attribution trailer the session gives you. Keep the status doc's progress outline and a dated section current as units land. Add new documents to `docs/README.md`.

**Hard constraints:**
- Do not publish to Theta Hub, push to any Hub repository, or merge to `main` unless the user asks. Unit 4g prepares releases only.
- Before moving the repository's own research workspace (unit 4f), show the user the dry-run summary and get a go-ahead. Back it up first, and verify that nothing is left behind.
- Never pick a winner between conflicting files or follow the `mimic` symlink (move it as a link). Leave historical results in place.
- Core must not branch on package IDs, backend names or institution names. Packages import host code only through `src.app.api`.
- Configuration UI changes follow `docs/reference/CONFIG_PARAMETER_FORMS.md`.

**Testing:**
- Run tests per file: `QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest -q -p no:cacheprovider tests/<file>.py`. A whole-suite run in one process can die silently.
- About 49 failures predate this work (research-workspace inputs, tuning and objective suites, `test_parallel_runs`). Before calling a failure a regression, compare it with and without your change: `git stash push -u`, rerun the file, `git stash pop`.
- Optional components (Slurm, Optuna) are not installed in the root research workspace. Tests install them with `tests/component_fixtures.py`.
- For the installed-distribution probe, build with a fresh setuptools build base and follow `scripts/verify_installed_distribution.py`. Do not reuse `/tmp/theta-boundary-baseline/env` without removing its stale `site-packages/src`; it also has Optuna installed.

**When you stop,** report what landed (with commits), what was verified and how, what failed or was skipped, and the next unit. Then update this prompt.
