# Repo rules (overseer and all subagents)

Research program: find a quantitative relation between the CKM and PMNS mixing matrices, using neural networks as the search instrument and measured data as the target. The full program, breakthrough criteria, and team structure are in `OVERSEER_PROMPT.md`. Read it before doing anything.

- `STATE.md` is the current state of the program. Read it first. Only the overseer edits it, `CANDIDATES.md`, and `BREAKTHROUGH.md`.
- Each experiment lives in `experiments/<YYYYMMDD>-<NNN>-<slug>/` with a `README.md` (hypothesis, method, free-parameter count, result, verdict) and the scripts or config needed to rerun it.
- Never train a network directly on the ~dozen measured numbers and call the fit a result. Networks learn from synthetic theory-space data, and measured data is the target or constraint.
- Count free parameters honestly: fitted constants, data-selected discrete choices, and unfixed coefficients all count.
- All measured values come from `data/` through the loader in `src/`. Don't hardcode measured numbers in experiment scripts.
- Python lives in `.venv/`. Add dependencies to `requirements.txt` with pinned versions.
- 14 CPU cores, Intel iGPU, no CUDA. Long jobs run in the background and log to `experiments/<id>/run.log`.
- Repo is public. No personal info, credentials, or absolute local paths in committed files. Don't change the git identity, don't force-push.
- Use "candidate" for every finding unless `BREAKTHROUGH.md` exists.
