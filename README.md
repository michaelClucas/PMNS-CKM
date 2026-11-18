# PMNS-CKM

An autonomous research program looking for a quantitative relation between the CKM (quark) and PMNS (lepton) mixing matrices. Neural networks are trained on synthetic theory-space data and used as search instruments, with measured values from PDG and NuFIT as the target.

Run by a Claude Code overseer agent that dispatches teams of research subagents in a loop. See `OVERSEER_PROMPT.md` for the program and its success criteria, `STATE.md` for current status, and `CANDIDATES.md` for the leaderboard.

Everything here is a candidate unless `BREAKTHROUGH.md` exists.

## Layout
- `data/`: measured inputs with sources
- `src/`: shared code (data loader, evaluators, models)
- `experiments/`: one folder per experiment, including negative results
