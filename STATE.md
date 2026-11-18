# State

Iteration: 0

## Status
Repo scaffolded. No data, code, or experiments yet.

## Next planned tasks
- Data curator: build `data/` from latest PDG (CKM, masses) and NuFIT (PMNS, both orderings, χ² tables), with `data/SOURCES.md` and a loader in `src/`.
- Second subagent: independently verify every number in `data/` against the sources.
- Set up `.venv`, `requirements.txt`, check PyTorch CPU and XPU availability.
- Theorist: list model families and known relations (baselines) with parameter counts.
- Build the baseline evaluator: given a candidate relation, compute χ², p-value, k, N.

## Current best candidates
None.

## Ruled out
None.

## Open questions
- Common energy scale and RG-running method for comparing quark and lepton sectors.
- Whether PyTorch XPU works on this iGPU and beats CPU.
