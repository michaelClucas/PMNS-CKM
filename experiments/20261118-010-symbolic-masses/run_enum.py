"""Enumerate all distinct expressions up to complexity CMAX -> artifacts/exp010/enum_C<CMAX>.npz."""
import sys
import time

import symlib as S

if __name__ == "__main__":
    cmax = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    t = time.time()
    E = S.Enum(cmax, verbose=True)
    S.ART.mkdir(parents=True, exist_ok=True)
    E.save(S.ART / f"enum_C{cmax}.npz")
    print(f"[enum] C<={cmax}: {len(E)} distinct, per level {E.counts}, raw {E.counts_raw}, {time.time() - t:.0f}s",
          flush=True)
