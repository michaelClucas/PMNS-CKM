"""SU(5) control: PMNS vs quark-sector dependence in the su5_min / su5_gj prior datasets (KSG mutual information,
20k samples). In su5_min U_e = 1 exactly in the Y_u-diagonal basis, so PMNS = U_nu is independent of the quark data;
in su5_gj U_e deviates through the 45_H entry. Also the m_e/m_d-type predictions (lg_me_mmu, lg_mmu_mtau, lg_mtau_mb)."""
import json, pathlib, sys
import numpy as np
from sklearn.feature_selection import mutual_info_regression
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import gen
ART = HERE.parents[1] / "artifacts" / "f3"
ref = gen.gut_reference()
out = {}
for v in ("su5_min", "su5_gj"):
    d = np.load(ART / f"ds_broad_{v}.npz")
    z = d["z"][:20000].astype(float)
    r = {}
    for k, n in zip(range(12, 15), ("s12sq_l", "s13sq_l", "s23sq_l")):
        mi = mutual_info_regression(z[:, [8, 9, 10]], z[:, k], random_state=0)
        r[n] = dict(zip(["s12q", "lg_s13q", "lg_s23q"], np.round(mi, 4).tolist()))
    r["lepton_mass_quantiles_5_50_95"] = {gen.Z_NAMES[j]: np.round(np.quantile(z[:, j], [0.05, .5, .95]), 3).tolist()
                                          for j in (5, 6, 7)}
    r["observed"] = {gen.Z_NAMES[j]: round(ref["z"][j], 3) for j in (5, 6, 7)}
    out[v] = r
(HERE / "su5_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
