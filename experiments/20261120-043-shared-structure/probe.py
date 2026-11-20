import time, numpy as np, sys
sys.path.insert(0, ".")
from src import yukawa, evaluate as ev, data
rng = np.random.default_rng(0)
Y = [rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)) for _ in range(4)]
m = Y[3]+Y[3].T
t=time.time()
for _ in range(2000): o = yukawa.observables(Y[0],Y[1],Y[2],m)
print("obs us", (time.time()-t)/2000*1e6)
USE = ("ckm_s12","ckm_s13","ckm_s23","ckm_delta_rad","s12sq","s13sq","s23sq","dcp_deg","mr_u_c","mr_c_t","mr_d_s","mr_s_b","mr_e_mu","mr_mu_tau","dm_ratio")
sc = ev.Scorer(USE, "NO", mass_floor=0.02)
best = ev.pmns_best("NO"); print(best)
pred = dict(o); pred.update(best)
vals, names, cov = ev.mass_ratio_targets(); print(vals, names)
for k in names: pred[k]=vals[k]
ck = data.load_ckm(); print({k: ck[k] for k in ("s12","s13","s23","delta_rad")})
pred.update({"ckm_s12":ck["s12"]["value"],"ckm_s13":ck["s13"]["value"],"ckm_s23":ck["s23"]["value"],"ckm_delta_rad":ck["delta_rad"]["value"],"dm_ratio":0.0295})
t=time.time()
for _ in range(500): c=sc.chi2(pred)
print("chi2 us", (time.time()-t)/500*1e6, c)
r = sc(pred); print(r.pulls)
pm = data.load_pmns("NO"); print({k: pm[k] for k in ("s12sq","s13sq","s23sq","dcp_deg")})
print(ev._dm_best("NO", data.DEFAULT_ANALYSIS), ev._dm_sigma("NO", data.DEFAULT_ANALYSIS))
