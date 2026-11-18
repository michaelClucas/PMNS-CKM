import time, torch
def T(name, f):
    t=time.perf_counter(); r=f(); torch.xpu.synchronize(); print(name, round(time.perf_counter()-t,3), flush=True); return r
s=torch.randn(4096,256,device="xpu")
T("elementwise sq", lambda: s*s)
T("sum", lambda: s.sum())
T("mean", lambda: s.mean())
T("sum 2nd call", lambda: s.sum())
