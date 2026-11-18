import time, torch
def T(name, f):
    t=time.perf_counter(); r=f(); torch.xpu.synchronize(); print(name, round(time.perf_counter()-t,3), flush=True); return r
lin=torch.nn.Linear(8,256).to("xpu"); x=torch.randn(4096,8,device="xpu")
h=T("linear fwd", lambda: lin(x))
s=T("silu fwd", lambda: torch.nn.functional.silu(h))
l=T("mse", lambda: torch.nn.functional.mse_loss(s, torch.zeros_like(s)))
T("backward", lambda: l.backward())
opt=torch.optim.Adam(lin.parameters(),1e-3, foreach=False)
T("adam step (foreach=False)", lambda: opt.step())
opt2=torch.optim.Adam(lin.parameters(),1e-3)
T("adam step (default)", lambda: opt2.step())
T("adam step (default) 2", lambda: opt2.step())
