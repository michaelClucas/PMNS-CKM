import time, torch
t=time.perf_counter(); a=torch.randn(4096,256,device="xpu"); torch.xpu.synchronize(); print("alloc", round(time.perf_counter()-t,3), flush=True)
w=torch.randn(256,256,device="xpu")
for i in range(5):
    t=time.perf_counter(); b=a@w; torch.xpu.synchronize(); print("matmul", i, round(time.perf_counter()-t,4), flush=True)
net=torch.nn.Sequential(torch.nn.Linear(8,256),torch.nn.SiLU(),torch.nn.Linear(256,256),torch.nn.SiLU(),torch.nn.Linear(256,256),torch.nn.SiLU(),torch.nn.Linear(256,8)).to("xpu")
opt=torch.optim.Adam(net.parameters(),1e-3); x=torch.randn(4096,8,device="xpu"); y=torch.randn(4096,8,device="xpu")
for i in range(30):
    t=time.perf_counter(); opt.zero_grad(); l=torch.nn.functional.mse_loss(net(x),y); l.backward(); opt.step(); torch.xpu.synchronize()
    if i<5 or i%5==0: print("step", i, round(time.perf_counter()-t,4), flush=True)
