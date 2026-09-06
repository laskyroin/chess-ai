import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from chessai.model import ChessNet
from pathlib import Path


X = np.load("data/full/X.npy", mmap_mode="r")
Y = np.load("data/full/policy.npy")

X = torch.from_numpy(X)
Y = torch.from_numpy(Y).long()


N = len(X)
n1 = int(0.8*N)
n2=int(0.9*N)


g = torch.Generator().manual_seed(44)
perm = torch.randperm(N, generator=g)
tr, dev, te = perm[:n1], perm[n1:n2], perm[n2:]

Xtr, Ytr = X[tr], Y[tr]
Xdev, Ydev  = X[dev], Y[dev]
Xte, Yte = X[te], Y[te]
V = torch.from_numpy(np.load("data/full/value.npy")).float()
Vtr = V[tr]
del X, Y


device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChessNet().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
ckpt = torch.load("runs/ckpt_28000.pt", map_location=device)
model.load_state_dict(ckpt["model"])
opt.load_state_dict(ckpt["opt"])
for pg in opt.param_groups:
    pg["lr"] = 1e-4
start = ckpt["step"]
model.train()
max_steps = 5000
batch_size = 512
lossi= []

Path("runs").mkdir(exist_ok=True)
for i in range(start,start + max_steps):

    #batching

    ix = torch.randint(0, Xtr.shape[0], (batch_size, ))
    Xb, Yb, Vb = Xtr[ix].float().to(device), Ytr[ix].to(device), Vtr[ix].to(device)

    #forward pass
    x, v = model(Xb)

    loss = F.cross_entropy(x, Yb) + F.mse_loss(v.squeeze(-1), Vb)

    #backward pass 
    #for m in model.modules():
    #    if hasattr(m, "out"):
     #       m.out.retain_grad()
    opt.zero_grad(set_to_none=True)
    loss.backward()
   
    #update
    opt.step()


    #tracking stats
    lossi.append(loss.item())
    if i % 500 == 0:
        model.eval()
        with torch.no_grad():
            jx = torch.randint(0, Xdev.shape[0], (2048,))
            ld, _ = model(Xdev[jx].float().to(device))
            dl = F.cross_entropy(ld, Ydev[jx].to(device)).item()
        model.train()
        print(f"{i}  train {loss.item():.3f}  dev {dl:.3f}")

    if i % 1000 == 0 and i > 0:
        torch.save({"model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "step": i}, f"runs/ckpt_{i}.pt")

torch.save({"model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": i}, f"runs/ckpt_{i}.pt")