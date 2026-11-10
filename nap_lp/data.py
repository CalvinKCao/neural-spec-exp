"""Data prep + training. Trained weights/splits are saved to artifacts/ so that the local
machine and the cluster use bit-identical networks (no retraining, no torch needed there)."""
import os
import sys
import numpy as np

from .net import Net

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
HIDDEN = [32, 32, 32, 32]
EPOCHS, LR, BS = 200, 1e-3, 32


def make_split(seed):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    d = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.2, stratify=d.target, random_state=seed)
    sc = MinMaxScaler().fit(Xtr)
    return sc.transform(Xtr), np.clip(sc.transform(Xte), 0, 1), ytr, yte


def train(seed):
    import torch
    torch.set_num_threads(1)
    Xtr, Xte, ytr, yte = make_split(seed)
    torch.manual_seed(seed)
    dims = [Xtr.shape[1]] + HIDDEN + [2]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.ReLU())
    m = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), BS):
            idx = perm[i:i + BS]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(m(Xt[idx]), yt[idx]).backward()
            opt.step()
    lin = [l for l in m if isinstance(l, torch.nn.Linear)]
    Ws = [l.weight.detach().double().numpy() for l in lin]
    bs = [l.bias.detach().double().numpy() for l in lin]
    net = Net(Ws, bs)
    acc_tr = float((net.predict(Xtr) == ytr).mean())
    acc_te = float((net.predict(Xte) == yte).mean())
    os.makedirs(ART, exist_ok=True)
    np.savez(os.path.join(ART, f"model_seed{seed}.npz"),
             **{f"W{i}": w for i, w in enumerate(Ws)}, **{f"b{i}": b for i, b in enumerate(bs)},
             Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, acc_tr=acc_tr, acc_te=acc_te)
    return net


def load(seed):
    """Returns (net, Xtr, ytr, Xte, yte, test_acc). Trains (needs torch) if the artifact is missing."""
    p = os.path.join(ART, f"model_seed{seed}.npz")
    if not os.path.exists(p):
        train(seed)
    d = np.load(p)
    n = len([k for k in d.files if k.startswith("W")])
    net = Net([d[f"W{i}"] for i in range(n)], [d[f"b{i}"] for i in range(n)])
    return net, d["Xtr"], d["ytr"], d["Xte"], d["yte"], float(d["acc_te"])


if __name__ == "__main__":
    for s in [int(a) for a in sys.argv[1:]] or range(5):
        train(s)
        net, Xtr, ytr, Xte, yte, acc = load(s)
        print(f"seed {s}: train acc {(net.predict(Xtr) == ytr).mean():.4f}  test acc {acc:.4f}  "
              f"(n_train={len(ytr)}, n_test={len(yte)}, test per class={np.bincount(yte).tolist()})")
