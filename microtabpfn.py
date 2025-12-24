"""
microTabPFN: In-context learning for tabular data (~100 lines of core code).

Key ideas:
- Train on synthetic tasks (SCM + BNN prior)
- Inference via in-context learning (no gradients, just attention)
- Two-axis attention: over features (column) AND samples (row)
- Permutation invariant: order of features/samples doesn't matter

Limitations vs full TabPFN:
- Binary classification only (TabPFN: multi-class)
- Fixed feature count (TabPFN: variable)
- Simple prior (TabPFN: complex SCM/BNN/GP mixture)
- No preprocessing (TabPFN: handles missing values, categoricals)
- Single model (TabPFN: ensemble)

Usage: uv run python microtabpfn.py
"""

import random, torch, torch.nn as nn, torch.nn.functional as F

# Device selection

def get_device():
    """Use GPU if available."""
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# Prior: synthetic classification tasks

def sample_scm(n_samples, n_feat, device):
    """SCM: features have causal dependencies (X_i depends on X_<i).
    
    Example: X = sample_scm(100, 4, 'cpu')  # 100 samples, 4 features
             X.shape -> (100, 4) where X[:,i] depends on X[:,:i]
    """
    X = torch.zeros(n_samples, n_feat, device=device)
    X[:, 0] = torch.randn(n_samples, device=device)
    for i in range(1, n_feat):
        # X[:,i] = w_0*X_0 + w_1*X_1 + ... (weighted sum of previous features) + noise
        X[:, i] = X[:, :i] @ torch.randn(i, device=device) / i**0.5 + torch.randn(n_samples, device=device) * 0.5
    return X

def sample_bnn(X):
    """BNN: random neural net defines decision boundary.
    
    Example: y = sample_bnn(X)  # X is (n_samples, n_feat)
             y.shape -> (n_samples,) with values 0 or 1
    """
    d, h, device = X.shape[1], 16, X.device
    W1, b1 = torch.randn(d, h, device=device) / d**0.5, torch.randn(h, device=device) * 0.5
    W2, b2 = torch.randn(h, 1, device=device) / h**0.5, torch.randn(1, device=device) * 0.5
    # y = (tanh(X @ W1 + b1) @ W2 + b2 > 0) — two-layer net with random weights
    return (torch.tanh(X @ W1 + b1) @ W2 + b2 > 0).long().squeeze()

def sample_task(n_train=50, n_test=20, n_feat=4, device=DEVICE):
    """Sample task: X from SCM, y from BNN."""
    X = sample_scm(n_train + n_test, n_feat, device)
    y = sample_bnn(X)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]

# Model: two-axis attention transformer

class Block(nn.Module):
    """Transformer block with attention over BOTH features and samples.
    
    Three attention patterns:
    1. attn_col: features talk to features (learn "petal_length correlates with petal_width")
    2. attn_row(train, train): train samples share info (build class clusters)  
    3. attn_row(test, train): test reads train (in-context learning)
    """
    
    def __init__(self, d, h):
        super().__init__()
        self.attn_col = nn.MultiheadAttention(d, h, batch_first=True)  # over features
        self.attn_row = nn.MultiheadAttention(d, h, batch_first=True)  # over samples
        self.ffn = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
    
    def forward(self, x, n_train):
        # Column attention: features interact (permutation equivariant)
        a, _ = self.attn_col(x, x, x)
        x = self.ln1(x + a)
        
        # Row attention: samples interact
        # Key insight: test ONLY attends to train (in-context learning!)
        xt = x.transpose(0, 1)
        tr, te = xt[:, :n_train], xt[:, n_train:]
        a_tr, _ = self.attn_row(tr, tr, tr)   # train self-attention
        a_te, _ = self.attn_row(te, tr, tr)   # test → train cross-attention
        x = self.ln2(torch.cat([tr + a_tr, te + a_te], 1).transpose(0, 1))
        
        return self.ln3(x + self.ffn(x))


class MicroTabPFN(nn.Module):
    """TabPFN in ~50 lines: learns to classify from context."""
    
    def __init__(self, n_feat=4, d=64, h=4, n_layers=3):
        super().__init__()
        self.n_feat = n_feat
        self.emb_x = nn.Linear(1, d)         # same for all features → permutation invariant
        self.emb_y = nn.Linear(1, d)         # 0, 1, or 0.5 for test (unknown)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
    
    def forward(self, x_train, y_train, x_test):
        n_tr, n_te = len(x_train), len(x_test)
        device = x_train.device
        
        # Embed: each feature independently (no positional encoding → invariant)
        x = self.emb_x(torch.cat([x_train, x_test]).unsqueeze(-1))
        y_vals = torch.cat([y_train.float(), torch.full((n_te,), 0.5, device=device)])
        y = self.emb_y(y_vals.unsqueeze(-1)).unsqueeze(1)
        h = torch.cat([x, y], dim=1)  # shape: (n_samples, n_features+1, d_model)
        
        for block in self.blocks:
            h = block(h, n_tr)
        
        return self.head(h[n_tr:, -1])  # predict from test label positions

# Training

def train(model, steps=2500, lr=1e-3, device=DEVICE):
    """Pre-train on synthetic prior. Note: NO training at inference time!
    
    Args:
        steps: Training steps. 2500 is fast (~15s), 5000 gives better results (~30s).
    """
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    print(f"Training on {device} for {steps} steps...")
    
    for step in range(steps):
        # Sample larger tasks for more diverse training
        n_train = random.randint(30, 100)
        n_test = random.randint(10, 50)
        x_tr, y_tr, x_te, y_te = sample_task(n_train, n_test, model.n_feat, device)
        
        loss = F.cross_entropy(model(x_tr, y_tr, x_te), y_te)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0: print(f"step {step:5d} | loss {loss.item():.4f}")
    
    return model

# Sklearn-compatible classifier

class MicroTabPFNClassifier:    
    def __init__(self, model, device=DEVICE):
        self.model = model.eval()
        self.device = device
    
    def fit(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.long, device=self.device)
        self.mean, self.std = self.X.mean(0), self.X.std(0) + 1e-8
        self.X = (self.X - self.mean) / self.std
        return self
    
    def predict_proba(self, X):
        # i know it's not probabilities, but whatever
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        X = (X - self.mean) / self.std
        with torch.no_grad(): 
            return F.softmax(self.model(self.X, self.y, X), -1).cpu().numpy()
    
    def predict(self, X): return self.predict_proba(X).argmax(1)

# Demo: evaluate on Iris

if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    print("="*55 + "\n microTabPFN: In-context learning for tabular data\n" + "="*55)
    print(f"Device: {DEVICE}")
    
    # Binary Iris: versicolor vs virginica (the hard pair)
    iris = load_iris()
    X, y = iris.data[iris.target != 0], iris.target[iris.target != 0] - 1
    
    print("\n[Training on synthetic data...]")
    torch.manual_seed(42)
    model = train(MicroTabPFN(), steps=5000)
    
    print("\n[Evaluating on Iris (no training, just inference!)...]")
    results = {n: [] for n in ["microTabPFN", "LogReg", "RF", "KNN", "Tree"]}
    
    for seed in range(5):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=seed)
        clf = MicroTabPFNClassifier(model); clf.fit(X_tr, y_tr)
        results["microTabPFN"].append(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
        for name, C in [("LogReg", LogisticRegression), ("RF", RandomForestClassifier),
                        ("KNN", KNeighborsClassifier), ("Tree", DecisionTreeClassifier)]:
            results[name].append(roc_auc_score(y_te, C().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]))
    
    print(f"\n{'Method':<15} {'ROC AUC':>8} {'Std':>8}\n" + "-"*33)
    for n, s in sorted(results.items(), key=lambda x: -np.mean(x[1])):
        print(f"{n:<15} {np.mean(s):>8.3f} {np.std(s):>8.3f}")
    print("\n" + "="*55 + "\n The key: model learns from CONTEXT, not gradients!\n" + "="*55)
