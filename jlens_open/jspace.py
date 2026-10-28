"""J-space (Gurnee et al. 2026): sparse nonnegative combinations of J-lens vectors.

The J-lens vectors at layer l are the rows of W_U diag(g) J_l, one per vocabulary token (g: the final norm's
gain). The paper writes an activation as a nonnegative combination of at most k of them (k <= 25, found by
gradient pursuit). Here a greedy nonnegative orthogonal matching pursuit does the same job: at each step it
adds the unit atom most aligned with the residual, then refits all coefficients by nonnegative least squares.
"""
import torch


@torch.no_grad()
def lens_atoms(lm, layer, use_jacobian=True, center=True, chunk=16384):
    """Unit-norm J-lens vectors (vocab, d) at `layer` in float16 (with use_jacobian=False: logit-lens vectors).

    center=True subtracts the mean over the vocabulary first. Adding one vector to every row of W_U leaves the
    softmax unchanged, so that shared component carries no readout; uncentered, it dominates (in GPT-2 small
    the unit J-lens vectors have cosine ~0.99 with their mean) and the atoms fill a narrow cone."""
    W, g = lm.model.lm_head.weight, lm.norm.weight.float().to(lm.device)
    mean_row = sum(W[s : s + chunk].to(lm.device).float().sum(0) for s in range(0, W.shape[0], chunk)) * g / W.shape[0]
    out = torch.empty(W.shape, dtype=torch.float16, device=lm.device)
    for s in range(0, W.shape[0], chunk):
        w = W[s : s + chunk].to(lm.device).float() * g
        if center:
            w = w - mean_row
        if use_jacobian and layer in lm.J:
            w = w @ lm.J[layer]
        out[s : s + chunk] = (w / w.norm(dim=1, keepdim=True)).half()
    return out


@torch.no_grad()
def random_atoms(n_atoms, d, device, seed=0, cov_sqrt=None, chunk=16384):
    """Unit-norm random atoms (n_atoms, d) in float16: isotropic, or shaped by cov_sqrt (d, d)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = torch.empty(n_atoms, d, dtype=torch.float16, device=device)
    for s in range(0, n_atoms, chunk):
        z = torch.randn(min(chunk, n_atoms - s), d, generator=gen).to(device)
        if cov_sqrt is not None:
            z = z @ cov_sqrt
        out[s : s + chunk] = (z / z.norm(dim=1, keepdim=True)).half()
    return out


def _nnls(G, b, c, n_iter=150):
    """Projected gradient for min_c>=0 |A^T c - x|^2 given G = A A^T (n, s, s), b = A x (n, s)."""
    step = 1.0 / G.abs().sum(-1).max(-1).values.clamp(min=1e-6)  # 1 / Gershgorin bound on the top eigenvalue
    for _ in range(n_iter):
        c = (c - step[:, None] * ((G @ c[..., None]).squeeze(-1) - b)).clamp(min=0)
    return c


@torch.no_grad()
def decompose(X, atoms, k, chunk=1024):
    """Nonnegative matching pursuit of the rows of X (n, d) on unit atoms (V, d), k steps.

    Returns idx (n, k) atom indices, coef (n, k) nonnegative coefficients, and the squared residual norm
    after each step (n, k)."""
    out_idx, out_coef, out_res = [], [], []
    for s in range(0, len(X), chunk):
        x = X[s : s + chunk].float()
        n = len(x)
        R = x.clone()
        idx = torch.empty(n, 0, dtype=torch.long, device=x.device)
        coef = torch.empty(n, 0, device=x.device)
        res = []
        for _ in range(k):
            scores = (R / R.norm(dim=1, keepdim=True).clamp(min=1e-6)).half() @ atoms.T
            if idx.shape[1]:
                scores.scatter_(1, idx, float("-inf"))
            idx = torch.cat([idx, scores.argmax(1, keepdim=True)], 1)
            A = atoms[idx].float()  # (n, s, d)
            G = A @ A.transpose(1, 2)
            b = (A @ x[..., None]).squeeze(-1)
            coef = _nnls(G, b, torch.cat([coef, torch.zeros(n, 1, device=x.device)], 1))
            R = x - (coef[:, None, :] @ A).squeeze(1)
            res.append((R**2).sum(1))
        out_idx.append(idx)
        out_coef.append(coef)
        out_res.append(torch.stack(res, 1))
    return torch.cat(out_idx), torch.cat(out_coef), torch.cat(out_res)


def component(idx, coef, atoms):
    """The reconstruction sum_i coef_i atom_i (n, d) in float32."""
    return (coef[:, None, :] @ atoms[idx].float()).squeeze(1)
