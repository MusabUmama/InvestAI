from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OptimizationResult:
    weights: np.ndarray  # shape (n,)
    sharpe: float
    mu_p: float
    sigma_p: float
    chosen_lambda: float


def _project_capped_simplex(v: np.ndarray, *, z: float, upper: float, iters: int = 40) -> np.ndarray:
    """
    Euclidean projection onto { w | sum(w)=z, 0<=w<=upper } using bisection on the Lagrange multiplier.
    KKT gives w_i = clip(v_i - lambda, 0, upper) with sum constraint.
    """
    n = v.size
    if upper * n + 1e-12 < z:
        raise ValueError(f"Infeasible: upper*n={upper*n} < z={z}")

    lo = float(np.min(v) - upper)  # tends to push weights up
    hi = float(np.max(v))          # tends to push weights down

    for _ in range(iters):
        mid = (lo + hi) / 2.0
        w = np.clip(v - mid, 0.0, upper)
        s = float(w.sum())
        if s > z:
            lo = mid
        else:
            hi = mid

    w = np.clip(v - hi, 0.0, upper)
    s = float(w.sum())
    if s <= 0:
        # Fallback: put everything into the best coordinate if projection failed numerically.
        w = np.zeros_like(v)
        w[int(np.argmax(v))] = z
        w = np.clip(w, 0.0, upper)
        w = w / w.sum()
        return w
    return w * (z / s)


def _portfolio_stats(mu: np.ndarray, sigma: np.ndarray, w: np.ndarray, rf: float) -> tuple[float, float, float]:
    mu_p = float(w @ mu)
    var_p = float(w @ sigma @ w)
    sigma_p = float(np.sqrt(max(var_p, 1e-18)))
    sharpe = float((mu_p - rf) / sigma_p) if sigma_p > 0 else -np.inf
    return sharpe, mu_p, sigma_p


def maximize_sharpe_via_mean_variance_sweep(
    *,
    mu: np.ndarray,
    sigma: np.ndarray,
    rf: float = 0.0,
    max_weight: float = 0.35,
    lambdas: np.ndarray | None = None,
    iters: int = 100,
    tol: float = 1e-6,
) -> OptimizationResult:
    """
    Practical approach for constrained Sharpe: solve a grid of mean-variance problems
    max_w (mu^T w - lambda * w^T Sigma w) under capped-simplex constraints, pick best Sharpe.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = mu.size
    if sigma.shape != (n, n):
        raise ValueError("sigma must be square (n,n)")

    if lambdas is None:
        # A tighter default grid keeps the API responsive while preserving
        # enough breadth to find a strong constrained Sharpe solution.
        lambdas = np.logspace(-2, 2, 11)

    # Start from equal weights, projected to bounds.
    w0 = np.full(n, 1.0 / n, dtype=float)
    w0 = _project_capped_simplex(w0, z=1.0, upper=max_weight)

    best = OptimizationResult(weights=w0, sharpe=-np.inf, mu_p=0.0, sigma_p=0.0, chosen_lambda=float(lambdas[0]))

    # Precompute eigen max for step sizing (smooth part).
    eig_max = float(np.linalg.eigvalsh(sigma).max())
    eig_max = max(eig_max, 1e-12)

    w_start = w0.copy()
    for lam in lambdas:
        lam = float(lam)
        w = w_start.copy()
        # Lipschitz for grad of (lam * w^T Sigma w) is 2*lam*eig_max
        step = 1.0 / (2.0 * lam * eig_max + 1.0) if lam > 0 else 0.1

        for _ in range(iters):
            grad = -mu + 2.0 * lam * (sigma @ w)
            w_new = _project_capped_simplex(w - step * grad, z=1.0, upper=max_weight)
            if float(np.max(np.abs(w_new - w))) < tol:
                w = w_new
                break
            w = w_new

        w_start = w

        sharpe, mu_p, sigma_p = _portfolio_stats(mu, sigma, w, rf)
        if sharpe > best.sharpe:
            best = OptimizationResult(weights=w, sharpe=sharpe, mu_p=mu_p, sigma_p=sigma_p, chosen_lambda=lam)

    return best
