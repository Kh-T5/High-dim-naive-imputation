"""
Equivalence verification: naive zero imputation + OLS  ≈  ridge on complete data.

The paper (Ayme et al., 2023) shows that under MCAR with missing rate π,
the OLS estimator on mean-imputed data converges to the ridge estimator with:

    λ* = π / (1 - π)

when features are standardized (zero mean, unit variance).

This module verifies this equivalence numerically across:
    - varying sample sizes n  (asymptotic convergence check)
    - varying missing rates π
    - varying dimensions d
"""

import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from dataclasses import dataclass, replace

from src.data_generator import ExperimentConfig, generate_data, apply_mcar
from src.imputer import ZeroImputer


def implicit_lambda(missing_rate: float) -> float:
    """
    Returns the implicit ridge regularization parameter equivalent to
    naive zero imputation under MCAR, as derived in Ayme et al. (2023).

        λ* = π / (1 - π)

    Args:
        missing_rate: π, the MCAR missing rate in (0, 1).
    """
    if not 0 < missing_rate < 1:
        raise ValueError("missing_rate must be in (0, 1).")
    return missing_rate / (1 - missing_rate)


def fit_naive_imputation(X_missing: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS on zero-imputed data. Returns coefficient vector."""
    X_imputed = ZeroImputer().fit_transform(X_missing)
    return LinearRegression(fit_intercept=False).fit(X_imputed, y).coef_


def fit_ridge(X_complete: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge regression with explicit λ on complete data. Returns coefficient vector."""
    return Ridge(alpha=lam, fit_intercept=False).fit(X_complete, y).coef_


# ---------------------------------------------------------------------------
# Single equivalence run
# ---------------------------------------------------------------------------


@dataclass
class EquivalenceResult:
    n: int
    d: int
    missing_rate: float
    lambda_theory: float
    coef_diff_norm: float
    coef_diff_norm_relative: float
    coef_naive: np.ndarray
    coef_ridge: np.ndarray


def run_equivalence(cfg: ExperimentConfig) -> EquivalenceResult:
    """
    For a single config, fits both estimators and measures how close
    their coefficients are.
    """
    X, y, W, X_test, y_test = generate_data(cfg)
    X_missing, _ = apply_mcar(X, cfg)

    lam = implicit_lambda(cfg.missing_rate)

    beta_naive = fit_naive_imputation(X_missing, y)
    beta_ridge = fit_ridge(X, y, lam)

    diff_norm = float(np.linalg.norm(beta_naive - beta_ridge))
    relative_diff = diff_norm / np.linalg.norm(beta_ridge)

    return EquivalenceResult(
        n=cfg.n,
        d=cfg.d,
        missing_rate=cfg.missing_rate,
        lambda_theory=lam,
        coef_diff_norm=diff_norm,
        coef_diff_norm_relative=relative_diff,
        coef_naive=beta_naive,
        coef_ridge=beta_ridge,
    )


def sweep_sample_size(
    base_cfg: ExperimentConfig,
    n_values: list[int],
) -> list[EquivalenceResult]:
    """
    Sweeps over sample sizes to verify asymptotic convergence as n grows.
    """
    results = []
    for n in n_values:
        cfg = replace(base_cfg, n=n)
        results.append(run_equivalence(cfg))
    return results


def sweep_missing_rate(
    base_cfg: ExperimentConfig,
    pi_values: list[float],
) -> list[EquivalenceResult]:
    """
    Sweeps over missing rates π.
    At each π, verifies that the closed-form λ* = π/(1-π) yields
    a ridge estimator close to naive imputation.
    """
    results = []
    for pi in pi_values:
        cfg = replace(base_cfg, missing_rate=pi)
        results.append(run_equivalence(cfg))
    return results


def sweep_dimension(
    base_cfg: ExperimentConfig,
    d_values: list[int],
) -> list[EquivalenceResult]:
    """
    Sweeps over feature dimensions d.
    Tests whether the equivalence holds as d/n ratio grows.
    """
    results = []
    for d in d_values:
        cfg = replace(base_cfg, d=d)
        results.append(run_equivalence(cfg))
    return results
