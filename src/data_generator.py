import numpy as np
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """
    Args:
        n: number of observations
        d: number of features
        snr: Signal-to-Noise Ratio
        sparsity: fraction of non-zero coefficients in W
        missing_rate: Bernouilli parameter for Ho-MCAR hypothesis.
        n_test: number of observations for test dataset.
        random_state: for reproductible results.
    """

    n: int = 1000
    d: int = 100
    rank: int = 5
    snr: float = 5.0
    sparsity: float = 1.0

    missing_rate: float = 0.5
    mechanism: str = "MCAR"

    n_test: int = 500
    random_state: int = 42


def generate_data(cfg: ExperimentConfig):
    """
    Returns:
        (X, y, W, X_test, y_test): data matrix, observation matrix, true weights, test data matrix, test obserations
    """
    rng = np.random.default_rng(cfg.random_state)

    X = rng.normal(loc=0, scale=1, size=(cfg.n, cfg.d))

    W = rng.normal(loc=0, scale=1, size=cfg.d) / np.sqrt(cfg.d)
    if cfg.sparsity < 1.0:
        mask = rng.random(cfg.d) > cfg.sparsity
        W[mask] = 0.0

    signal_var = np.var(X @ W)
    sigma = np.sqrt(signal_var / cfg.snr)

    noise = rng.normal(loc=0, scale=sigma, size=cfg.n)
    y = X @ W + noise

    X_test = rng.normal(0, 1, size=(cfg.n_test, cfg.d))
    y_test = X_test @ W + rng.normal(0, sigma, size=cfg.n_test)

    return X, y, W, X_test, y_test


def generate_data_low_rank(cfg: ExperimentConfig):
    """
    Generates data according to the low-rank model from Ayme et al. (2023).

    Low-rank structure:
        Z ~ N(0, I_r)                    — latent factors, dim r << d
        X = A @ Z + mu                   — inputs, dim d (rank r)
        Y = beta @ Z + eps               — output
          = X @ theta_star + eps         — with theta_star = (A†)^T @ beta

    This matches the paper's DGP exactly (Ex. 3.3 and 3.5).
    Falls back to isotropic DGP if cfg.rank is None.

    Returns:
        (X, y, theta_star, X_test, y_test)
    """
    rng = np.random.default_rng(cfg.random_state)
    r = cfg.rank
    if cfg.d < r:
        raise ValueError(
            f"Low-rank DGP requires d >= r, got d={cfg.d} < r={r}. "
            f"The latent dimension cannot exceed the input dimension."
        )

    Z = rng.normal(0, 1, size=(cfg.n, r))
    Z_test = rng.normal(0, 1, size=(cfg.n_test, r))

    A_raw = rng.normal(0, 1, size=(cfg.d, r))
    A, _ = np.linalg.qr(A_raw)
    A = A[:, :r]
    mu = rng.normal(0, 1, size=cfg.d)

    X = Z @ A.T + mu
    X_test = Z_test @ A.T + mu

    beta = rng.normal(0, 1, size=r) / np.sqrt(r)

    A_pinv = np.linalg.pinv(A)  # (r, d)
    theta_star = A_pinv.T @ beta  # (d,)

    if cfg.sparsity < 1.0:
        mask = rng.random(r) > cfg.sparsity
        beta[mask] = 0.0
        theta_star = A_pinv.T @ beta

    signal_var = np.var(Z @ beta)
    sigma = np.sqrt(signal_var / cfg.snr)

    noise = rng.normal(0, sigma, size=cfg.n)
    noise_test = rng.normal(0, sigma, size=cfg.n_test)

    y = Z @ beta + noise
    y_test = Z_test @ beta + noise_test

    return X, y, theta_star, X_test, y_test


def apply_mcar(X, cfg):
    """
    Given data matrix X and missing rate,
    applies Ho-MCAR on X and returns a copy with masked data aswell as the mask used.
    """
    rng = np.random.default_rng(cfg.random_state + 1)
    mask = rng.random(X.shape) < cfg.missing_rate
    X_missing = X.copy().astype(float)
    X_missing[mask] = np.nan
    return X_missing, mask


def get_d_values(n: int, min_d: int = 6) -> list[int]:
    sqrt_n = int(np.sqrt(n))
    below = list(range(min_d, sqrt_n // 2, max(1, sqrt_n // 8)))
    around = list(range(max(min_d, sqrt_n // 2), sqrt_n * 4, max(1, sqrt_n // 6)))
    above = list(range(sqrt_n * 4, int(n * 0.9), max(1, int(n * 0.05))))
    return sorted(set(below + around + above))
