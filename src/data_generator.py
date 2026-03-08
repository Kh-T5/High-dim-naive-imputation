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
    snr: float = 5.0
    sparsity: float = 1.0

    missing_rate: float = 0.3
    mechanism: str = "MCAR"

    n_test: int = 500
    random_state: int = 42


def generate_data(cfg: ExperimentConfig):
    """
    Returns:
        (X, y, W): data matrix, observation matrix, true weights
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

    return X, y, W


def apply_mcar(X, cfg):
    """
    Given data matrix X and missing rate,
    applies Ho-MCAR on X and returns a copy with masked data aswell as the mask used.
    """
    rng = np.random.default_rng(cfg.random_state)
    mask = rng.random(X.shape) < cfg.missing_rate
    X_missing = X.copy().astype(float)
    X_missing[mask] = np.nan
    return X_missing, mask
