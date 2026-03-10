"""
The paper (Ayme et al., 2023) states that the equivalence becomes meaningful
in the regime d >> sqrt(n). This experiment sweeps d from small values all
the way through sqrt(n) and beyond (up to d < 2n) for multiple values of n,
measuring the relative coefficient norm ||β_asgd - β_ridge||_2 / ||β_ridge||_2.

ASGD (SGDRegressor averaged=True) is used as the downstream estimator on
zero-imputed data, consistent with the paper's optimization framework.

Results are saved to results/threshold_experiment.csv.
Plots  are saved to results/threshold_vs_d.png and results/threshold_vs_ratio.png.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor

from dataclasses import replace
from pathlib import Path
from tqdm import tqdm

from src.data_generator import (
    ExperimentConfig,
    generate_data,
    apply_mcar,
    generate_data_low_rank,
    get_d_values,
)
from src.imputer import ZeroImputer
from src.equivalence import implicit_lambda, fit_ridge
from src.plots import plot_threshold_experiment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}

N_VALUES = [100, 500, 1000, 2000]
N_SEEDS = 10
BASE_CFG = ExperimentConfig(snr=5.0, sparsity=1.0, missing_rate=0.5)


def run_single(cfg: ExperimentConfig) -> dict:
    """
    For one (n, d, seed): fits ASGD on zero-imputed data and ridge with
    theoretical λ, then measures the relative coefficient gap.
    """
    X, y, theta_star, X_test, y_test = generate_data_low_rank(cfg)
    X_missing, _ = apply_mcar(X, cfg)
    lam = implicit_lambda(cfg.missing_rate)

    X_zero = ZeroImputer().fit_transform(X_missing)
    n_obs, n_features = X.shape
    eta0 = 1.0 / (n_features * np.sqrt(n_obs))

    model = SGDRegressor(
        loss="squared_error",
        penalty=None,
        fit_intercept=False,
        average=True,
        max_iter=1000,
        tol=1e-4,
        learning_rate="constant",
        eta0=eta0,
        random_state=cfg.random_state,
    )
    model.fit(X_zero, y)
    beta_naive = model.coef_

    beta_ridge = fit_ridge(X, y, lam)

    abs_norm = float(np.linalg.norm(beta_naive - beta_ridge))
    ridge_norm = float(np.linalg.norm(beta_ridge))
    relative_norm = abs_norm / ridge_norm if ridge_norm > 1e-10 else np.nan

    return {
        "n": cfg.n,
        "d": cfg.d,
        "sqrt_n": np.sqrt(cfg.n),
        "d_over_sqrt_n": cfg.d / np.sqrt(cfg.n),
        "missing_rate": cfg.missing_rate,
        "lambda_theory": lam,
        "abs_norm": abs_norm,
        "relative_norm": relative_norm,
        "seed": cfg.random_state,
    }


def run_threshold_experiment() -> pd.DataFrame:
    """
    Runs the full sweep and returns aggregated equivalence metrics.
    """
    records = []

    for n in N_VALUES:
        d_values = get_d_values(n)
        print(
            f"\nn={n:>5d} | √n={np.sqrt(n):.1f} | "
            f"{len(d_values)} d-values from {d_values[0]} to {d_values[-1]}"
        )

        for d in tqdm(d_values, desc=f"  n={n}"):
            if d >= 2 * n:
                continue
            for seed in range(N_SEEDS):
                cfg = replace(
                    BASE_CFG,
                    n=n,
                    d=d,
                    n_test=min(500, n // 2),
                    random_state=seed,
                )
                try:
                    records.append(run_single(cfg))
                except Exception as e:
                    print(f"    run_single failed (n={n}, d={d}, seed={seed}): {e}")

    raw = pd.DataFrame(records)

    equiv_agg = (
        raw.groupby(["n", "d", "sqrt_n", "d_over_sqrt_n"])
        .agg(
            relative_norm_mean=("relative_norm", "mean"),
            relative_norm_std=("relative_norm", "std"),
            abs_norm_mean=("abs_norm", "mean"),
        )
        .reset_index()
    )
    equiv_agg.to_csv(RESULTS_DIR / "threshold_experiment.csv", index=False)
    print(f"\nEquivalence results → {RESULTS_DIR / 'threshold_experiment.csv'}")
    return equiv_agg


if __name__ == "__main__":
    print("Running threshold experiment...")
    print(f"  n values:  {N_VALUES}")
    print(f"  seeds:     {N_SEEDS}")
    print(f"  π (MCAR):  {BASE_CFG.missing_rate}")

    df = run_threshold_experiment()

    print("\nSample results (mean relative norm at key d/√n ratios):")
    sample = df[df["d_over_sqrt_n"].between(0.8, 1.2)][
        ["n", "d", "d_over_sqrt_n", "relative_norm_mean", "relative_norm_std"]
    ].sort_values(["n", "d_over_sqrt_n"])
    print(sample.to_string(index=False, float_format="{:.4f}".format))

    print("\nPlotting...")
    plot_threshold_experiment(RESULTS_DIR, PLOT_STYLE, df)
