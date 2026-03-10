"""
MSE Benchmark: test MSE vs. dimension d across all imputation methods.

Sweeps d across the sqrt(n) threshold for multiple n values, measuring
test MSE for each imputer + ASGD pipeline and Ridge (oracle) as baseline.

All imputers use ASGD (SGDRegressor averaged=True) as the downstream
estimator, consistent with the paper's optimization framework.

Results are saved to results/benchmark_mse.csv.
Plots  are saved to results/mse_vs_d.png and results/mse_vs_ratio_per_method.png.
"""

import numpy as np
import pandas as pd
from dataclasses import replace
from pathlib import Path
from sklearn.linear_model import SGDRegressor
from joblib import Parallel, delayed

from src.data_generator import (
    ExperimentConfig,
    generate_data,
    apply_mcar,
    generate_data_low_rank,
    get_d_values,
)
from src.imputer import (
    ZeroImputer,
    MeanImputer,
    KNNImputer_,
    MICEImputer,
    MissForestImputer,
)
from src.equivalence import implicit_lambda, fit_ridge
from src.plots import plot_mse_benchmark

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

N_VALUES = [100, 500]
N_SEEDS = 10
BASE_CFG = ExperimentConfig(snr=5.0, sparsity=1.0, missing_rate=0.3)

IMPUTER_REGISTRY = [
    ("Zero", lambda seed: ZeroImputer()),
    ("Mean", lambda seed: MeanImputer()),
    ("KNN", lambda seed: KNNImputer_(n_neighbors=5)),
    ("MICE", lambda seed: MICEImputer(random_state=seed)),
    # ("MissForest", lambda seed: MissForestImputer(random_state=seed)),
]

METHOD_STYLE = {
    "Zero": {"color": "#E63946", "lw": 2.5, "ls": "-", "zorder": 5},
    "Mean": {"color": "#F4A261", "lw": 1.8, "ls": "--", "zorder": 4},
    "KNN (k=5)": {"color": "#2A9D8F", "lw": 1.8, "ls": "-.", "zorder": 4},
    "MICE": {"color": "#457B9D", "lw": 1.8, "ls": ":", "zorder": 4},
    # "MissForest": {"color": "#6A4C93", "lw": 1.8, "ls": "-", "zorder": 4},
    "Ridge (oracle)": {"color": "#1D3557", "lw": 2.5, "ls": "--", "zorder": 6},
}

PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}


def fit_asgd(X: np.ndarray, y: np.ndarray, random_state: int = 0) -> np.ndarray:
    """
    Fits ASGD (Polyak-Ruppert averaging) on imputed data.
    No explicit regularization — implicit regularization comes from imputation.

    Learning rate follows Proposition 4.3 of Ayme et al. (2023):
        γ = 1 / (d * sqrt(n))
    where d = X.shape[1] and n = X.shape[0], assuming normalized inputs (L²=κ=1).
    """
    n, d = X.shape
    eta0 = 1.0 / (d * np.sqrt(n))

    model = SGDRegressor(
        loss="squared_error",
        penalty=None,
        fit_intercept=False,
        average=True,
        max_iter=1000,
        tol=1e-4,
        learning_rate="constant",
        eta0=eta0,
        random_state=random_state,
    )
    model.fit(X, y)
    return model.coef_


def compute_mse(beta: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    return float(np.mean((y_test - X_test @ beta) ** 2))


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def run_single(cfg: ExperimentConfig) -> dict:
    """
    For one (n, d, seed):
      - Fits each imputer + ASGD pipeline and records test MSE
      - Fits ridge with theoretical λ as oracle baseline
    """
    X, y, theta_star, X_test, y_test = generate_data_low_rank(cfg)
    X_missing, _ = apply_mcar(X, cfg)
    lam = implicit_lambda(cfg.missing_rate)

    record = {
        "n": cfg.n,
        "d": cfg.d,
        "sqrt_n": np.sqrt(cfg.n),
        "d_over_sqrt_n": cfg.d / np.sqrt(cfg.n),
        "seed": cfg.random_state,
    }

    beta_ridge = fit_ridge(X, y, lam)
    record["mse_Ridge (oracle)"] = compute_mse(beta_ridge, X_test, y_test)

    for name, factory in IMPUTER_REGISTRY:
        try:
            X_imputed = factory(cfg.random_state).fit_transform(X_missing)
            beta = fit_asgd(X_imputed, y, random_state=cfg.random_state)
            record[f"mse_{name}"] = compute_mse(beta, X_test, y_test)
        except Exception as e:
            record[f"mse_{name}"] = np.nan
            print(
                f"    '{name}' failed (n={cfg.n}, d={cfg.d}, seed={cfg.random_state}): {e}"
            )

    return record


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------


def run_mse_benchmark() -> pd.DataFrame:
    """
    Runs the full sweep and returns a long-format DataFrame with one row
    per (n, d, imputer), aggregated over seeds.
    Uses joblib for embarrassingly parallel execution across all configs.
    """

    all_configs = [
        replace(BASE_CFG, n=n, d=d, n_test=min(500, n // 2), random_state=seed)
        for n in N_VALUES
        for d in get_d_values(n)
        if d < 2 * n
        for seed in range(N_SEEDS)
    ]
    print(
        f"Total configs: {len(all_configs)} "
        f"({len(N_VALUES)} n-values x ~{len(all_configs) // len(N_VALUES) // N_SEEDS} d-values x {N_SEEDS} seeds)"
    )

    records = Parallel(n_jobs=5, verbose=1)(
        delayed(run_single)(cfg) for cfg in all_configs
    )
    records = [r for r in records if r is not None]

    raw = pd.DataFrame(records)
    mse_cols = [c for c in raw.columns if c.startswith("mse_")]
    group_cols = ["n", "d", "sqrt_n", "d_over_sqrt_n"]

    rows = []
    for keys, grp in raw.groupby(group_cols):
        n, d, sqrt_n, ratio = keys
        for col in mse_cols:
            rows.append(
                {
                    "n": n,
                    "d": d,
                    "sqrt_n": sqrt_n,
                    "d_over_sqrt_n": ratio,
                    "imputer": col[len("mse_") :],
                    "mse_mean": grp[col].mean(),
                    "mse_std": grp[col].std(),
                }
            )

    mse_df = pd.DataFrame(rows)
    mse_df.to_csv(RESULTS_DIR / "benchmark_mse.csv", index=False)
    print(f"\nResults saved → {RESULTS_DIR / 'benchmark_mse.csv'}")
    return mse_df


if __name__ == "__main__":
    print("Running MSE benchmark...")
    print(f"  n values  : {N_VALUES}")
    print(f"  seeds     : {N_SEEDS}")
    print(f"  π (MCAR)  : {BASE_CFG.missing_rate}")
    print(f"  imputers  : {[name for name, _ in IMPUTER_REGISTRY]} + Ridge (oracle)")
    print(f"  estimator : ASGD (SGDRegressor, averaged=True)\n")

    mse_df = run_mse_benchmark()

    print("\n--- MSE sample (n=1000, around d/√n = 1) ---")
    sample = mse_df[(mse_df["n"] == 1000) & mse_df["d_over_sqrt_n"].between(0.8, 1.2)][
        ["imputer", "d", "d_over_sqrt_n", "mse_mean", "mse_std"]
    ].sort_values(["d", "imputer"])
    print(sample.to_string(index=False, float_format="{:.4f}".format))

    print("\nPlotting...")
    plot_mse_benchmark(RESULTS_DIR, PLOT_STYLE, METHOD_STYLE, mse_df)
