from src.equivalence import (
    ExperimentConfig,
    sweep_dimension,
    sweep_missing_rate,
    sweep_sample_size,
)

if __name__ == "__main__":
    base_cfg = ExperimentConfig(n=500, d=50, missing_rate=0.5, random_state=42)

    print("=== Asymptotic convergence (increasing n) ===")
    for res in sweep_sample_size(base_cfg, [100, 500, 1000, 2000, 5000, 10000]):
        print(
            f"  n={res.n:>6d} | λ*={res.lambda_theory:.4f} | "
            f"||β_asgd - β_ridge||={res.coef_diff_norm:.6f} | "
            f"relative={res.relative_norm:.6f}"
        )

    print("\n=== Varying missing rate π ===")
    for res in sweep_missing_rate(base_cfg, [0.1, 0.2, 0.3, 0.5, 0.7]):
        print(
            f"  π={res.missing_rate:.1f} | λ*={res.lambda_theory:.4f} | "
            f"||β_asgd - β_ridge||={res.coef_diff_norm:.6f} | "
            f"relative={res.relative_norm:.6f}"
        )

    print("\n=== Varying dimension d ===")
    for res in sweep_dimension(base_cfg, [100, 200, 500, 1000, 1500]):
        print(
            f"  d={res.d:>4d} | λ*={res.lambda_theory:.4f} | "
            f"||β_asgd - β_ridge||={res.coef_diff_norm:.6f} | "
            f"relative={res.relative_norm:.6f}"
        )
