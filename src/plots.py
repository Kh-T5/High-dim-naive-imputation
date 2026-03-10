import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


def plot_threshold_experiment(RESULTS_DIR, PLOT_STYLE, df: pd.DataFrame) -> None:
    plt.rcParams.update(PLOT_STYLE)
    n_values = sorted(df["n"].unique())
    n_cols = min(2, len(n_values))
    n_rows = int(np.ceil(len(n_values) / n_cols))
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(n_values)))

    fig1, axes1 = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False
    )
    fig1.suptitle(
        r"Equivalence gap vs. dimension $d$" + "\n"
        r"$\|\hat{\beta}_{\rm naive} - \hat{\beta}_{\rm ridge}\|_2 \;/\; \|\hat{\beta}_{\rm ridge}\|_2$",
        fontsize=14,
        y=1.01,
    )
    for idx, (n, color) in enumerate(zip(n_values, colors)):
        ax = axes1[idx // n_cols][idx % n_cols]
        sub = df[df["n"] == n].sort_values("d")
        sqrt_n = np.sqrt(n)
        ax.plot(sub["d"], sub["relative_norm_mean"], color=color, lw=2, label=f"n={n}")
        ax.fill_between(
            sub["d"],
            sub["relative_norm_mean"] - sub["relative_norm_std"],
            sub["relative_norm_mean"] + sub["relative_norm_std"],
            alpha=0.2,
            color=color,
        )
        ax.axvline(
            sqrt_n,
            color="crimson",
            lw=1.5,
            ls="--",
            label=r"$d=\sqrt{n}$" + f" ({sqrt_n:.0f})",
        )
        ax.set_xlabel("Dimension $d$")
        ax.set_ylabel("Relative norm")
        ax.set_title(f"n = {n}")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    for idx in range(len(n_values), n_rows * n_cols):
        axes1[idx // n_cols][idx % n_cols].set_visible(False)
    fig1.tight_layout()
    fig1.savefig(RESULTS_DIR / "threshold_vs_d.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {RESULTS_DIR / 'threshold_vs_d.png'}")

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    fig2.suptitle(
        r"Equivalence gap vs. $d\,/\,\sqrt{n}$ (all $n$ overlaid)", fontsize=14
    )
    for n, color in zip(n_values, colors):
        sub = df[df["n"] == n].sort_values("d_over_sqrt_n")
        ax2.plot(
            sub["d_over_sqrt_n"],
            sub["relative_norm_mean"],
            color=color,
            lw=2,
            label=f"n={n}",
        )
        ax2.fill_between(
            sub["d_over_sqrt_n"],
            sub["relative_norm_mean"] - sub["relative_norm_std"],
            sub["relative_norm_mean"] + sub["relative_norm_std"],
            alpha=0.15,
            color=color,
        )
    ax2.axvline(1.0, color="crimson", lw=2, ls="--", label=r"$d/\sqrt{n}=1$")
    ax2.set_xlabel(r"$d\;/\;\sqrt{n}$")
    ax2.set_ylabel("Relative norm")
    ax2.legend(fontsize=10)
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / "threshold_vs_ratio.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {RESULTS_DIR / 'threshold_vs_ratio.png'}")
    plt.show()


def plot_mse_benchmark(
    RESULTS_DIR, PLOT_STYLE, METHOD_STYLE, mse_df: pd.DataFrame
) -> None:
    plt.rcParams.update(PLOT_STYLE)
    n_values = sorted(mse_df["n"].unique())
    imputers = list(mse_df["imputer"].unique())
    n_cols = min(2, len(n_values))
    n_rows = int(np.ceil(len(n_values) / n_cols))

    fig1, axes1 = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), squeeze=False
    )
    fig1.suptitle(
        "Test MSE vs. dimension $d$ — all imputation methods\n"
        r"Downstream estimator: ASGD (Polyak-Ruppert averaging)",
        fontsize=14,
        y=1.01,
    )
    for idx, n in enumerate(n_values):
        ax = axes1[idx // n_cols][idx % n_cols]
        sqrt_n = np.sqrt(n)
        sub_n = mse_df[mse_df["n"] == n]

        for imputer in imputers:
            style = METHOD_STYLE.get(
                imputer, {"color": "grey", "lw": 1.5, "ls": "-", "zorder": 3}
            )
            sub = sub_n[sub_n["imputer"] == imputer].sort_values("d")
            ax.plot(
                sub["d"],
                sub["mse_mean"],
                label=imputer,
                color=style["color"],
                lw=style["lw"],
                ls=style["ls"],
                zorder=style["zorder"],
            )
            ax.fill_between(
                sub["d"],
                sub["mse_mean"] - sub["mse_std"],
                sub["mse_mean"] + sub["mse_std"],
                alpha=0.1,
                color=style["color"],
            )

        ax.axvline(
            sqrt_n,
            color="crimson",
            lw=1.2,
            ls="--",
            label=r"$d=\sqrt{n}$" + f" ({sqrt_n:.0f})",
            zorder=2,
        )
        ax.set_xlabel("Dimension $d$")
        ax.set_ylabel("Test MSE")
        ax.set_title(f"n = {n}")
        ax.legend(fontsize=8, ncol=2)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))

    for idx in range(len(n_values), n_rows * n_cols):
        axes1[idx // n_cols][idx % n_cols].set_visible(False)
    fig1.tight_layout()
    fig1.savefig(RESULTS_DIR / "mse_vs_d.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {RESULTS_DIR / 'mse_vs_d.png'}")

    n_imp = len(imputers)
    n_imp_cols = min(3, n_imp)
    n_imp_rows = int(np.ceil(n_imp / n_imp_cols))
    n_colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(n_values)))

    fig2, axes2 = plt.subplots(
        n_imp_rows,
        n_imp_cols,
        figsize=(6 * n_imp_cols, 4 * n_imp_rows),
        squeeze=False,
    )
    fig2.suptitle(
        r"Test MSE vs. $d\,/\,\sqrt{n}$ — all $n$ overlaid, per method",
        fontsize=14,
        y=1.01,
    )
    for imp_idx, imputer in enumerate(imputers):
        ax = axes2[imp_idx // n_imp_cols][imp_idx % n_imp_cols]
        style = METHOD_STYLE.get(
            imputer, {"color": "grey", "lw": 1.5, "ls": "-", "zorder": 3}
        )

        for n, nc in zip(n_values, n_colors):
            sub = mse_df[
                (mse_df["imputer"] == imputer) & (mse_df["n"] == n)
            ].sort_values("d_over_sqrt_n")
            ax.plot(
                sub["d_over_sqrt_n"], sub["mse_mean"], color=nc, lw=1.8, label=f"n={n}"
            )
            ax.fill_between(
                sub["d_over_sqrt_n"],
                sub["mse_mean"] - sub["mse_std"],
                sub["mse_mean"] + sub["mse_std"],
                alpha=0.1,
                color=nc,
            )

        ax.axvline(1.0, color="crimson", lw=1.2, ls="--", label=r"$d/\sqrt{n}=1$")
        ax.set_xlabel(r"$d\;/\;\sqrt{n}$")
        ax.set_ylabel("Test MSE")
        ax.set_title(imputer, color=style["color"], fontweight="bold")
        ax.legend(fontsize=7)

    for idx in range(n_imp, n_imp_rows * n_imp_cols):
        axes2[idx // n_imp_cols][idx % n_imp_cols].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(
        RESULTS_DIR / "mse_vs_ratio_per_method.png", dpi=150, bbox_inches="tight"
    )
    print(f"Saved: {RESULTS_DIR / 'mse_vs_ratio_per_method.png'}")

    plt.show()
