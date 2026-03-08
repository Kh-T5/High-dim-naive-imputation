
# Naive Imputation as Implicit Regularization — An Empirical Challenge


This repository provides an empirical evaluation of the theoretical findings in *"Naive imputation implicitly regularizes high-dimensional linear models"* (Ayme et al., 2023), which proves that naive mean imputation under MCAR is asymptotically equivalent to ridge regression on complete data.

## Motivation

While the theoretical equivalence is elegant, several practical questions remain open: Does the implicit ridge equivalence hold across all dimensionality regimes? How does naive imputation fare against modern imputation methods as d grows? This project investigates these questions through controlled numerical experiments.

## Experiments

- **Equivalence verification** — Numerically confirm that the implicit λ of naive imputation matches the paper's closed-form expression across varying missingness rates π and dimensions d.
- **Dimensionality benchmark** — Track coefficient recovery and test MSE as d scales, comparing naive imputation against more advanced techniques.
- **Regime analysis** — Evaluate behavior separately in the n ≫ d and n ≪ d regimes.

## Installation

```bash
git clone https://github.com/Kh-T5/High-dim-naive-imputation.git
cd High-dim-naive-imputation
pip install -r requirements.txt
```

## Reference

Ayme et al. (2023). *Naive imputation implicitly regularizes high-dimensional linear models*. NeurIPS 2023.

---

