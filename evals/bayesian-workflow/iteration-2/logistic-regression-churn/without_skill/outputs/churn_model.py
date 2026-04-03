"""
Bayesian Logistic Regression for Customer Churn Prediction
==========================================================

This script builds a Bayesian logistic regression model to predict customer churn
probability with full uncertainty estimates. It uses PyMC for inference and
generates synthetic data for demonstration purposes.

Features:
- age: customer age in years
- tenure_months: how long the customer has been with the company
- monthly_spend: average monthly spend in dollars
- support_tickets_last_90d: number of support tickets filed in last 90 days

Target:
- churned: binary (0 = retained, 1 = churned)

Requirements:
    pip install pymc arviz numpy pandas matplotlib scikit-learn
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# =============================================================================
# 1. Generate Synthetic Data
# =============================================================================

def generate_churn_data(n: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic customer churn data with realistic relationships.

    The data-generating process encodes domain knowledge:
    - Higher age slightly reduces churn (loyal, stable customers)
    - Longer tenure strongly reduces churn (invested in the product)
    - Higher monthly spend weakly reduces churn (getting value)
    - More support tickets strongly increases churn (frustrated customers)
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(loc=45, scale=12, size=n).clip(18, 85)
    tenure_months = rng.exponential(scale=24, size=n).clip(1, 120)
    monthly_spend = rng.lognormal(mean=4.0, sigma=0.5, size=n).clip(10, 500)
    support_tickets = rng.poisson(lam=1.5, size=n)

    # True data-generating process (on standardized scale, roughly):
    # We'll work with raw values and set coefficients accordingly.
    # Standardize internally for the linear predictor:
    age_z = (age - age.mean()) / age.std()
    tenure_z = (tenure_months - tenure_months.mean()) / tenure_months.std()
    spend_z = (monthly_spend - monthly_spend.mean()) / monthly_spend.std()
    tickets_z = (support_tickets - support_tickets.mean()) / support_tickets.std()

    # True coefficients (on standardized scale)
    intercept_true = -0.8  # base churn rate ~ 31% at average values
    beta_age_true = -0.15
    beta_tenure_true = -0.6
    beta_spend_true = -0.2
    beta_tickets_true = 0.7

    logit_p = (
        intercept_true
        + beta_age_true * age_z
        + beta_tenure_true * tenure_z
        + beta_spend_true * spend_z
        + beta_tickets_true * tickets_z
    )

    p_churn = 1 / (1 + np.exp(-logit_p))
    churned = rng.binomial(n=1, p=p_churn, size=n)

    df = pd.DataFrame({
        "age": np.round(age, 1),
        "tenure_months": np.round(tenure_months, 1),
        "monthly_spend": np.round(monthly_spend, 2),
        "support_tickets_last_90d": support_tickets,
        "churned": churned,
    })

    return df


# =============================================================================
# 2. Data Preparation
# =============================================================================

def prepare_data(df: pd.DataFrame):
    """
    Split into train/test and standardize features.

    Standardization is critical for Bayesian logistic regression:
    - It makes weakly informative priors meaningful (same scale for all betas)
    - It improves sampler efficiency (better geometry)
    - It makes coefficients directly comparable in magnitude
    """
    feature_cols = ["age", "tenure_months", "monthly_spend", "support_tickets_last_90d"]
    target_col = "churned"

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


# =============================================================================
# 3. Exploratory Data Analysis
# =============================================================================

def run_eda(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Print summary statistics and churn rates by feature quartiles."""
    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape}")
    print(f"Churn rate: {df['churned'].mean():.1%}")
    print(f"\nFeature summary statistics:")
    print(df[feature_cols].describe().round(2).to_string())

    print("\n\nChurn rate by feature quartile:")
    for col in feature_cols:
        df["_quartile"] = pd.qcut(df[col], q=4, labels=["Q1", "Q2", "Q3", "Q4"],
                                   duplicates="drop")
        churn_by_q = df.groupby("_quartile", observed=True)["churned"].mean()
        print(f"\n  {col}:")
        for q, rate in churn_by_q.items():
            print(f"    {q}: {rate:.1%}")
        df.drop(columns="_quartile", inplace=True)

    # Correlation matrix
    print("\n\nCorrelation matrix:")
    print(df[feature_cols + ["churned"]].corr().round(3).to_string())
    print()


# =============================================================================
# 4. Build the Bayesian Model
# =============================================================================

def build_model(X_train: np.ndarray, y_train: np.ndarray,
                feature_names: list[str]) -> pm.Model:
    """
    Build a Bayesian logistic regression model using PyMC.

    Prior choices (all on the standardized feature scale):
    ---------------------------------------------------------
    - Intercept ~ Normal(0, 1.5):
        On the logit scale, 0 means 50% baseline churn. SD of 1.5 allows
        the prior to cover a wide range of baseline rates (roughly 5% to 95%
        at +/- 2 SD), which is diffuse enough to let the data speak while
        keeping the prior weakly informative and regularizing.

    - Betas ~ Normal(0, 1):
        On standardized features, a coefficient of 1 means a 1-SD change in
        the feature shifts the log-odds by 1 -- this is already a substantial
        effect. SD = 1 allows for effects up to ~2-3 on the logit scale
        (very large effects) while gently regularizing against implausibly
        extreme coefficients. This is a standard weakly informative prior
        for logistic regression (see Gelman et al., 2008).

    These priors are weakly informative: they encode the prior belief that
    effects are unlikely to be astronomically large, but they are diffuse
    enough that with n=4000 training observations, the posterior will be
    overwhelmingly driven by the data.
    """
    with pm.Model() as model:
        # --- Data containers ---
        X_data = pm.Data("X", X_train)
        y_data = pm.Data("y", y_train)

        # --- Priors ---
        intercept = pm.Normal("intercept", mu=0, sigma=1.5)
        betas = pm.Normal("betas", mu=0, sigma=1, shape=X_train.shape[1])

        # --- Linear predictor ---
        logit_p = intercept + pm.math.dot(X_data, betas)

        # --- Likelihood ---
        pm.Bernoulli("churn", logit_p=logit_p, observed=y_data)

    return model


# =============================================================================
# 5. Fit the Model (MCMC Sampling)
# =============================================================================

def fit_model(model: pm.Model) -> az.InferenceData:
    """
    Sample from the posterior using NUTS.

    We run 4 chains with 2000 draws each (plus 1000 tuning steps).
    This gives us 8000 posterior draws total, which is plenty for
    stable estimates and reliable convergence diagnostics.
    """
    with model:
        idata = pm.sample(
            draws=2000,
            tune=1000,
            chains=4,
            cores=4,
            target_accept=0.9,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
        )
    return idata


# =============================================================================
# 6. Diagnostics
# =============================================================================

def run_diagnostics(idata: az.InferenceData, feature_names: list[str]) -> None:
    """
    Run and print comprehensive MCMC diagnostics.

    Key diagnostics:
    - R-hat: should be < 1.01 for all parameters (chain convergence)
    - ESS (bulk & tail): should be > 400 per parameter (effective sample size)
    - Divergences: should be 0 (indicates geometric pathologies if present)
    - Trace plots: visual check for mixing and stationarity
    """
    print("=" * 70)
    print("MCMC DIAGNOSTICS")
    print("=" * 70)

    summary = az.summary(idata, var_names=["intercept", "betas"],
                         hdi_prob=0.94)
    print("\nPosterior summary (94% HDI):")
    # Rename beta indices to feature names for clarity
    new_index = ["intercept"] + [f"beta_{name}" for name in feature_names]
    summary.index = new_index
    print(summary.to_string())

    # Check R-hat
    rhat_vals = summary["r_hat"].values
    max_rhat = rhat_vals.max()
    print(f"\nMax R-hat: {max_rhat:.4f}", end="")
    if max_rhat < 1.01:
        print(" -- GOOD (< 1.01, chains have converged)")
    else:
        print(" -- WARNING: R-hat >= 1.01, chains may not have converged!")

    # Check ESS
    ess_bulk_min = summary["ess_bulk"].min()
    ess_tail_min = summary["ess_tail"].min()
    print(f"Min ESS (bulk): {ess_bulk_min:.0f}", end="")
    if ess_bulk_min > 400:
        print(" -- GOOD (> 400)")
    else:
        print(" -- WARNING: ESS too low, increase draws or check model")
    print(f"Min ESS (tail): {ess_tail_min:.0f}", end="")
    if ess_tail_min > 400:
        print(" -- GOOD (> 400)")
    else:
        print(" -- WARNING: Tail ESS too low")

    # Check divergences
    divergences = idata.sample_stats["diverging"].sum().values
    print(f"Divergences: {divergences}", end="")
    if divergences == 0:
        print(" -- GOOD (no divergences)")
    else:
        print(f" -- WARNING: {divergences} divergences detected! "
              "Consider reparameterization or increasing target_accept.")

    print()


def plot_diagnostics(idata: az.InferenceData, feature_names: list[str],
                     output_dir: str) -> None:
    """Generate and save diagnostic plots."""
    var_names = ["intercept", "betas"]

    # Trace plots
    axes = az.plot_trace(idata, var_names=var_names, compact=True, figsize=(14, 8))
    plt.suptitle("Trace Plots", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/trace_plots.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Posterior distributions
    axes = az.plot_posterior(idata, var_names=var_names, hdi_prob=0.94,
                            figsize=(14, 6))
    plt.suptitle("Posterior Distributions (94% HDI)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/posterior_distributions.png", dpi=150,
                bbox_inches="tight")
    plt.close()

    # Forest plot for betas
    axes = az.plot_forest(idata, var_names=["betas"], hdi_prob=0.94,
                          combined=True, figsize=(10, 5))
    # Relabel y-axis with feature names
    ax = plt.gca()
    ax.set_yticklabels(feature_names[::-1])
    plt.title("Coefficient Forest Plot (94% HDI)")
    plt.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/forest_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Rank plots (better than trace plots for convergence assessment)
    axes = az.plot_rank(idata, var_names=var_names, figsize=(14, 8))
    plt.suptitle("Rank Plots (uniform = good mixing)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rank_plots.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Diagnostic plots saved to {output_dir}/")


# =============================================================================
# 7. Posterior Predictive Checks & Predictions
# =============================================================================

def posterior_predictions(model: pm.Model, idata: az.InferenceData,
                         X_test: np.ndarray, y_test: np.ndarray,
                         feature_names: list[str],
                         output_dir: str) -> np.ndarray:
    """
    Generate posterior predictive samples for the test set.

    This is where the Bayesian approach shines: instead of a single predicted
    probability, we get a full distribution of predicted probabilities for
    each customer. This lets us quantify uncertainty in our predictions.
    """
    print("=" * 70)
    print("POSTERIOR PREDICTIVE ANALYSIS")
    print("=" * 70)

    with model:
        pm.set_data({"X": X_test, "y": np.zeros(len(y_test), dtype=int)})
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["churn"],
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
        )

    # Extract predicted probabilities from the linear predictor
    # We need to compute these manually from the posterior samples
    intercept_samples = idata.posterior["intercept"].values  # (chains, draws)
    beta_samples = idata.posterior["betas"].values  # (chains, draws, n_features)

    # Reshape for matrix multiplication: (n_samples, n_features)
    n_chains, n_draws = intercept_samples.shape
    intercept_flat = intercept_samples.reshape(-1)  # (n_samples,)
    betas_flat = beta_samples.reshape(-1, X_test.shape[1])  # (n_samples, n_features)

    # Compute logit(p) for each posterior sample and each test observation
    # Result shape: (n_samples, n_test_obs)
    logit_p = intercept_flat[:, None] + betas_flat @ X_test.T

    # Convert to probabilities
    p_churn = 1 / (1 + np.exp(-logit_p))  # (n_samples, n_test_obs)

    # Summary statistics
    p_mean = p_churn.mean(axis=0)
    p_std = p_churn.std(axis=0)
    p_lower = np.percentile(p_churn, 3, axis=0)   # 94% HDI lower
    p_upper = np.percentile(p_churn, 97, axis=0)   # 94% HDI upper

    # Classification using 0.5 threshold on mean probability
    y_pred = (p_mean >= 0.5).astype(int)
    accuracy = (y_pred == y_test).mean()

    print(f"\nTest set accuracy: {accuracy:.1%}")
    print(f"Test set size: {len(y_test)}")
    print(f"Churn rate in test set: {y_test.mean():.1%}")

    print(f"\nPredicted probability summary:")
    print(f"  Mean of predicted probabilities: {p_mean.mean():.3f}")
    print(f"  Std of predicted probabilities:  {p_mean.std():.3f}")
    print(f"  Average uncertainty (posterior SD): {p_std.mean():.3f}")

    # Show some example predictions with uncertainty
    print(f"\nExample predictions (first 10 test observations):")
    print(f"{'Obs':>4s} | {'True':>5s} | {'P(churn)':>9s} | {'94% HDI':>18s} | {'Width':>6s}")
    print("-" * 55)
    for i in range(min(10, len(y_test))):
        width = p_upper[i] - p_lower[i]
        print(f"{i:4d} | {y_test[i]:5d} | {p_mean[i]:9.3f} | "
              f"[{p_lower[i]:.3f}, {p_upper[i]:.3f}] | {width:.3f}")

    # --- Plots ---

    # 1. Predicted probability distribution by true outcome
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(p_mean[y_test == 0], bins=40, alpha=0.6, label="Retained (y=0)",
                 color="steelblue", density=True)
    axes[0].hist(p_mean[y_test == 1], bins=40, alpha=0.6, label="Churned (y=1)",
                 color="coral", density=True)
    axes[0].set_xlabel("Mean Predicted P(Churn)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Predicted Probabilities by True Outcome")
    axes[0].legend()

    # 2. Calibration-style plot: uncertainty vs correctness
    correct = (y_pred == y_test)
    axes[1].scatter(p_std[correct], p_mean[correct], alpha=0.15, s=8,
                    label="Correct", color="steelblue")
    axes[1].scatter(p_std[~correct], p_mean[~correct], alpha=0.3, s=12,
                    label="Incorrect", color="coral")
    axes[1].set_xlabel("Posterior Uncertainty (SD)")
    axes[1].set_ylabel("Mean Predicted P(Churn)")
    axes[1].set_title("Prediction Uncertainty vs Mean Prediction")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/predictions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Uncertainty intervals for a subset of observations
    n_show = 50
    idx_sorted = np.argsort(p_mean[:n_show])
    fig, ax = plt.subplots(figsize=(12, 6))
    for rank, i in enumerate(idx_sorted):
        color = "coral" if y_test[i] == 1 else "steelblue"
        ax.plot([rank, rank], [p_lower[i], p_upper[i]], color=color, alpha=0.5, lw=2)
        ax.plot(rank, p_mean[i], "o", color=color, markersize=4)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Observation (sorted by predicted probability)")
    ax.set_ylabel("P(Churn)")
    ax.set_title(f"Churn Probability with 94% HDI (first {n_show} test obs)")

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="coral", lw=2, label="Churned (true)"),
        Line2D([0], [0], color="steelblue", lw=2, label="Retained (true)"),
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/uncertainty_intervals.png", dpi=150,
                bbox_inches="tight")
    plt.close()

    print(f"\nPrediction plots saved to {output_dir}/")

    return p_churn


# =============================================================================
# 8. Interpret Coefficients
# =============================================================================

def interpret_coefficients(idata: az.InferenceData, feature_names: list[str],
                           scaler: StandardScaler, output_dir: str) -> None:
    """
    Interpret posterior coefficient distributions.

    Since features were standardized, the beta coefficients represent the
    change in log-odds of churn per 1-SD change in the feature. We can
    convert these to odds ratios for more intuitive interpretation.
    """
    print("=" * 70)
    print("COEFFICIENT INTERPRETATION")
    print("=" * 70)

    beta_samples = idata.posterior["betas"].values.reshape(-1, len(feature_names))
    intercept_samples = idata.posterior["intercept"].values.reshape(-1)

    print("\nCoefficients on standardized scale (change in log-odds per 1-SD):")
    print(f"{'Feature':<30s} | {'Mean':>7s} | {'SD':>6s} | {'94% HDI':>18s} | "
          f"{'P(beta>0)':>9s}")
    print("-" * 85)

    for j, name in enumerate(feature_names):
        b = beta_samples[:, j]
        mean_b = b.mean()
        sd_b = b.std()
        hdi = az.hdi(b, hdi_prob=0.94)
        prob_positive = (b > 0).mean()
        print(f"{name:<30s} | {mean_b:7.3f} | {sd_b:6.3f} | "
              f"[{hdi[0]:.3f}, {hdi[1]:.3f}] | {prob_positive:9.3f}")

    # Odds ratios (per 1-SD change)
    print("\n\nOdds ratios (per 1-SD change in feature):")
    print(f"{'Feature':<30s} | {'Median OR':>10s} | {'94% HDI':>22s}")
    print("-" * 70)
    for j, name in enumerate(feature_names):
        or_samples = np.exp(beta_samples[:, j])
        median_or = np.median(or_samples)
        hdi_or = az.hdi(or_samples, hdi_prob=0.94)
        print(f"{name:<30s} | {median_or:10.3f} | "
              f"[{hdi_or[0]:.3f}, {hdi_or[1]:.3f}]")

    # Convert to original scale for practical interpretation
    print("\n\nPractical interpretation (per unit change on original scale):")
    feature_sds = scaler.scale_
    feature_means = scaler.mean_
    print(f"{'Feature':<30s} | {'1 SD =':<12s} | {'Per-unit log-odds change':>25s}")
    print("-" * 75)
    for j, name in enumerate(feature_names):
        sd_j = feature_sds[j]
        mean_beta_j = beta_samples[:, j].mean()
        per_unit = mean_beta_j / sd_j
        print(f"{name:<30s} | {sd_j:10.2f}  | {per_unit:25.4f}")

    # Intercept interpretation
    int_mean = intercept_samples.mean()
    baseline_prob = 1 / (1 + np.exp(-int_mean))
    print(f"\nIntercept (mean): {int_mean:.3f}")
    print(f"Baseline churn probability (at mean feature values): {baseline_prob:.1%}")

    print()


# =============================================================================
# 9. Prior Predictive Check
# =============================================================================

def prior_predictive_check(X_train: np.ndarray, feature_names: list[str],
                           output_dir: str) -> None:
    """
    Sample from the prior predictive distribution to verify priors are sensible.

    This is a critical step in the Bayesian workflow: before seeing any data,
    we should check that our priors generate plausible predictions. If the
    prior predictive distribution puts too much mass on extreme probabilities
    (all 0 or all 1), our priors are likely too diffuse.
    """
    print("=" * 70)
    print("PRIOR PREDICTIVE CHECK")
    print("=" * 70)

    with pm.Model() as prior_model:
        X_data = pm.Data("X", X_train[:100])  # use subset for speed

        intercept = pm.Normal("intercept", mu=0, sigma=1.5)
        betas = pm.Normal("betas", mu=0, sigma=1, shape=X_train.shape[1])

        logit_p = intercept + pm.math.dot(X_data, betas)
        pm.Bernoulli("churn", logit_p=logit_p, shape=100)

        prior_ppc = pm.sample_prior_predictive(samples=1000,
                                                random_seed=RANDOM_SEED,
                                                return_inferencedata=True)

    # Compute implied churn rates from prior
    prior_churn = prior_ppc.prior_predictive["churn"].values.reshape(1000, -1)
    prior_rates = prior_churn.mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(prior_rates, bins=40, color="steelblue", alpha=0.7, density=True)
    axes[0].set_xlabel("Prior Predicted Churn Rate")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Prior Predictive Distribution of Churn Rate")
    axes[0].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="50%")
    axes[0].legend()

    # Also show distribution of individual probabilities
    intercept_prior = prior_ppc.prior["intercept"].values.reshape(-1)
    betas_prior = prior_ppc.prior["betas"].values.reshape(-1, X_train.shape[1])

    # Compute implied probabilities for a few observations
    logits = intercept_prior[:, None] + betas_prior @ X_train[:20].T
    probs = 1 / (1 + np.exp(-logits))

    axes[1].hist(probs.ravel(), bins=50, color="steelblue", alpha=0.7, density=True)
    axes[1].set_xlabel("Prior Predicted P(Churn) for Individual")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Prior Distribution of Individual Churn Probabilities")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/prior_predictive.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Prior predictive churn rate: mean = {prior_rates.mean():.2f}, "
          f"SD = {prior_rates.std():.2f}")
    print(f"Prior predictive check plot saved to {output_dir}/prior_predictive.png")
    print()


# =============================================================================
# 10. Model Comparison (vs. simpler model)
# =============================================================================

def compare_models(X_train: np.ndarray, y_train: np.ndarray,
                   feature_names: list[str], output_dir: str) -> None:
    """
    Compare our full model against an intercept-only model using LOO-CV.

    This serves as a sanity check: the full model should substantially
    outperform the intercept-only model if our features are informative.
    """
    print("=" * 70)
    print("MODEL COMPARISON (LOO-CV)")
    print("=" * 70)

    # Intercept-only model
    with pm.Model() as null_model:
        intercept = pm.Normal("intercept", mu=0, sigma=1.5)
        pm.Bernoulli("churn", logit_p=intercept, observed=y_train)
        idata_null = pm.sample(draws=2000, tune=1000, chains=4,
                               random_seed=RANDOM_SEED,
                               return_inferencedata=True)
        pm.compute_log_likelihood(idata_null)

    # Full model
    with pm.Model() as full_model:
        X_data = pm.Data("X", X_train)
        intercept = pm.Normal("intercept", mu=0, sigma=1.5)
        betas = pm.Normal("betas", mu=0, sigma=1, shape=X_train.shape[1])
        logit_p = intercept + pm.math.dot(X_data, betas)
        pm.Bernoulli("churn", logit_p=logit_p, observed=y_train)
        idata_full = pm.sample(draws=2000, tune=1000, chains=4,
                               random_seed=RANDOM_SEED,
                               return_inferencedata=True)
        pm.compute_log_likelihood(idata_full)

    # Compare using LOO
    comp = az.compare(
        {"intercept_only": idata_null, "full_model": idata_full},
        ic="loo",
    )
    print("\nLOO-CV comparison:")
    print(comp.to_string())
    print()

    # Plot comparison
    az.plot_compare(comp, figsize=(10, 4))
    plt.title("Model Comparison (LOO-CV)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Model comparison plot saved to {output_dir}/model_comparison.png")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = (
        "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
        "bayesian-workflow-workspace/iteration-1/logistic-regression-churn/"
        "without_skill/outputs"
    )

    # 1. Generate data
    print("Generating synthetic churn data (n=5000)...\n")
    df = generate_churn_data(n=5000)

    # 2. Prepare data
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df)
    print(f"Train set: {len(y_train)} observations ({y_train.mean():.1%} churn)")
    print(f"Test set:  {len(y_test)} observations ({y_test.mean():.1%} churn)\n")

    # 3. EDA
    run_eda(df, feature_names)

    # 4. Prior predictive check
    prior_predictive_check(X_train, feature_names, output_dir)

    # 5. Build and fit the model
    print("Building Bayesian logistic regression model...\n")
    model = build_model(X_train, y_train, feature_names)
    print(model)
    print()

    print("Sampling from posterior (this may take a few minutes)...\n")
    idata = fit_model(model)

    # 6. Diagnostics
    run_diagnostics(idata, feature_names)
    plot_diagnostics(idata, feature_names, output_dir)

    # 7. Interpret coefficients
    interpret_coefficients(idata, feature_names, scaler, output_dir)

    # 8. Posterior predictions with uncertainty
    p_churn = posterior_predictions(model, idata, X_test, y_test,
                                    feature_names, output_dir)

    # 9. Model comparison
    compare_models(X_train, y_train, feature_names, output_dir)

    # 10. Save the inference data for later use
    idata.to_netcdf(f"{output_dir}/churn_model_idata.nc")
    print(f"\nInference data saved to {output_dir}/churn_model_idata.nc")
    print("\nDone!")


if __name__ == "__main__":
    main()
