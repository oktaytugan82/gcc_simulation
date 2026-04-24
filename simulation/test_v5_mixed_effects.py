"""
Mixed-effects analysis of the V6 retuning-control sweep.

Goal: test whether the selective-backbone effect (condition B vs A in
phase 3, on backbone S) is statistically distinguishable from generic
retuning, treating K_baseline as a within-ensemble covariate.

Primary model: Pi ~ Condition * K_baseline + (1 | Seed)
Primary test: Condition * K_baseline interaction term
Secondary: main effect of Condition, per-K contrasts

Also runs a simpler paired-omnibus test: pooled paired Wilcoxon of
Pi_S^B vs Pi_S^A across all (seed, K) cells.

Addresses Codex v6 findings #2 (post-hoc K=2.0 selection) and #4
(no formal interaction statistic).
"""

import numpy as np
import pickle
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load the retuning control results: {K: {pi_A_S, pi_B_S, ...}}
results = pickle.load(open("../results/v5_retuning_control.pkl", "rb"))

# Build long-format dataframe for phase 3 (re-entry window) on backbone S
rows = []
for K in sorted(results.keys()):
    r = results[K]
    sA = r["pi_A_S"]["P3 re-entry"]
    sB = r["pi_B_S"]["P3 re-entry"]
    n_seeds = len(sA)
    for seed_idx in range(n_seeds):
        rows.append({"Seed": seed_idx, "K": K, "Condition": "A", "Pi": sA[seed_idx]})
        rows.append({"Seed": seed_idx, "K": K, "Condition": "B", "Pi": sB[seed_idx]})

df = pd.DataFrame(rows)
df["ConditionB"] = (df["Condition"] == "B").astype(int)
df["K_c"] = df["K"] - df["K"].mean()   # center K for cleaner interaction

print(f"Dataset: {len(df)} rows, {df['Seed'].nunique()} seeds, "
      f"{df['K'].nunique()} K values, {df['Condition'].nunique()} conditions")
print(df.groupby(["K", "Condition"])["Pi"].describe().round(3))

# ---------------------------------------------------------------
# Primary: mixed-effects model
# Pi ~ Condition + K_c + Condition*K_c, random intercept for Seed
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("PRIMARY MODEL: Pi ~ Condition * K_c + (1 | Seed)")
print("=" * 70)
try:
    model = smf.mixedlm("Pi ~ ConditionB * K_c", df, groups=df["Seed"])
    fit = model.fit(method="lbfgs")
    fit_type = "mixedlm"
except Exception as e:
    print(f"\n  Mixed-LM failed ({type(e).__name__}: {e})")
    print("  Falling back to OLS with cluster-robust SEs (clustered by Seed).")
    model = smf.ols("Pi ~ ConditionB * K_c", data=df)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": df["Seed"]})
    fit_type = "ols_cluster_robust"
print(fit.summary())

# ---------------------------------------------------------------
# Extract key statistics
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("KEY STATISTICS")
print("=" * 70)

# Main effect of Condition (averaged across K, centered K)
cond_coef = fit.params["ConditionB"]
cond_se   = fit.bse["ConditionB"]
cond_z    = fit.tvalues["ConditionB"]
cond_p    = fit.pvalues["ConditionB"]
print(f"\nMain effect of Condition B vs A (at mean K):")
print(f"  Estimate:   {cond_coef:+.4f}")
print(f"  SE:         {cond_se:.4f}")
print(f"  z-value:    {cond_z:.3f}")
print(f"  p-value:    {cond_p:.4g}")

# Main effect of K (at mean Condition)
k_coef = fit.params["K_c"]
k_p    = fit.pvalues["K_c"]
print(f"\nMain effect of K (slope of Pi on K, averaged over Conditions):")
print(f"  Estimate:   {k_coef:+.4f} per unit K")
print(f"  p-value:    {k_p:.4g}")

# Interaction: this is the headline test for Codex finding #4
int_coef = fit.params["ConditionB:K_c"]
int_se   = fit.bse["ConditionB:K_c"]
int_z    = fit.tvalues["ConditionB:K_c"]
int_p    = fit.pvalues["ConditionB:K_c"]
print(f"\nCondition x K interaction (i.e., does the B-vs-A gap depend on K?):")
print(f"  Estimate:   {int_coef:+.4f}")
print(f"  SE:         {int_se:.4f}")
print(f"  z-value:    {int_z:.3f}")
print(f"  p-value:    {int_p:.4g}")

# ---------------------------------------------------------------
# Secondary: per-K paired Wilcoxon (already computed in original)
# and omnibus pooled test
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("SECONDARY: per-K paired Wilcoxon and pooled test")
print("=" * 70)
for K in sorted(results.keys()):
    r = results[K]
    sA = r["pi_A_S"]["P3 re-entry"]
    sB = r["pi_B_S"]["P3 re-entry"]
    try:
        W, p = stats.wilcoxon(sB, sA)
        diff_mean = (sB - sA).mean()
        rel = (sB.mean() - sA.mean()) / sA.mean() * 100 if sA.mean() > 0.001 else np.inf
        print(f"  K={K:.1f}: mean diff = {diff_mean:+.3f}, "
              f"rel = {rel:+.1f}%, W={W:.1f}, p={p:.4g}")
    except ValueError as e:
        print(f"  K={K:.1f}: test failed ({e})")

# Pooled paired test: combine all K into one paired sample
all_A = []
all_B = []
for K in sorted(results.keys()):
    all_A.extend(results[K]["pi_A_S"]["P3 re-entry"].tolist())
    all_B.extend(results[K]["pi_B_S"]["P3 re-entry"].tolist())
all_A = np.array(all_A); all_B = np.array(all_B)
diff = all_B - all_A
W, p_omni = stats.wilcoxon(all_B, all_A)
t, p_t = stats.ttest_rel(all_B, all_A)
print(f"\nOmnibus (pooled across all K, all seeds, N={len(diff)} pairs):")
print(f"  mean diff = {diff.mean():+.4f} ± {diff.std():.4f}")
print(f"  paired Wilcoxon W={W:.1f}, p={p_omni:.4g}")
print(f"  paired t-test t={t:.2f}, p={p_t:.4g}")
print(f"  Cohen's d (paired) = {diff.mean()/diff.std():.3f}")

# ---------------------------------------------------------------
# Save for reference
# ---------------------------------------------------------------
summary = {
    "main_condition_coef": cond_coef,
    "main_condition_p": cond_p,
    "main_K_coef": k_coef,
    "main_K_p": k_p,
    "interaction_coef": int_coef,
    "interaction_p": int_p,
    "omnibus_wilcoxon_W": W,
    "omnibus_wilcoxon_p": p_omni,
    "omnibus_mean_diff": diff.mean(),
    "omnibus_cohens_d": diff.mean() / diff.std(),
    "n_total": len(diff),
    "fit_summary": str(fit.summary()),
}
with open("../results/mixed_effects_summary.pkl", "wb") as f:
    pickle.dump(summary, f)

print("\nSaved mixed_effects_summary.pkl")
