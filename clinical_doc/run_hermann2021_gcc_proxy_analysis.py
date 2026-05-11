from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = Path(__file__).resolve().parent
CSV_PATH = DOC_ROOT / "Hermann2021_supplementary_metadata_mmc2.csv"
OUT_FEATURES = ROOT / "results" / "hermann2021_gcc_proxy_features.csv"
OUT_SUMMARY = ROOT / "results" / "hermann2021_gcc_proxy_summary.json"
OUT_REPORT = ROOT / "results" / "hermann2021_gcc_proxy_report.md"
OUT_DIAG_FIG = ROOT / "figures" / "hermann2021_gcc_proxy_diagnostic.png"
OUT_OUTCOME_FIG = ROOT / "figures" / "hermann2021_gcc_proxy_outcome.png"

RANDOM_STATE = 20260427
N_BOOT = 5000
N_BOOT_DIFF = 50000
EPS = 1e-12


def repo_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def clean_header(value: str) -> str:
    return " ".join(value.replace("\n", " ").split())


def clean_text(value: str) -> str:
    return " ".join(str(value).replace("\n", " ").split()).strip()


def as_float(value: str) -> float:
    value = clean_text(value).replace(",", ".")
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))
    header = [clean_header(v) for v in rows[1]]
    out = []
    for row in rows[2:]:
        item = {key: clean_text(value) for key, value in zip(header, row)}
        out.append(item)
    return out


def normalize_state(value: str) -> str:
    value = clean_text(value)
    if value == "Concious":
        return "Conscious"
    return value


def baseline_binary(state: str) -> float:
    if state == "VS/UWS":
        return 0.0
    if state in {"MCS-", "MCS+"}:
        return 1.0
    return math.nan


def baseline_order(state: str) -> float:
    return {
        "VS/UWS": 0.0,
        "MCS-": 1.0,
        "MCS+": 2.0,
        "EMCS": 3.0,
    }.get(state, math.nan)


def outcome_command_following(state: str) -> float:
    return 1.0 if state in {"MCS+", "Conscious"} else 0.0


def local_global_score(value: str) -> float:
    return {
        "None": 0.0,
        "Local": 0.5,
        "Global": 1.0,
    }.get(value, math.nan)


def zscore(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return out
    mean = float(np.mean(values[mask]))
    sd = float(np.std(values[mask], ddof=1))
    if sd < EPS:
        return out
    out[mask] = (values[mask] - mean) / sd
    return out


def finite_xy(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y) & np.isfinite(score)
    return y[mask].astype(int), score[mask].astype(float)


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    y, score = finite_xy(y, score)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return math.nan
    pos = score[y == 1]
    neg = score[y == 0]
    ranks = stats.rankdata(np.concatenate([pos, neg]), method="average")
    n_pos = len(pos)
    n_neg = len(neg)
    rank_sum_pos = float(np.sum(ranks[:n_pos]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auc_ci(y: np.ndarray, score: np.ndarray, rng: np.random.Generator) -> list[float]:
    y, score = finite_xy(y, score)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return [math.nan, math.nan]
    pos = score[y == 1]
    neg = score[y == 0]
    aucs = []
    for _ in range(N_BOOT):
        b_pos = rng.choice(pos, size=len(pos), replace=True)
        b_neg = rng.choice(neg, size=len(neg), replace=True)
        b_y = np.array([1] * len(b_pos) + [0] * len(b_neg), dtype=int)
        b_score = np.concatenate([b_pos, b_neg])
        aucs.append(auc_score(b_y, b_score))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def mannwhitney_p(y: np.ndarray, score: np.ndarray) -> float:
    y, score = finite_xy(y, score)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return math.nan
    pos = score[y == 1]
    neg = score[y == 0]
    try:
        return float(stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
    except ValueError:
        return math.nan


def spearman(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(score)
    if mask.sum() < 3:
        return {"rho": math.nan, "p": math.nan, "n": int(mask.sum())}
    res = stats.spearmanr(y[mask], score[mask])
    return {"rho": float(res.statistic), "p": float(res.pvalue), "n": int(mask.sum())}


def optimal_threshold(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y, score = finite_xy(y, score)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"threshold": math.nan, "sensitivity": math.nan, "specificity": math.nan, "youden": math.nan}
    candidates = sorted(set(float(v) for v in score))
    best = None
    for thr in candidates:
        pred = score >= thr
        tp = int(np.sum((pred == 1) & (y == 1)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        tn = int(np.sum((pred == 0) & (y == 0)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        sensitivity = tp / (tp + fn) if (tp + fn) else math.nan
        specificity = tn / (tn + fp) if (tn + fp) else math.nan
        youden = sensitivity + specificity - 1.0
        candidate = (youden, sensitivity, specificity, -thr)
        if best is None or candidate > best[0]:
            best = (candidate, thr, sensitivity, specificity)
    _, thr, sensitivity, specificity = best
    return {
        "threshold": float(thr),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "youden": float(sensitivity + specificity - 1.0),
    }


def model_row(name: str, y: np.ndarray, score: np.ndarray, rng: np.random.Generator) -> dict[str, object]:
    yy, ss = finite_xy(y, score)
    return {
        "score": name,
        "n": int(len(yy)),
        "n_positive": int(np.sum(yy == 1)),
        "n_negative": int(np.sum(yy == 0)),
        "auc": float(auc_score(yy, ss)),
        "auc_ci95": auc_ci(yy, ss, rng),
        "mannwhitney_p": mannwhitney_p(yy, ss),
        "threshold_youden": optimal_threshold(yy, ss),
    }


def paired_auc_delta(
    target_name: str,
    y: np.ndarray,
    score_a_name: str,
    score_a: np.ndarray,
    score_b_name: str,
    score_b: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, object]:
    mask = np.isfinite(y) & np.isfinite(score_a) & np.isfinite(score_b)
    yy = y[mask].astype(int)
    aa = score_a[mask].astype(float)
    bb = score_b[mask].astype(float)
    if len(yy) == 0 or len(np.unique(yy)) < 2:
        return {
            "target": target_name,
            "score_a": score_a_name,
            "score_b": score_b_name,
            "n": int(len(yy)),
            "delta_auc": math.nan,
            "delta_auc_ci95": [math.nan, math.nan],
            "bootstrap_p_two_sided": math.nan,
        }

    observed_a = auc_score(yy, aa)
    observed_b = auc_score(yy, bb)
    observed_delta = observed_a - observed_b
    idx = np.arange(len(yy))
    deltas = []
    skipped = 0
    for _ in range(N_BOOT_DIFF):
        sample = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(yy[sample])) < 2:
            skipped += 1
            continue
        deltas.append(auc_score(yy[sample], aa[sample]) - auc_score(yy[sample], bb[sample]))
    deltas = np.asarray(deltas, dtype=float)
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    p_two_sided = 2.0 * min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0)))
    return {
        "target": target_name,
        "score_a": score_a_name,
        "score_b": score_b_name,
        "n": int(len(yy)),
        "n_positive": int(np.sum(yy == 1)),
        "n_negative": int(np.sum(yy == 0)),
        "auc_a": float(observed_a),
        "auc_b": float(observed_b),
        "delta_auc": float(observed_delta),
        "delta_auc_ci95": [float(lower), float(upper)],
        "bootstrap_p_two_sided": min(1.0, float(p_two_sided)),
        "n_bootstrap": int(len(deltas)),
        "n_skipped": int(skipped),
    }


def group_counts(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    return dict(Counter(str(row[key]) for row in rows))


def write_features(rows: list[dict[str, object]]) -> None:
    fields = [
        "patient_id",
        "age",
        "gender",
        "etiology",
        "baseline_state",
        "baseline_mcs_binary",
        "baseline_order",
        "outcome_state",
        "outcome_command_following",
        "gose",
        "pet_mibh",
        "eeg_svm_oos",
        "local_global",
        "local_global_score",
        "pet_z",
        "eeg_z",
        "local_global_z",
        "gcc_pet_eeg_proxy",
        "gcc_pet_eeg_lg_proxy",
        "pet_positive_youden",
        "eeg_positive_050",
        "gcc_gate",
    ]
    with OUT_FEATURES.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize_gates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["gcc_gate"])].append(row)
    out = []
    for gate, items in sorted(grouped.items()):
        diag = [x["baseline_state"] for x in items]
        outcome = [x["outcome_state"] for x in items]
        out.append(
            {
                "gate": gate,
                "n": len(items),
                "baseline_counts": dict(Counter(diag)),
                "outcome_counts": dict(Counter(outcome)),
                "outcome_command_following_rate": float(np.mean([x["outcome_command_following"] for x in items])),
                "patients": [x["patient_id"] for x in items],
            }
        )
    return out


def plot_diagnostic(rows: list[dict[str, object]]) -> None:
    states = ["VS/UWS", "MCS-", "MCS+", "EMCS"]
    scores = [
        ("pet_mibh", "PET MIBH"),
        ("eeg_svm_oos", "EEG SVM p(MCS)"),
        ("gcc_pet_eeg_proxy", "GCC PET+EEG proxy"),
    ]
    fig, axes = plt.subplots(1, len(scores), figsize=(12, 4), constrained_layout=True)
    colors = {"VS/UWS": "#6f7d8c", "MCS-": "#4c78a8", "MCS+": "#f58518", "EMCS": "#54a24b"}
    for ax, (field, title) in zip(axes, scores):
        data = []
        positions = []
        for i, state in enumerate(states, start=1):
            vals = [float(r[field]) for r in rows if r["baseline_state"] == state and np.isfinite(float(r[field]))]
            data.append(vals)
            positions.append(i)
        ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
        for i, state in enumerate(states, start=1):
            vals = np.array(data[i - 1], dtype=float)
            if len(vals):
                jitter = np.linspace(-0.13, 0.13, len(vals)) if len(vals) > 1 else np.array([0.0])
                ax.scatter(np.full(len(vals), i) + jitter, vals, s=28, color=colors[state], alpha=0.82, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(states, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Hermann 2021: GCC proxy by baseline state", fontsize=13)
    fig.savefig(OUT_DIAG_FIG, dpi=220)
    plt.close(fig)


def plot_outcome(rows: list[dict[str, object]], pet_threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    color_map = {0.0: "#6f7d8c", 1.0: "#e45756"}
    label_map = {0.0: "No command-following outcome", 1.0: "MCS+ / conscious outcome"}
    for outcome in [0.0, 1.0]:
        subset = [
            r for r in rows
            if r["outcome_command_following"] == outcome
            and np.isfinite(float(r["pet_mibh"]))
            and np.isfinite(float(r["eeg_svm_oos"]))
        ]
        ax.scatter(
            [float(r["pet_mibh"]) for r in subset],
            [float(r["eeg_svm_oos"]) for r in subset],
            s=44,
            color=color_map[outcome],
            alpha=0.82,
            label=label_map[outcome],
        )
    ax.axvline(pet_threshold, color="#333333", linestyle="--", linewidth=1.2, label="PET diagnostic threshold")
    ax.axhline(0.5, color="#333333", linestyle=":", linewidth=1.2, label="EEG SVM 0.50")
    ax.set_xlabel("PET metabolic index of best preserved hemisphere")
    ax.set_ylabel("Out-of-sample EEG SVM p(MCS)")
    ax.set_title("Hermann 2021: PET capacity vs EEG dynamics")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT_OUTCOME_FIG, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIAG_FIG.parent.mkdir(parents=True, exist_ok=True)
    raw = load_rows(CSV_PATH)
    rows: list[dict[str, object]] = []
    for item in raw:
        baseline_state = normalize_state(item["State of consciousness"])
        outcome_state = normalize_state(item["State of Consciousness"])
        rows.append(
            {
                "patient_id": item["Patient_ID"],
                "age": as_float(item["Age (years)"]),
                "gender": item["Gender"],
                "etiology": item["Etiology"],
                "baseline_state": baseline_state,
                "baseline_mcs_binary": baseline_binary(baseline_state),
                "baseline_order": baseline_order(baseline_state),
                "outcome_state": outcome_state,
                "outcome_command_following": outcome_command_following(outcome_state),
                "gose": as_float(item["Glasgow Outcome Scale Extended"]),
                "pet_mibh": as_float(item["Metabolic index of the best preserved hemisphere"]),
                "eeg_svm_oos": as_float(item["Out-of-sample SVM prediction"]),
                "local_global": item["Local-Global paradigm"],
                "local_global_score": local_global_score(item["Local-Global paradigm"]),
            }
        )

    for field, zfield in [
        ("pet_mibh", "pet_z"),
        ("eeg_svm_oos", "eeg_z"),
        ("local_global_score", "local_global_z"),
    ]:
        values = np.array([float(row[field]) for row in rows], dtype=float)
        z = zscore(values)
        for row, value in zip(rows, z):
            row[zfield] = float(value) if np.isfinite(value) else math.nan

    for row in rows:
        pair = np.array([row["pet_z"], row["eeg_z"]], dtype=float)
        triplet = np.array([row["pet_z"], row["eeg_z"], row["local_global_z"]], dtype=float)
        row["gcc_pet_eeg_proxy"] = float(np.nanmean(pair)) if np.isfinite(pair).all() else math.nan
        row["gcc_pet_eeg_lg_proxy"] = float(np.nanmean(triplet)) if np.isfinite(triplet).all() else math.nan

    y_diag = np.array([row["baseline_mcs_binary"] for row in rows], dtype=float)
    y_order = np.array([row["baseline_order"] for row in rows], dtype=float)
    y_outcome = np.array([row["outcome_command_following"] for row in rows], dtype=float)

    scores = {
        "PET MIBH": np.array([row["pet_mibh"] for row in rows], dtype=float),
        "EEG SVM p(MCS)": np.array([row["eeg_svm_oos"] for row in rows], dtype=float),
        "Local-global ordinal": np.array([row["local_global_score"] for row in rows], dtype=float),
        "GCC proxy PET+EEG": np.array([row["gcc_pet_eeg_proxy"] for row in rows], dtype=float),
        "GCC proxy PET+EEG+LG": np.array([row["gcc_pet_eeg_lg_proxy"] for row in rows], dtype=float),
    }

    rng = np.random.default_rng(RANDOM_STATE)
    diagnostic = [model_row(name, y_diag, score, rng) for name, score in scores.items()]
    outcome = [model_row(name, y_outcome, score, rng) for name, score in scores.items()]
    ordinal = {name: spearman(y_order, score) for name, score in scores.items()}
    rng_delta = np.random.default_rng(RANDOM_STATE + 101)
    paired_auc_deltas = [
        paired_auc_delta(
            "Baseline MCS-/MCS+ vs VS/UWS",
            y_diag,
            "GCC proxy PET+EEG",
            scores["GCC proxy PET+EEG"],
            "PET MIBH",
            scores["PET MIBH"],
            rng_delta,
        ),
        paired_auc_delta(
            "Baseline MCS-/MCS+ vs VS/UWS",
            y_diag,
            "GCC proxy PET+EEG",
            scores["GCC proxy PET+EEG"],
            "EEG SVM p(MCS)",
            scores["EEG SVM p(MCS)"],
            rng_delta,
        ),
        paired_auc_delta(
            "6-month MCS+/Conscious vs Other",
            y_outcome,
            "GCC proxy PET+EEG",
            scores["GCC proxy PET+EEG"],
            "PET MIBH",
            scores["PET MIBH"],
            rng_delta,
        ),
        paired_auc_delta(
            "6-month MCS+/Conscious vs Other",
            y_outcome,
            "GCC proxy PET+EEG",
            scores["GCC proxy PET+EEG"],
            "EEG SVM p(MCS)",
            scores["EEG SVM p(MCS)"],
            rng_delta,
        ),
    ]

    pet_threshold = optimal_threshold(y_diag, scores["PET MIBH"])["threshold"]
    gcc_threshold = optimal_threshold(y_diag, scores["GCC proxy PET+EEG"])["threshold"]

    for row in rows:
        pet_ok = bool(np.isfinite(row["pet_mibh"]) and row["pet_mibh"] >= pet_threshold)
        eeg_ok = bool(np.isfinite(row["eeg_svm_oos"]) and row["eeg_svm_oos"] >= 0.5)
        row["pet_positive_youden"] = int(pet_ok)
        row["eeg_positive_050"] = int(eeg_ok)
        if not np.isfinite(row["eeg_svm_oos"]):
            gate = "missing_eeg"
        elif pet_ok and eeg_ok:
            gate = "concordant_high"
        elif pet_ok and not eeg_ok:
            gate = "capacity_only"
        elif (not pet_ok) and eeg_ok:
            gate = "dynamics_only"
        else:
            gate = "concordant_low"
        row["gcc_gate"] = gate

    baseline_vsuws = [row for row in rows if row["baseline_state"] == "VS/UWS"]
    vsuws_concordant_high = [row["patient_id"] for row in baseline_vsuws if row["gcc_gate"] == "concordant_high"]
    vsuws_any_positive = [
        row["patient_id"]
        for row in baseline_vsuws
        if row["gcc_gate"] in {"concordant_high", "capacity_only", "dynamics_only"}
    ]

    write_features(rows)
    plot_diagnostic(rows)
    plot_outcome(rows, pet_threshold)

    summary = {
        "source_csv": repo_path(CSV_PATH),
        "n_total": len(rows),
        "baseline_counts": group_counts(rows, "baseline_state"),
        "outcome_counts": group_counts(rows, "outcome_state"),
        "diagnostic_tests": diagnostic,
        "outcome_tests": outcome,
        "paired_auc_delta_tests": paired_auc_deltas,
        "ordinal_baseline_spearman": ordinal,
        "pet_threshold_youden_for_mcs_vs_vsuws": pet_threshold,
        "eeg_threshold_fixed": 0.5,
        "gcc_threshold_youden_for_mcs_vs_vsuws": gcc_threshold,
        "gcc_gate_summary": summarize_gates(rows),
        "baseline_vsuws_concordant_high_patients": vsuws_concordant_high,
        "baseline_vsuws_any_pet_or_eeg_positive_patients": vsuws_any_positive,
        "outputs": {
            "features_csv": repo_path(OUT_FEATURES),
            "summary_json": repo_path(OUT_SUMMARY),
            "report_md": repo_path(OUT_REPORT),
            "diagnostic_figure": repo_path(OUT_DIAG_FIG),
            "outcome_figure": repo_path(OUT_OUTCOME_FIG),
        },
        "interpretation_guardrail": (
            "This is a GCC proxy analysis of post-processed biomarkers. "
            "It does not extract GCC observables from raw EEG/PET time series."
        ),
    }

    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    diag_lines = []
    for item in diagnostic:
        ci = item["auc_ci95"]
        diag_lines.append(
            f"| {item['score']} | {item['n']} | {item['auc']:.3f} | "
            f"{ci[0]:.3f}-{ci[1]:.3f} | {item['mannwhitney_p']:.4g} |"
        )
    outcome_lines = []
    for item in outcome:
        ci = item["auc_ci95"]
        outcome_lines.append(
            f"| {item['score']} | {item['n']} | {item['auc']:.3f} | "
            f"{ci[0]:.3f}-{ci[1]:.3f} | {item['mannwhitney_p']:.4g} |"
        )

    delta_lines = []
    for item in paired_auc_deltas:
        ci = item["delta_auc_ci95"]
        delta_lines.append(
            f"| {item['target']} | {item['score_a']} - {item['score_b']} | "
            f"{item['n']} | {item['delta_auc']:+.3f} | "
            f"{ci[0]:+.3f} to {ci[1]:+.3f} | "
            f"{item['bootstrap_p_two_sided']:.4g} |"
        )

    gate_lines = []
    for item in summary["gcc_gate_summary"]:
        gate_lines.append(
            f"| {item['gate']} | {item['n']} | {item['outcome_command_following_rate']:.3f} | "
            f"{item['patients']} |"
        )

    report = f"""# Hermann 2021 GCC Proxy Analysis

This analysis uses the public Hermann et al. 2021 supplementary metadata table.
Raw EEG and FDG-PET recordings are not public, so this is not a raw-signal GCC
feature extraction. It is a conservative proxy analysis:

- PET MIBH is treated as metabolic capacity / preserved substrate.
- Out-of-sample EEG SVM p(MCS) is treated as dynamic access evidence.
- Local-global response is used only as a secondary ordinal task-response proxy.
- GCC PET+EEG proxy is the mean of z-scored PET MIBH and EEG SVM p(MCS).

## Cohort

- Total rows: {len(rows)}
- Baseline: {summary['baseline_counts']}
- 6-month outcome: {summary['outcome_counts']}

## Diagnostic Target: Baseline MCS-/MCS+ vs VS/UWS

| Score | N | AUC | 95% bootstrap CI | Mann-Whitney p |
|---|---:|---:|---:|---:|
{chr(10).join(diag_lines)}

PET Youden threshold for this table: {pet_threshold:.3f}

## Prognostic Target: 6-Month MCS+/Conscious vs Other

| Score | N | AUC | 95% bootstrap CI | Mann-Whitney p |
|---|---:|---:|---:|---:|
{chr(10).join(outcome_lines)}

## Paired AUC-Delta Tests

The table below uses paired patient-level bootstrap resampling over complete
cases for both compared scores. It tests whether the descriptive AUC advantage
of the GCC PET+EEG proxy is stable against paired sampling uncertainty.

| Target | Comparison | N | Delta AUC | 95% paired bootstrap CI | Two-sided bootstrap p |
|---|---|---:|---:|---:|---:|
{chr(10).join(delta_lines)}

## GCC Gate Summary

PET is thresholded at the diagnostic Youden threshold above. EEG is thresholded
at the out-of-sample SVM decision boundary of 0.50.

| Gate | N | Command-following outcome rate | Patients |
|---|---:|---:|---|
{chr(10).join(gate_lines)}

Baseline VS/UWS patients with concordant PET+EEG high evidence:
{vsuws_concordant_high}

Baseline VS/UWS patients with any PET or EEG positive evidence:
{vsuws_any_positive}

## Outputs

- Features: `{repo_path(OUT_FEATURES)}`
- Summary: `{repo_path(OUT_SUMMARY)}`
- Diagnostic figure: `{repo_path(OUT_DIAG_FIG)}`
- Outcome figure: `{repo_path(OUT_OUTCOME_FIG)}`
"""
    OUT_REPORT.write_text(report, encoding="utf-8")

    print(json.dumps({
        "n_total": summary["n_total"],
        "diagnostic": diagnostic,
        "outcome": outcome,
        "paired_auc_deltas": paired_auc_deltas,
        "pet_threshold": pet_threshold,
        "gcc_threshold": gcc_threshold,
        "vsuws_concordant_high": vsuws_concordant_high,
        "features": str(OUT_FEATURES),
        "report": str(OUT_REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
