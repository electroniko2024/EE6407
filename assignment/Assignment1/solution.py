"""EE6407 Assignment 1: auditable Gaussian Naive Bayes solution.

Run: python solution.py
Dependencies: numpy, pandas, openpyxl. Original inputs are read-only.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES = ["A", "B", "C", "D"]


def fit_gaussian_nb(x, y):
    classes, counts = np.unique(y, return_counts=True)
    means = np.array([x[y == c].mean(axis=0) for c in classes])
    variances = np.array([x[y == c].var(axis=0, ddof=0) for c in classes])
    if np.any(variances <= 0):
        raise ValueError("Gaussian density requires strictly positive variances")
    return classes, counts / len(y), means, variances


def score_gaussian_nb(x, priors, means, variances):
    return np.log(priors)[None, :] - 0.5 * np.sum(
        np.log(2 * np.pi * variances)[None, :, :]
        + (x[:, None, :] - means[None, :, :]) ** 2 / variances[None, :, :],
        axis=2,
    )


def solve():
    source = ROOT / "TrainingData (2).xlsx"
    test_source = ROOT / "TestData.xlsx"
    raw = pd.read_excel(source, header=None)
    test_df = pd.read_excel(test_source, header=None)
    assert raw.shape == (120, 5) and test_df.shape == (30, 4)
    # Only known missing markers are permitted; unexpected text is an error.
    numeric = raw.replace({"?": np.nan}).apply(pd.to_numeric, errors="raise")
    missing = numeric.isna().any(axis=1).to_numpy()
    x_complete = numeric.loc[~missing].iloc[:, :4].to_numpy(float)
    # One pass, after deletion of incomplete rows. No test information is used.
    q1, q3 = np.quantile(x_complete, [0.25, 0.75], axis=0, method="linear")
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    x_raw = numeric.iloc[:, :4].to_numpy(float)
    flagged = (x_raw < lower) | (x_raw > upper)
    outliers = (~missing) & flagged.any(axis=1)
    keep = ~(missing | outliers)
    x = x_raw[keep]
    y = numeric.loc[keep, 4].to_numpy(int)
    assert np.isfinite(x).all() and set(y) == {1, 2, 3}
    xt = test_df.apply(pd.to_numeric, errors="raise").to_numpy(float)
    assert np.isfinite(xt).all()
    classes, priors, means, variances = fit_gaussian_nb(x, y)
    scores = score_gaussian_nb(xt, priors, means, variances)
    predictions = classes[np.argmax(scores, axis=1)]
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    posterior = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    # These are density-based model posteriors, not measured test accuracy.
    quadratic = -0.5 / variances
    linear = means / variances
    constant = np.log(priors) - 0.5 * np.sum(
        np.log(2 * np.pi * variances) + means**2 / variances, axis=1
    )
    expanded = xt**2 @ quadratic.T + xt @ linear.T + constant
    assert np.allclose(scores, expanded, atol=1e-10)
    assert len(x) == 112

    # Auxiliary sensitivity: retain the two mild B-column flags (53 and 98).
    # This is deliberately separate from the rule-defined primary model.
    severe_rows = [91, 96, 117]
    keep_aux = ~missing & ~np.isin(np.arange(1, 121), severe_rows)
    aux_model = fit_gaussian_nb(x_raw[keep_aux], numeric.loc[keep_aux, 4].to_numpy(int))
    aux_scores = score_gaussian_nb(xt, *aux_model[1:])
    aux_predictions = aux_model[0][np.argmax(aux_scores, axis=1)]

    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    audit = []
    missing_cells = []
    outlier_details = []
    for i in range(len(raw)):
        reasons = []
        for j in range(5):
            if pd.isna(numeric.iloc[i, j]):
                cell = f"{chr(65+j)}{i+1}"
                missing_cells.append({"cell": cell, "row": i + 1, "marker": str(raw.iloc[i, j])})
                reasons.append(f"missing {cell}")
        if outliers[i]:
            for j in np.flatnonzero(flagged[i]):
                detail = {"row": i + 1, "cell": f"{FEATURES[j]}{i+1}",
                          "value": float(x_raw[i, j]), "lower": float(lower[j]),
                          "upper": float(upper[j]), "label": int(numeric.iloc[i, 4])}
                outlier_details.append(detail)
                reasons.append(f"outlier {detail['cell']}={detail['value']}")
        audit.append({"original_excel_row": i+1, "status": "retained" if keep[i] else "removed",
                      "reason": "; ".join(reasons)})
    pd.DataFrame(audit).to_csv(output / "cleaning_audit.csv", index=False, encoding="utf-8-sig")
    clean = pd.DataFrame(x, columns=FEATURES)
    clean.insert(0, "original_excel_row", np.flatnonzero(keep) + 1)
    clean["class_label"] = y
    clean.to_csv(output / "cleaned_training.csv", index=False, encoding="utf-8-sig")
    prediction_table = pd.DataFrame(xt, columns=FEATURES)
    prediction_table.insert(0, "test_sample", np.arange(1, len(xt)+1))
    for j, c in enumerate(classes):
        prediction_table[f"g_class_{c}"] = scores[:, j]
        prediction_table[f"posterior_class_{c}"] = posterior[:, j]
    prediction_table["predicted_label"] = predictions
    prediction_table.to_csv(output / "test_predictions.csv", index=False, encoding="utf-8-sig")
    result = {
        "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, test_source]},
        "training_shape": list(raw.shape), "test_shape": list(xt.shape),
        "missing_cells": missing_cells, "missing_rows": (np.flatnonzero(missing)+1).tolist(),
        "outlier_rule": "one-pass featurewise Tukey 1.5 IQR after complete-case deletion; linear quantiles; strict outside bounds",
        "q1": q1.tolist(), "q3": q3.tolist(), "iqr": iqr.tolist(),
        "lower": lower.tolist(), "upper": upper.tolist(),
        "outlier_details": outlier_details, "outlier_rows": (np.flatnonzero(outliers)+1).tolist(),
        "removed_rows": (np.flatnonzero(~keep)+1).tolist(), "retained_count": int(keep.sum()),
        "class_order": classes.tolist(), "class_counts": [int((y == c).sum()) for c in classes],
        "priors": priors.tolist(), "means": means.tolist(), "variances_mle": variances.tolist(),
        "quadratic_coefficients": quadratic.tolist(), "linear_coefficients": linear.tolist(),
        "constants": constant.tolist(), "predicted_labels": predictions.tolist(),
        "test_features": xt.tolist(), "log_scores": scores.tolist(), "posteriors": posterior.tolist(),
        "sensitivity": {"retained_count": int(keep_aux.sum()), "gross_outlier_rows": severe_rows,
                        "retained_mild_flags": [53, 98], "class_counts": [int((numeric.loc[keep_aux, 4] == c).sum()) for c in classes],
                        "predicted_labels": aux_predictions.tolist(),
                        "changed_test_samples": (np.flatnonzero(aux_predictions != predictions)+1).tolist()},
        "test_accuracy": None,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    np.savez(output / "model.npz", classes=classes, priors=priors, means=means, variances=variances,
             retained_rows=np.flatnonzero(keep)+1, lower=lower, upper=upper)
    print(json.dumps({"retained": len(x), "removed_rows": result["removed_rows"],
                      "class_counts": result["class_counts"], "predictions": predictions.tolist(),
                      "sensitivity_changed": result["sensitivity"]["changed_test_samples"]}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    solve()
