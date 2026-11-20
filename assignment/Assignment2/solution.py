"""EE6407 Assignment 2: Fisher linear discriminant with midpoint threshold.

Run: python solution.py
Dependencies: numpy, scipy. Original MATLAB files are read-only.
"""
from pathlib import Path
import hashlib
import json
import csv
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent


def solve():
    x = loadmat(ROOT / "data_train.mat")["data_train"].astype(float)
    y = loadmat(ROOT / "label_train.mat")["label_train"].reshape(-1)
    xt = loadmat(ROOT / "data_test.mat")["data_test"].astype(float)
    assert x.shape == (577, 5) and y.shape == (577,) and xt.shape == (23, 5)
    assert set(y) == {-1, 1} and np.isfinite(x).all() and np.isfinite(xt).all()
    xp, xn = x[y == 1], x[y == -1]
    mp, mn = xp.mean(axis=0), xn.mean(axis=0)
    cp, cn = xp - mp, xn - mn
    sp, sn = cp.T @ cp, cn.T @ cn
    sw = sp + sn
    delta = mp - mn
    assert np.linalg.matrix_rank(sw) == 5
    # Solve the linear system rather than explicitly computing an inverse.
    w = np.linalg.solve(sw, delta)
    w0 = float(-0.5 * w @ (mp + mn))
    scores = xt @ w + w0
    predictions = np.where(scores >= 0, 1, -1)
    residual = float(np.linalg.norm(sw @ w - delta) / np.linalg.norm(delta))
    assert residual < 1e-12
    unit_w = w / np.linalg.norm(w)
    unit_bias = w0 / np.linalg.norm(w)
    assert np.array_equal(predictions, np.where(xt @ unit_w + unit_bias >= 0, 1, -1))
    midpoint = float(0.5 * w @ (mp + mn))
    assert float(mp @ w + w0) > 0 and float(mn @ w + w0) < 0
    training_pred = np.where(x @ w + w0 >= 0, 1, -1)
    confusion = [[int(np.sum((y == a) & (training_pred == b))) for b in [-1, 1]] for a in [-1, 1]]
    pooled_w = (len(x) - 2) * w
    pooled_bias = (len(x) - 2) * w0
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["test_sample", "x1", "x2", "x3", "x4", "x5", "fisher_score", "predicted_label"])
        for i, (features, score, label) in enumerate(zip(xt, scores, predictions), 1):
            writer.writerow([i, *features.tolist(), float(score), int(label)])
    result = {
        "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT / "data_train.mat", ROOT / "label_train.mat", ROOT / "data_test.mat"]},
        "training_shape": list(x.shape), "test_shape": list(xt.shape),
        "class_counts": {"+1": len(xp), "-1": len(xn)},
        "mean_positive": mp.tolist(), "mean_negative": mn.tolist(),
        "scatter_positive": sp.tolist(), "scatter_negative": sn.tolist(), "within_scatter": sw.tolist(),
        "within_scatter_condition_number": float(np.linalg.cond(sw)),
        "w": w.tolist(), "w0": w0, "projected_midpoint": midpoint,
        "projected_positive_mean": float(mp @ w), "projected_negative_mean": float(mn @ w),
        "unit_w": unit_w.tolist(), "unit_w0": float(unit_bias),
        "pooled_covariance_w": pooled_w.tolist(), "pooled_covariance_w0": pooled_bias,
        "threshold_convention": "midpoint of projected class means; equal priors and equal error costs; g>=0 predicts +1",
        "linear_solve_relative_residual": residual,
        "test_features": xt.tolist(), "test_scores": scores.tolist(), "predicted_labels": predictions.tolist(),
        "training_confusion_label_order": [-1, 1], "training_confusion": confusion,
        "training_resubstitution_accuracy": float(np.mean(training_pred == y)), "test_accuracy": None,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    np.savez(output / "model.npz", w=w, w0=w0, mean_positive=mp, mean_negative=mn, within_scatter=sw)
    print(json.dumps({"class_counts": result["class_counts"], "w": w.tolist(), "w0": w0,
                      "predictions": predictions.tolist(), "relative_residual": residual}))
    return result


if __name__ == "__main__":
    solve()
