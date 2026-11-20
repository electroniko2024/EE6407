"""Independent verification of saved results; run after both solution.py files.

Dependencies: numpy, pandas, openpyxl, scipy, scikit-learn.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ROOT = Path(__file__).resolve().parent


def verify():
    r1 = json.loads((ROOT / "Assignment1/results/results.json").read_text(encoding="utf-8"))
    r2 = json.loads((ROOT / "Assignment2/results/results.json").read_text(encoding="utf-8"))
    for folder, result in [("Assignment1", r1), ("Assignment2", r2)]:
        for name, digest in result["inputs_sha256"].items():
            assert hashlib.sha256((ROOT / folder / name).read_bytes()).hexdigest() == digest

    # Independently reproduce deletion from the original workbook.
    original = pd.read_excel(ROOT / "Assignment1/TrainingData (2).xlsx", header=None)
    numeric = original.replace("?", np.nan).astype(float)
    complete = numeric.dropna()
    quartiles = complete.iloc[:, :4].quantile([.25, .75], interpolation="linear")
    low = quartiles.loc[.25] - 1.5 * (quartiles.loc[.75] - quartiles.loc[.25])
    high = quartiles.loc[.75] + 1.5 * (quartiles.loc[.75] - quartiles.loc[.25])
    mask = ((complete.iloc[:, :4] >= low) & (complete.iloc[:, :4] <= high)).all(axis=1)
    expected = complete.loc[mask]
    saved = pd.read_csv(ROOT / "Assignment1/results/cleaned_training.csv")
    assert np.array_equal(expected.index.to_numpy()+1, saved.original_excel_row.to_numpy())
    assert np.allclose(expected.iloc[:, :4], saved[["A", "B", "C", "D"]])
    assert np.array_equal(expected.iloc[:, 4].astype(int), saved.class_label)
    xt = pd.read_excel(ROOT / "Assignment1/TestData.xlsx", header=None).to_numpy(float)
    # Disable smoothing to compare with the explicitly reported MLE variances.
    nb = GaussianNB(var_smoothing=0.0).fit(expected.iloc[:, :4].to_numpy(), expected.iloc[:, 4].to_numpy(int))
    assert np.array_equal(nb.classes_, r1["class_order"])
    assert np.allclose(nb.theta_, r1["means"], rtol=1e-12, atol=1e-12)
    assert np.allclose(nb.var_, r1["variances_mle"], rtol=1e-12, atol=1e-12)
    assert np.array_equal(nb.predict(xt), r1["predicted_labels"])
    assert np.allclose(nb.predict_proba(xt), r1["posteriors"], rtol=1e-10, atol=1e-12)
    nb_table = pd.read_csv(ROOT / "Assignment1/results/test_predictions.csv")
    assert np.array_equal(nb_table.test_sample, np.arange(1, 31))
    assert np.array_equal(nb_table.predicted_label, r1["predicted_labels"])
    assert np.allclose(nb_table[["A", "B", "C", "D"]], xt)
    assert np.allclose(nb_table[[f"g_class_{c}" for c in [1, 2, 3]]], r1["log_scores"])

    x = loadmat(ROOT / "Assignment2/data_train.mat")["data_train"]
    y = loadmat(ROOT / "Assignment2/label_train.mat")["label_train"].ravel()
    xt2 = loadmat(ROOT / "Assignment2/data_test.mat")["data_test"]
    # Equal priors match the midpoint convention. SVD LDA uses Sw/(N-2).
    lda = LinearDiscriminantAnalysis(solver="svd", priors=[.5, .5]).fit(x, y)
    scale = len(x) - len(np.unique(y))
    assert np.array_equal(lda.classes_, [-1, 1])
    assert np.allclose(lda.coef_[0], scale*np.asarray(r2["w"]), rtol=1e-10, atol=1e-12)
    assert np.allclose(lda.intercept_[0], scale*r2["w0"], rtol=1e-10, atol=1e-12)
    assert np.array_equal(lda.predict(xt2), r2["predicted_labels"])
    assert np.allclose(lda.decision_function(xt2), scale*np.asarray(r2["test_scores"]), rtol=1e-10, atol=1e-12)
    f_table = pd.read_csv(ROOT / "Assignment2/results/test_predictions.csv")
    assert np.array_equal(f_table.test_sample, np.arange(1, 24))
    assert np.allclose(f_table[[f"x{i}" for i in range(1, 6)]], xt2)
    assert np.array_equal(f_table.predicted_label, r2["predicted_labels"])
    assert np.allclose(f_table.fisher_score, r2["test_scores"])
    # Check that the displayed precision cannot flip any test prediction.
    rounded_w = np.round(r2["w"], 12)
    rounded_bias = round(r2["w0"], 12)
    assert np.array_equal(np.where(xt2 @ rounded_w + rounded_bias >= 0, 1, -1), r2["predicted_labels"])

    verification = {
        "status": "passed", "original_input_hashes": "unchanged",
        "assignment1": {"cleaning_independent_reproduction": "passed",
                        "GaussianNB_var_smoothing": 0.0, "parameters_probabilities_predictions": "passed",
                        "prediction_rows": 30, "test_accuracy_available": False},
        "assignment2": {"LDA_solver": "svd", "LDA_priors": [.5, .5], "coefficient_scale": scale,
                        "weights_bias_scores_predictions": "passed", "prediction_rows": 23,
                        "linear_solve_relative_residual": r2["linear_solve_relative_residual"],
                        "reported_12_decimal_coefficients_preserve_predictions": True,
                        "test_accuracy_available": False},
        "versions": {},
    }
    import scipy, sklearn
    verification["versions"] = {"numpy": np.__version__, "pandas": pd.__version__,
                                 "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}
    (ROOT / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return verification


if __name__ == "__main__":
    verify()
