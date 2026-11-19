# ========================================
# CVMS Classification Pipeline
# ========================================
# Reproduces the 1,268-case consensus cohort analysis across Six-, Five-, and
# Three-Class CVMS formats with fixed random seeds across all stochastic steps.
#
# Key details:
#   * Consensus cohort: rows with 'Level of dentists agreement' == 'Disagreement'
#     are excluded if present (1,380 -> 1,268 cases).
#   * Data split: single unstratified 70/30 split (random_state=7; n_train=887,
#     n_test=381) shared across all three classification formats so every
#     format is evaluated on the identical 381 held-out radiographs.
#   * Preprocessing: +/-inf replaced by 0; StandardScaler fitted on the
#     TRAINING split only and applied unchanged to the test split (no test-set
#     leakage), with NaN filled with 0.
#   * Feature selection: Random Forest (10,000 trees, random_state=0) fitted on
#     the training partition only; SelectFromModel(threshold=0.015, prefit=True).
#     Full-51 vs. selected-feature RF comparison tunes RF separately on each
#     feature set via BayesSearchCV.
#   * Models (4): SVM, RForest, LogReg (BayesSearchCV, 50 iterations, 10-fold
#     RepeatedKFold, random_state=0) and MLP (GridSearchCV, 10-fold CV,
#     random_state=0).
#   * Evaluation: all test metrics, confusion matrices, per-case outputs, and
#     statistical tests compare model predictions against ground-truth consensus
#     labels (y_test).
#
# Usage:
#   python project2_CVMS.py [data.csv] [output_dir]

import os
import sys
import json
import warnings
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from matplotlib import pyplot as plt
from scipy.stats import chi2
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ========================================
# Configuration
# ========================================

NAME_TYPE = "Voting"
MAX_ITER = 1000
TEST_SIZE = 0.3
SPLIT_SEED = 7
MODEL_SEED = 0
CV_SEED = 1
SEL_THRESHOLD = 0.015
LABEL_COLS = ["Six_Class", "Five_Class", "Three_Class"]
MODELS = ["SVM", "RForest", "LogReg", "MLP"]


# ========================================
# Function Definitions
# ========================================

def save_confusion_matrix(y_true, y_pred, labels, save_path, title="Confusion Matrix"):
    """Generate and save confusion matrix plot."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        annot_kws={"size": 12},
        cbar=False,
        square=True,
        cmap="Reds",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted label", fontsize=12)
    plt.ylabel("True label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def cochrans_q_test(X: np.ndarray):
    """
    Cochran's Q test for multiple models evaluated on the same cases.
    X: (n_subjects, k_models) binary matrix; 1 = correct, 0 = incorrect.
    Returns (Q_stat, df, p_value).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_subjects x k_models)")
    n, k = X.shape
    C = X.sum(axis=0)  # per-model sums
    R = X.sum(axis=1)  # per-subject sums
    T = C.sum()
    num = k * (k - 1) * ((C ** 2).sum() - (k * (T ** 2) / n))
    den = (k * T) - (R ** 2).sum()
    if den == 0:
        return np.nan, k - 1, np.nan
    Q = num / den
    p = 1.0 - chi2.cdf(Q, df=k - 1)
    return float(Q), k - 1, float(p)


def load_dataset(csv_path: str):
    """Load consensus dataset and extract the 51 feature columns."""
    df = pd.read_csv(csv_path)
    if "Level of dentists agreement" in df.columns:
        df = df[df["Level of dentists agreement"] != "Disagreement"]
    df = df.reset_index(drop=True)

    if "No." in df.columns:
        case_ids = df["No."].values
    elif "Order" in df.columns:
        case_ids = df["Order"].values
    else:
        case_ids = df.index.values

    start = list(df.columns).index("Gender")
    feature_cols = list(df.columns[start:start + 51])
    if len(feature_cols) != 51 or "Age month" not in feature_cols:
        raise ValueError(f"Expected 51 feature columns starting at 'Gender', got {len(feature_cols)}")

    missing_labels = [c for c in LABEL_COLS if c not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing target label columns: {missing_labels}")

    return df, feature_cols, case_ids


def get_algorithms_and_spaces():
    """Return seeded estimators and hyperparameter search spaces."""
    return [
        (
            "SVM",
            SVC(probability=True, random_state=MODEL_SEED),
            {
                "C": [0.1, 1, 10, 100, 200],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto", 0.01, 0.001],
                "degree": [2, 3, 4],
                "coef0": [0, 0.1, 0.5],
            },
        ),
        (
            "RForest",
            RandomForestClassifier(random_state=MODEL_SEED),
            {
                "max_depth": Integer(5, 30),
                "min_samples_split": Integer(2, 10),
                "n_estimators": Integer(1, 100),
                "max_features": Categorical(["sqrt", "log2"]),
                "min_samples_leaf": Integer(1, 5),
                "bootstrap": Categorical([True, False]),
            },
        ),
        (
            "LogReg",
            LogisticRegression(max_iter=MAX_ITER),
            {
                "penalty": Categorical(["l2"]),
                "C": Real(1e-4, 1e4, prior="log-uniform"),
                "solver": Categorical(["lbfgs", "newton-cg", "liblinear", "sag", "saga"]),
            },
        ),
        (
            "MLP",
            MLPClassifier(max_iter=MAX_ITER, random_state=MODEL_SEED),
            {
                "hidden_layer_sizes": [
                    (31, 16), (31, 16, 8), (22, 11), (22, 11, 5),
                    (20, 10), (20, 10, 5), (31,), (22,), (20,), (16, 8), (10, 5),
                ],
                "activation": ["tanh", "relu"],
                "solver": ["sgd", "adam"],
                "alpha": [0.0001, 0.05],
                "learning_rate": ["constant", "adaptive"],
            },
        ),
    ]


# ========================================
# Main Script
# ========================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, "project2_CVMS.csv")
    csv_path = sys.argv[1] if len(sys.argv) > 1 else (
        "project2_CVMS.csv" if os.path.exists("project2_CVMS.csv") else default_csv
    )
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()

    # Output directories
    dirs = {
        "model": os.path.join(out_dir, "Model"),
        "params": os.path.join(out_dir, "BestParameters"),
        "cm": os.path.join(out_dir, "ConfusionMatrix"),
        "report": os.path.join(out_dir, "Classification_report"),
        "acc": os.path.join(out_dir, "ModelAccuracy"),
        "sel": os.path.join(out_dir, "SelectedFeatures"),
        "pred": os.path.join(out_dir, "PredictedResults"),
        "stats": os.path.join(out_dir, "StatisticalTests"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Load data
    print(f"Loading data from {csv_path}...")
    dataframe, feature_cols, case_ids = load_dataset(csv_path)
    feature = dataframe[feature_cols].replace([np.inf, -np.inf], 0)
    print(f"Loaded {len(dataframe)} consensus cases and {len(feature_cols)} features.")

    # Single unstratified 70/30 split reused across all three formats
    train_idx, test_idx = train_test_split(
        np.arange(len(dataframe)), test_size=TEST_SIZE, random_state=SPLIT_SEED
    )
    test_case_ids = case_ids[test_idx]
    print(f"Train/test split: n_train={len(train_idx)}, n_test={len(test_idx)} (random_state={SPLIT_SEED})")

    # Fit StandardScaler on the training split only and apply unchanged to test
    scaler = StandardScaler().fit(feature.iloc[train_idx])
    scale_df = pd.DataFrame(scaler.transform(feature), columns=feature_cols).fillna(0)

    # ========================================
    # Training, Evaluation & Per-Case Saving
    # ========================================
    comparison_data = []

    for idx, tlabel in enumerate(tqdm(LABEL_COLS, desc="Formats"), 1):
        print(f"\nProcessing format: {tlabel}")
        label = dataframe[tlabel].reset_index(drop=True)

        if tlabel == "Three_Class":
            labels = ["Pre-pubertal", "Pubertal", "Post-pubertal"]
        else:
            labels = sorted(label.unique().tolist())

        X_train = scale_df.iloc[train_idx]
        X_test = scale_df.iloc[test_idx]
        y_train = label.iloc[train_idx]
        y_test = label.iloc[test_idx]

        # 1. Feature selection (training split only, prefit=True avoids redundant RF fit)
        rf_selector = RandomForestClassifier(n_estimators=10000, random_state=MODEL_SEED, n_jobs=-1)
        rf_selector.fit(X_train, y_train)
        sfm = SelectFromModel(rf_selector, threshold=SEL_THRESHOLD, prefit=True)
        selected_features = list(scale_df.columns[sfm.get_support()])
        X_train_sfm = sfm.transform(X_train)
        X_test_sfm = sfm.transform(X_test)

        pd.DataFrame(selected_features, columns=["Selected_Feature"]).to_csv(
            os.path.join(dirs["sel"], f"selected_feat_{tlabel}.csv"), index=False
        )
        with open(os.path.join(dirs["sel"], f"sel_all_{tlabel}.json"), "w", encoding="utf-8") as f:
            json.dump(selected_features, f, indent=2)

        # 2. Full 51-feature tuned Random Forest baseline
        _, rf_est, rf_space = [a for a in get_algorithms_and_spaces() if a[0] == "RForest"][0]
        rf_full_search = BayesSearchCV(
            estimator=rf_est,
            search_spaces=rf_space,
            n_jobs=-1,
            cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=CV_SEED),
            scoring="accuracy",
            random_state=MODEL_SEED,
        )
        rf_full_search.fit(X_train.values, y_train)
        full_acc = float(accuracy_score(y_test, rf_full_search.predict(X_test.values)))
        print(f"  Tuned RF full features (51) test accuracy: {full_acc:.4f}")

        # 3. Train and evaluate the 4 classifiers on selected features
        model_scores = []
        pred_summary = pd.DataFrame({"idx": test_idx, "No.": test_case_ids, "y": y_test.values})
        per_case_all = []
        sel_rf_acc = None

        for model_name, model, param in get_algorithms_and_spaces():
            print(f"  Training {model_name}...")
            cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=CV_SEED)
            if model_name == "MLP":
                clf_search = GridSearchCV(model, param, n_jobs=-1, cv=10)
            else:
                clf_search = BayesSearchCV(
                    estimator=model,
                    search_spaces=param,
                    n_jobs=-1,
                    cv=cv,
                    scoring="accuracy",
                    random_state=MODEL_SEED,
                )

            clf_search.fit(X_train_sfm, y_train)
            cv_acc = float(clf_search.best_score_)

            # Save trained model
            model_file = os.path.join(dirs["model"], f"{NAME_TYPE}_{tlabel}_{model_name}.joblib")
            dump(clf_search, model_file)

            # Save best hyperparameters (sorted keys)
            best_params = {k: clf_search.best_params_[k] for k in sorted(clf_search.best_params_)}
            param_file = os.path.join(dirs["params"], f"{NAME_TYPE}_{tlabel}_{model_name}_bestPara.csv")
            pd.DataFrame.from_dict(best_params, orient="index").to_csv(param_file)

            # Evaluate strictly against ground-truth consensus labels (y_test)
            y_pred_test = clf_search.predict(X_test_sfm)
            test_acc = float(accuracy_score(y_test, y_pred_test))
            if model_name == "RForest":
                sel_rf_acc = test_acc
            pct_diff = ((cv_acc - test_acc) / cv_acc) * 100.0
            overfit = cv_acc - test_acc
            print(f"    {model_name}: 10-fold CV={cv_acc:.4f}, test={test_acc:.4f}, gap={abs(overfit):.4f}")

            model_scores.append([model_name, cv_acc, test_acc, pct_diff, overfit])
            pred_summary[model_name] = y_pred_test

            # Save per-case predictions with true consensus label (y_test)
            result_df = pd.DataFrame(X_test_sfm, columns=selected_features)
            result_df["Label"] = y_test.values
            result_df["Prediction"] = y_pred_test
            result_df["No."] = test_case_ids
            raw_out = os.path.join(dirs["pred"], f"{NAME_TYPE}_{tlabel}_{model_name}_resultsRaw.csv")
            result_df.to_csv(raw_out, index=False)

            per_case = pd.DataFrame({
                "Stage": tlabel,
                "Model": model_name,
                "CaseID": test_case_ids,
                "Label": y_test.values,
                "Prediction": y_pred_test,
                "Correct": (y_test.values == y_pred_test).astype(int),
            })
            per_case_all.append(per_case)

            # Confusion matrix & classification report
            cm_path = os.path.join(dirs["cm"], f"{NAME_TYPE}_{tlabel}_{model_name}.png")
            save_confusion_matrix(
                y_test, y_pred_test, labels, cm_path, title=f"Confusion Matrix ({tlabel} - {model_name})"
            )

            report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
            rep_path = os.path.join(dirs["report"], f"{NAME_TYPE}_{tlabel}_{model_name}_class_report.csv")
            pd.DataFrame(report).transpose().to_csv(rep_path)

        comparison_data.append({
            "Class": tlabel,
            "Number of Full Features": X_train.shape[1],
            "Accuracy (Full Features)": full_acc,
            "Number of Selected Features": X_train_sfm.shape[1],
            "Accuracy (Selected Features)": sel_rf_acc,
        })

        # Save model accuracy summary
        dscore = pd.DataFrame(
            model_scores,
            columns=["classifier", "cv_accuracy", "testing_accuracy", "percent_difference", "overfitting_measure"],
        ).sort_values("testing_accuracy", ascending=False)
        dscore.to_csv(os.path.join(dirs["acc"], f"{NAME_TYPE}_CVMS_{tlabel}.csv"), index=False)

        # Save consolidated test predictions (compatible with supplementary_analyses.py)
        pred_summary.to_csv(os.path.join(dirs["pred"], f"pred_all_{tlabel}.csv"), index=False)

        # Save per-case pivot table
        per_case_df = pd.concat(per_case_all, ignore_index=True)
        per_case_df.to_csv(os.path.join(dirs["pred"], f"ALLMODELS_{tlabel}_percase.csv"), index=False)
        pivot_df = per_case_df.pivot_table(index="CaseID", columns="Model", values="Prediction", aggfunc="first")
        pivot_df.columns = [f"Pred_{c}" for c in pivot_df.columns]
        labels_df = per_case_df.drop_duplicates(subset=["CaseID"])[["CaseID", "Label"]].set_index("CaseID")
        pivot_df = labels_df.join(pivot_df)
        pivot_df.to_csv(os.path.join(dirs["pred"], f"PIVOT_{tlabel}_percase.csv"))

        # ========================================
        # Statistical Tests (Cochran's Q & Post-hoc McNemar with Holm correction)
        # ========================================
        model_cols = [f"Pred_{m}" for m in MODELS]
        sub = pivot_df[["Label"] + model_cols].dropna()
        correct_mat = np.column_stack([(sub[c].values == sub["Label"].values).astype(int) for c in model_cols])
        n_cases, k_models = correct_mat.shape

        Q, df_q, p_q = cochrans_q_test(correct_mat)
        print(f"  Cochran's Q ({tlabel}, n={n_cases}, k={k_models}): Q={Q:.4f}, df={df_q}, p={p_q:.6g}")

        pairs = list(combinations(range(k_models), 2))
        raw_pvals = []
        pair_names = []
        for i, j in pairs:
            a, b = correct_mat[:, i], correct_mat[:, j]
            table = [
                [int(((a == 1) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())],
                [int(((a == 0) & (b == 1)).sum()), int(((a == 0) & (b == 0)).sum())],
            ]
            res = mcnemar(table, exact=False, correction=True)
            raw_pvals.append(float(res.pvalue))
            pair_names.append((MODELS[i], MODELS[j]))

        # Holm-Bonferroni correction
        m_tests = len(raw_pvals)
        order = np.argsort(raw_pvals)
        adj_p = np.empty(m_tests, dtype=float)
        for rank, pair_idx in enumerate(order, start=1):
            adj = (m_tests - rank + 1) * raw_pvals[pair_idx]
            if rank == 1:
                adj_p[pair_idx] = adj
            else:
                adj_p[pair_idx] = max(adj_p[order[rank - 2]], adj)

        posthoc_df = pd.DataFrame({
            "Stage": tlabel,
            "Model_A": [pair_names[i][0] for i in range(m_tests)],
            "Model_B": [pair_names[i][1] for i in range(m_tests)],
            "p_raw": raw_pvals,
            "p_Holm": np.minimum(adj_p, 1.0),
        }).sort_values("p_Holm").reset_index(drop=True)
        posthoc_df.to_csv(os.path.join(dirs["stats"], f"mcnemar_{tlabel}.csv"), index=False)

    # Save full vs selected feature comparison
    pd.DataFrame(comparison_data).to_csv(
        os.path.join(dirs["sel"], "comparison_selectedFeature.csv"), index=False
    )
    print("\nAll phases completed.")


if __name__ == "__main__":
    main()
