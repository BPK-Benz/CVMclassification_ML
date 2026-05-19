# ========================================
# CVMS Classification Project - Reorganized
# ========================================

# Imports
import os
import glob
import pandas as pd
import numpy as np
import json
from csv import reader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils import Bunch
from sklearn import metrics

from joblib import dump, load
import joblib
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

from itertools import combinations
from scipy.stats import chi2
from statsmodels.stats.contingency_tables import mcnemar

import warnings
warnings.filterwarnings("ignore")

# ========================================
# Function Definitions
# ========================================

def confusion_matrix_graph(y_test, y_pred, labels, name_img):
    """Generate and save confusion matrix plot."""
    plt.clf()
    plt.figure(figsize=(8, 6))
    print('Confusion Matrix')
    sns.heatmap(metrics.confusion_matrix(y_test, y_pred, labels=labels),
                annot=True, fmt="d", annot_kws={"size": 12}, cbar=False,
                square=True, cmap='Reds', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(name_img, bbox_inches='tight')

def save_confusion_matrix(y_true, y_pred, labels, save_path, title="Confusion Matrix"):
    """Save confusion matrix plot for evaluation."""
    plt.clf()
    plt.figure(figsize=(7, 6))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                square=True, cmap="Reds",
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def cochrans_q_test(X: np.ndarray):
    """
    Cochran's Q test for multiple models on the same cases.
    X: (n_subjects, k_models) binary matrix; 1=correct, 0=incorrect
    Returns (Q_stat, df, p_value)
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_subjects x k_models)")
    n, k = X.shape
    C = X.sum(axis=0)  # per-model sums
    R = X.sum(axis=1)  # per-subject sums
    T = C.sum()
    num = k * (k - 1) * ((C**2).sum() - (k * (T**2) / n))
    den = (k * T) - (R**2).sum()
    if den == 0:
        return np.nan, k - 1, np.nan
    Q = num / den
    p = 1 - chi2.cdf(Q, df=k - 1)
    return Q, k - 1, p

# ========================================
# Main Script
# ========================================

# Configuration
name_type = "Voting"
max_iter = 1000
test_size = 0.3
random_state = 7

# Load and preprocess data
print("Loading data...")
dataframe = pd.read_csv(os.path.join(os.getcwd(), 'project2_CVMS.csv')).set_index('No.')
scaler = StandardScaler()
feature = dataframe.iloc[:, :51]
scale_feature = scaler.fit_transform(feature.replace(np.inf, 0))
scale_df = pd.DataFrame(scale_feature, columns=feature.columns).fillna(0)
print("Data loaded and scaled.")

# ========================================
# Training Phase
# ========================================

comparison_data = []
for idx, tlabel in enumerate(tqdm(dataframe.iloc[:, -3:].columns.tolist()), 1):
    print(f"\nProcessing class: {tlabel}")
    label = dataframe.reset_index(drop=True).loc[:, tlabel]

    # Set labels
    if len(set(label)) == 3:
        labels = ['Pre-pubertal', 'Pubertal', 'Post-pubertal']
    else:
        labels = sorted(list(set(label)))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(scale_df, label, test_size=test_size, random_state=random_state)

    # Full features model
    clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    full_acc = accuracy_score(y_test, y_pred)
    print(f'Full features accuracy: {full_acc:.4f}')

    # Feature selection
    sfm = SelectFromModel(clf, threshold=0.015)
    sfm.fit(X_train, y_train)
    selected_features = scale_df.columns[sfm.get_support()]
    X_train_sfm = sfm.transform(X_train)
    X_test_sfm = sfm.transform(X_test)

    # Save selected features
    selected_features_df = pd.DataFrame(selected_features, columns=["Selected_Feature"])
    selected_features_filename = os.path.join(os.getcwd(), 'Project2', f"selected_feat_{idx}.csv")
    selected_features_df.to_csv(selected_features_filename, index=False)

    # Selected features model
    clf_sfm = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    clf_sfm.fit(X_train_sfm, y_train)
    y_pred2 = clf_sfm.predict(X_test_sfm)
    sel_acc = accuracy_score(y_test, y_pred2)
    print(f'Selected features accuracy: {sel_acc:.4f}')

    comparison_data.append({
        'Class': tlabel,
        'Number of Full Features': X_train.shape[1],
        'Accuracy (Full Features)': full_acc,
        'Number of Selected Features': X_train_sfm.shape[1],
        'Accuracy (Selected Features)': sel_acc
    })

    # Model training with hyperparameter tuning
    algo = [
        [SVC(probability=True), 'SVM'],
        [RandomForestClassifier(), 'RForest'],
        [LogisticRegression(max_iter=max_iter), 'LogReg'],
        [MLPClassifier(max_iter=max_iter), 'MLP']
    ]

    parameters = [
        # SVM
        {'C': [0.1, 1, 10, 100, 200], 'kernel': ['linear', 'rbf', 'poly'],
         'gamma': ['scale', 'auto', 0.01, 0.001], 'degree': [2, 3, 4], 'coef0': [0, 0.1, 0.5]},
        # RandomForest
        {'max_depth': Integer(5, 30), 'min_samples_split': Integer(2, 10), 'n_estimators': Integer(1, 100),
         'max_features': Categorical(['sqrt', 'log2']), 'min_samples_leaf': Integer(1, 5), 'bootstrap': Categorical([True, False])},
        # LogisticRegression
        {'penalty': Categorical(['l2']), 'C': Real(1e-4, 1e4, prior='log-uniform'),
         'solver': Categorical(['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'])},
        # MLP
        {'hidden_layer_sizes': [(31,16), (31,16,8), (22,11), (22,11,5), (20,10), (20,10,5), (31,), (22,), (20,), (16,8), (10,5)],
         'activation': ['tanh', 'relu'], 'solver': ['sgd', 'adam'], 'alpha': [0.0001, 0.05], 'learning_rate': ['constant', 'adaptive']}
    ]

    model_scores = []
    for (model, model_name), param in zip(algo, parameters):
        print(f"Training {model_name}...")

        cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
        if model_name == 'MLP':
            clf_search = GridSearchCV(model, param, n_jobs=-1, cv=10)
        else:
            clf_search = BayesSearchCV(estimator=model, search_spaces=param, n_jobs=-1, cv=cv, scoring='accuracy')

        clf_search.fit(X_train_sfm, y_train)
        train_acc = clf_search.best_score_
        print(f'{model_name} train accuracy: {train_acc:.4f}')

        # Save model
        model_path = os.path.join(os.getcwd(), 'Project2', 'Model', f"{name_type}_{tlabel}_{model_name}.joblib")
        dump(clf_search, model_path)

        # Save best params
        param_path = os.path.join(os.getcwd(), 'Project2', 'BestParameters', f"{name_type}_{tlabel}_{model_name}_bestPara.csv")
        pd.DataFrame.from_dict(clf_search.best_params_, orient="index").to_csv(param_path)

        # Test
        y_pred = clf_search.predict(X_test_sfm)
        test_acc = clf_search.score(X_test_sfm, y_test)
        print(f'{model_name} test accuracy: {test_acc:.4f}')

        percent_diff = ((train_acc - test_acc) / train_acc) * 100
        overfit = train_acc - test_acc

        model_scores.append([model_name, train_acc, test_acc, percent_diff, overfit])

        # Confusion matrix
        cm_path = os.path.join(os.getcwd(), 'Project2', 'ConfusionMatrix', f"{name_type}_{tlabel}_{model_name}.png")
        confusion_matrix_graph(y_test, y_pred, labels, cm_path)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = os.path.join(os.getcwd(), 'Project2', 'Classification_report', f"{name_type}_{tlabel}_{model_name}_class_report.csv")
        pd.DataFrame(report).transpose().to_csv(report_path)

    # Save model accuracies
    dscore = pd.DataFrame(model_scores, columns=['classifier', 'training_accuracy', 'testing_accuracy', 'percent_difference', 'overfitting_measure'])
    final = dscore.sort_values('testing_accuracy', ascending=False)
    acc_path = os.path.join(os.getcwd(), 'Project2', 'ModelAccuracy', f"{name_type}_CVMS_{tlabel}.csv")
    final.to_csv(acc_path, index=False)

# Save comparison
comparison_df = pd.DataFrame(comparison_data)
comp_path = os.path.join(os.getcwd(), 'Project2', "comparison_selectedFeature.csv")
comparison_df.to_csv(comp_path, index=False)
print("Training phase completed.")

# ========================================
# Prediction Phase
# ========================================

print("Starting prediction phase...")
base_path = os.getcwd()
result_path = os.path.join(base_path, 'Project2', 'Predicted results')
model_path = os.path.join(base_path, 'Project2', 'Model')

index_dict = {}
for idx, tlabel in enumerate(tqdm(dataframe.iloc[:, -3:].columns.tolist()), 1):
    label = dataframe.reset_index(drop=True).loc[:, tlabel]
    unique_classes = len(set(label))

    X_train, X_test, y_train, y_test = train_test_split(scale_df, label, test_size=test_size, random_state=random_state)
    index_list = pd.concat([X_test, y_test], axis=1).index.tolist()
    index_dict[tlabel] = index_list

    # Feature selection
    clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    sfm = SelectFromModel(clf, threshold=0.015)
    sfm.fit(X_train, y_train)
    X_test_sfm = sfm.transform(X_test)
    y_pred = clf.predict(X_test)

    # Model files
    if unique_classes == 3:
        model_files = {f'Voting_Three_Class_{m}': os.path.join(model_path, f'Voting_Three_Class_{m}.joblib')
                       for m in ["GraBoost", "LogReg", "MLP", "RForest", "SVM"]}
    elif unique_classes == 5:
        model_files = {f'Voting_Five_Class_{m}': os.path.join(model_path, f'Voting_Five_Class_{m}.joblib')
                       for m in ["GraBoost", "LogReg", "MLP", "RForest", "SVM"]}
    elif unique_classes == 6:
        model_files = {f'Voting_Six_Class_{m}': os.path.join(model_path, f'Voting_Six_Class_{m}.joblib')
                       for m in ["GraBoost", "LogReg", "MLP", "RForest", "SVM"]}
    else:
        raise ValueError(f"Unsupported number of classes: {unique_classes}")

    for model_name, model_file in model_files.items():
        model = joblib.load(model_file)
        y_pred2 = model.predict(X_test_sfm)

        result_df = pd.DataFrame(X_test_sfm, columns=[f'Feature_{i}' for i in range(X_test_sfm.shape[1])])
        result_df['Label'] = y_pred
        result_df['Prediction'] = y_pred2
        result_df['No.'] = index_list

        output_filename = os.path.join(result_path, f"{model_name}_resultsRaw.csv")
        result_df.to_csv(output_filename, index=False)

print("Prediction phase completed.")

# ========================================
# Evaluation Phase
# ========================================

print("Starting evaluation phase...")
STAGE = "Three"  # Change as needed: "Three", "Five", "Six"
MODELS = ["GraBoost", "LogReg", "MLP", "RForest", "SVM"]

BASE_DIR = os.path.join(os.getcwd(), "Predicted results", "RawData")
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUT_DIR_CM = os.path.join(PARENT_DIR, f"ConfusionMatrix_fromResultsRaw_{STAGE}")
OUT_DIR_REP = os.path.join(PARENT_DIR, f"ClassificationReport_fromResultsRaw_{STAGE}")
OUT_DIR_CASE = os.path.join(PARENT_DIR, f"PerCase_fromResultsRaw_{STAGE}")
os.makedirs(OUT_DIR_CM, exist_ok=True)
os.makedirs(OUT_DIR_REP, exist_ok=True)
os.makedirs(OUT_DIR_CASE, exist_ok=True)

FILES = [f"Voting_{STAGE}_Class_{m}_resultsRaw.csv" for m in MODELS]
per_case_all = []

for fname in FILES:
    fpath = os.path.join(BASE_DIR, fname)
    if not os.path.exists(fpath):
        print(f"Skipping {fname}: file not found")
        continue

    df = pd.read_csv(fpath)
    if not {"Label", "Prediction"}.issubset(df.columns):
        print(f"Skipping {fname}: missing columns")
        continue

    case_id_col = "No." if "No." in df.columns else None
    case_id = df[case_id_col] if case_id_col else df.index

    y_true = df["Label"].values
    y_pred = df["Prediction"].values
    labels = sorted(pd.Series(y_true).unique().tolist())

    acc = accuracy_score(y_true, y_pred)
    print(f"\n{fname} - Accuracy: {acc:.4f}")

    # Save report
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(report_dict).transpose()
    rep_out = os.path.join(OUT_DIR_REP, fname.replace("_resultsRaw.csv", "_report2025.csv"))
    rep_df.to_csv(rep_out)

    # Save CM
    cm_out = os.path.join(OUT_DIR_CM, fname.replace("_resultsRaw.csv", "_cm2025.png"))
    save_confusion_matrix(y_true, y_pred, labels, cm_out, title=f"Confusion Matrix ({fname.replace('_resultsRaw.csv','')})")

    # Per-case
    per_case = pd.DataFrame({"Stage": STAGE, "Model": fname.split("_")[3], "CaseID": case_id, "Label": y_true, "Prediction": y_pred})
    per_case["Correct"] = (per_case["Label"] == per_case["Prediction"]).astype(int)
    per_case_out = os.path.join(OUT_DIR_CASE, fname.replace("_resultsRaw.csv", "_percase2025.csv"))
    per_case.to_csv(per_case_out, index=False)
    per_case_all.append(per_case)

if per_case_all:
    per_case_all_df = pd.concat(per_case_all, ignore_index=True)
    combined_out = os.path.join(OUT_DIR_CASE, f"ALLMODELS_{STAGE}_percase2025.csv")
    per_case_all_df.to_csv(combined_out, index=False)

    try:
        pivot_df = per_case_all_df.pivot_table(index="CaseID", columns="Model", values="Prediction", aggfunc="first")
        pivot_df.columns = [f"Pred_{c}" for c in pivot_df.columns]
        labels_df = per_case_all_df.drop_duplicates(subset=["CaseID"])[["CaseID", "Label"]].set_index("CaseID")
        pivot_df = labels_df.join(pivot_df)
        pivot_out = os.path.join(OUT_DIR_CASE, f"PIVOT_{STAGE}_percase2025.csv")
        pivot_df.to_csv(pivot_out)
    except Exception as e:
        print(f"Pivot not created: {e}")

print("Evaluation phase completed.")

# ========================================
# Statistical Tests
# ========================================

print("Starting statistical tests...")
STAGE = "Six"  # Change as needed
BASE_DIR = os.path.join(os.getcwd(), "Predicted results", f"PerCase_fromResultsRaw_{STAGE}")
pivot_name = f"PIVOT_{STAGE}_percase2025.csv"
PIVOT_CSV = os.path.join(BASE_DIR, pivot_name)

if not os.path.exists(PIVOT_CSV):
    print("Pivot file not found. Run evaluation first.")
else:
    df = pd.read_csv(PIVOT_CSV)
    MODEL_COLS = ["Pred_SVM", "Pred_RForest", "Pred_LogReg", "Pred_MLP", "Pred_GraBoost"]
    LABEL_COL = "Label"

    missing = [c for c in [LABEL_COL] + MODEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[[LABEL_COL] + MODEL_COLS].dropna()
    correct_mat = np.column_stack([(df[c].values == df[LABEL_COL].values).astype(int) for c in MODEL_COLS])
    n, k = correct_mat.shape
    print(f"Using {n} cases, {k} models")

    # Cochran's Q
    Q, df_q, p_q = cochrans_q_test(correct_mat)
    print(f"Cochran's Q: {Q:.4f}, df={df_q}, p={p_q:.6g}")

    # Post-hoc McNemar
    pairs = list(combinations(range(k), 2))
    raw_pvals = []
    pair_names = []

    for i, j in pairs:
        a = correct_mat[:, i]
        b = correct_mat[:, j]
        both_correct = ((a == 1) & (b == 1)).sum()
        a_correct_b_wrong = ((a == 1) & (b == 0)).sum()
        a_wrong_b_correct = ((a == 0) & (b == 1)).sum()
        both_wrong = ((a == 0) & (b == 0)).sum()
        table = [[both_correct, a_correct_b_wrong], [a_wrong_b_correct, both_wrong]]

        res = mcnemar(table, exact=False, correction=True)
        raw_pvals.append(res.pvalue)
        pair_names.append((MODEL_COLS[i], MODEL_COLS[j]))

    # Holm correction
    m = len(raw_pvals)
    order = np.argsort(raw_pvals)
    adj_p = np.empty(m, dtype=float)

    for rank, idx in enumerate(order, start=1):
        adj = (m - rank + 1) * raw_pvals[idx]
        if rank == 1:
            adj_p[idx] = adj9
        else:
            adj_p[idx] = max(adj_p[order[rank - 2]], adj)

    posthoc_df = pd.DataFrame({
        "Model_A": [pair_names[i][0] for i in range(m)],
        "Model_B": [pair_names[i][1] for i in range(m)],
        "p_raw": raw_pvals,
        "p_Holm": np.minimum(adj_p, 1.0)
    }).sort_values("p_Holm").reset_index(drop=True)

    print("\nPost-hoc McNemar with Holm correction:")
    print(posthoc_df.to_string(index=False))

print("All phases completed.")