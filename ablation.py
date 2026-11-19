# ========================================
# CVMS feature-ablation analysis
# ========================================
# Arms (candidate pool offered to the Random-Forest selector):
#   all    age + sex + morphology
#   noage  sex + morphology
#   nosex  age + morphology
#   morph  morphology only
#   age    age only (no selection step)
#
# Pipeline details:
#   * Every stochastic component is seeded (SVC, RF, MLP, BayesSearchCV).
#   * StandardScaler is fitted on the TRAINING split only (no test-set leakage).
#   * Accuracy is computed against the consensus label (y_test), never
#     against another model's prediction.
#   * Full-51 tuned Random Forest baseline is also saved (rf_all51_<format>.csv).
#   * If two arms end up with the same selected features for the same
#     label format, the fitted models/predictions are reused, so any
#     difference between them can only come from the features.
#
# Usage:  python ablation.py [data.csv] [outdir] [n_jobs]
#   Optional env var: CVMS_LABELS=Three_Class (to run a single format).
#   data.csv defaults to project2_CVMS.csv and must contain 51 feature columns
#   starting at 'Gender' and the label columns Six_Class, Five_Class, Three_Class.
#   Rows whose 'Level of dentists agreement' is 'Disagreement' are dropped if present.

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

warnings.filterwarnings("ignore")

SPLIT_SEED = 7
SEL_THRESHOLD = 0.015
TEST_SIZE = 0.3
NBOOT = 2000
BOOT_SEED = 2026
LABELS = (os.environ.get("CVMS_LABELS") or "Three_Class,Five_Class,Six_Class").split(",")
ARMS = ["all", "noage", "nosex", "morph", "age"]
MODELS = ["SVM", "RForest", "LogReg", "MLP"]
AGE, SEX = "Age month", "Gender"
CONTRASTS = [("nosex", "all"), ("morph", "noage"), ("nosex", "morph"),
             ("noage", "morph"), ("all", "age"), ("morph", "age")]


def algorithms():
    return {
        "SVM": (SVC(probability=True, random_state=0),
                {"C": [0.1, 1, 10, 100, 200], "kernel": ["linear", "rbf", "poly"],
                 "gamma": ["scale", "auto", 0.01, 0.001], "degree": [2, 3, 4], "coef0": [0, 0.1, 0.5]}),
        "RForest": (RandomForestClassifier(random_state=0),
                    {"max_depth": Integer(5, 30), "min_samples_split": Integer(2, 10),
                     "n_estimators": Integer(1, 100), "max_features": Categorical(["sqrt", "log2"]),
                     "min_samples_leaf": Integer(1, 5), "bootstrap": Categorical([True, False])}),
        "LogReg": (LogisticRegression(max_iter=1000),
                   {"penalty": Categorical(["l2"]), "C": Real(1e-4, 1e4, prior="log-uniform"),
                    "solver": Categorical(["lbfgs", "newton-cg", "liblinear", "sag", "saga"])}),
        "MLP": (MLPClassifier(max_iter=1000, random_state=0),
                {"hidden_layer_sizes": [(31, 16), (31, 16, 8), (22, 11), (22, 11, 5), (20, 10),
                                        (20, 10, 5), (31,), (22,), (20,), (16, 8), (10, 5)],
                 "activation": ["tanh", "relu"], "solver": ["sgd", "adam"],
                 "alpha": [0.0001, 0.05], "learning_rate": ["constant", "adaptive"]}),
    }


def load(path):
    df = pd.read_csv(path)
    if "Level of dentists agreement" in df.columns:
        df = df[df["Level of dentists agreement"] != "Disagreement"]
    df = df.reset_index(drop=True)
    start = list(df.columns).index(SEX)
    feats = list(df.columns[start:start + 51])
    assert len(feats) == 51 and AGE in feats, "expected 51 feature columns from 'Gender'"
    X = df[feats].replace([np.inf, -np.inf], 0)
    return df, X, feats


def candidates(arm, feats):
    return {"all": feats,
            "noage": [c for c in feats if c != AGE],
            "nosex": [c for c in feats if c != SEX],
            "morph": [c for c in feats if c not in (AGE, SEX)],
            "age": [AGE]}[arm]


def select(Xtr, ytr, cols):
    if len(cols) == 1:
        return list(cols)
    rf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1).fit(Xtr[cols], ytr)
    sfm = SelectFromModel(rf, threshold=SEL_THRESHOLD, prefit=True)
    return [c for c, keep in zip(cols, sfm.get_support()) if keep]


def fit_predict(A, ytr, B, n_jobs):
    preds, info = {}, {}
    for name, (est, space) in algorithms().items():
        if name == "MLP":
            s = GridSearchCV(est, space, n_jobs=n_jobs, cv=10)
        else:
            s = BayesSearchCV(est, space, n_jobs=n_jobs, scoring="accuracy", random_state=0,
                              cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=1))
        s.fit(A, ytr)
        preds[name] = s.predict(B)
        info[name] = {"cv": float(s.best_score_),
                      "params": {k: str(v) for k, v in s.best_params_.items()}}
    return preds, info


def fit_rf_only(A, ytr, B, n_jobs):
    est, space = algorithms()["RForest"]
    s = BayesSearchCV(est, space, n_jobs=n_jobs, scoring="accuracy", random_state=0,
                      cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=1))
    s.fit(A, ytr)
    return s, s.predict(B)


def mcnemar_exact(a, b):
    a_only, b_only = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)
    return a_only, b_only, p


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, "project2_CVMS.csv")
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "project2_CVMS.csv" if os.path.exists("project2_CVMS.csv") else default_csv
    )
    out = sys.argv[2] if len(sys.argv) > 2 else "ablation_out"
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    os.makedirs(out, exist_ok=True)
    df, X, feats = load(path)
    print(f"{len(df)} cases, {len(feats)} features")

    all_pred, summary, cache = {}, [], {}
    for lab in LABELS:
        y = df[lab].reset_index(drop=True)
        # Same split for every arm and format (unstratified, as in project2_CVMS.py).
        tr, te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE, random_state=SPLIT_SEED)
        scaler = StandardScaler().fit(X.iloc[tr])           # train-only scaling
        Xs = pd.DataFrame(scaler.transform(X), columns=feats).fillna(0)
        Xtr, Xte, ytr, yte = Xs.iloc[tr], Xs.iloc[te], y.iloc[tr], y.iloc[te]
        _, _rf51 = fit_rf_only(Xtr.values, ytr, Xte.values, n_jobs)
        pd.DataFrame([dict(label=lab, rf_all51_test=float(np.mean(_rf51 == yte.values)))]).to_csv(
            os.path.join(out, f"rf_all51_{lab}.csv"), index=False)
        for arm in ARMS:
            sel = select(Xtr, ytr, candidates(arm, feats))
            key = (lab, tuple(sorted(sel)))
            reused = key in cache
            if not reused:
                cache[key] = fit_predict(Xtr[sel].values, ytr, Xte[sel].values, n_jobs)
            preds, info = cache[key]
            print(f"{lab:12} {arm:6} k={len(sel):2} {'(reused)' if reused else ''} "
                  + " ".join(f"{m}={np.mean(preds[m] == yte.values) * 100:.1f}" for m in MODELS), flush=True)
            frame = pd.DataFrame({"idx": te, "y": yte.values, **{m: preds[m] for m in MODELS}})
            frame.to_csv(os.path.join(out, f"pred_{arm}_{lab}.csv"), index=False)
            with open(os.path.join(out, f"sel_{arm}_{lab}.json"), "w", encoding="utf-8") as f:
                json.dump(sel, f)
            all_pred[(arm, lab)] = frame
            for m in MODELS:
                summary.append(dict(label=lab, arm=arm, model=m, k=len(sel), reused=reused,
                                    test_acc=float(np.mean(preds[m] == yte.values)),
                                    cv_acc=info[m]["cv"], params=json.dumps(info[m]["params"])))
    pd.DataFrame(summary).to_csv(os.path.join(out, "ablation_per_model_" + "_".join(LABELS) + ".csv"), index=False)
    if len(LABELS) < 3:
        return

    # Mean accuracy + paired contrasts with a shared patient-resample bootstrap.
    rows, paired = [], []
    for lab in LABELS:
        n = len(all_pred[("all", lab)])
        idx = np.random.default_rng(BOOT_SEED).integers(0, n, size=(NBOOT, n))

        def correct(arm):
            f = all_pred[(arm, lab)]
            return np.column_stack([(f[m].to_numpy() == f["y"].to_numpy()) for m in MODELS])

        for arm in ARMS:
            c = correct(arm).mean(axis=1)
            d = np.array([c[b].mean() for b in idx])
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append(dict(label=lab, arm=arm, mean_acc=float(c.mean()), lo=float(lo), hi=float(hi)))
        for left, right in CONTRASTS:
            ca, cb = correct(left), correct(right)
            ma, mb = ca.mean(axis=1), cb.mean(axis=1)
            d = np.array([ma[b].mean() - mb[b].mean() for b in idx])
            lo, hi = np.percentile(d, [2.5, 97.5])
            for j, m in enumerate(MODELS):
                a_only, b_only, p = mcnemar_exact(ca[:, j], cb[:, j])
                paired.append(dict(label=lab, contrast=f"{left} - {right}", model=m,
                                   mean_diff=float(ma.mean() - mb.mean()), lo=float(lo), hi=float(hi),
                                   left_only=a_only, right_only=b_only, p=p))
    pd.DataFrame(rows).to_csv(os.path.join(out, "ablation_mean_ci.csv"), index=False)
    pd.DataFrame(paired).to_csv(os.path.join(out, "ablation_paired.csv"), index=False)
    print(pd.DataFrame(rows).assign(mean_acc=lambda d: (d.mean_acc * 100).round(1),
                                    lo=lambda d: (d.lo * 100).round(1),
                                    hi=lambda d: (d.hi * 100).round(1)).to_string(index=False))


if __name__ == "__main__":
    main()
