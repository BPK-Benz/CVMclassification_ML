# CVMS Classification Project

This repository contains the de-identified consensus dataset and reproducible machine learning pipeline for classifying Cervical Vertebral Maturation Stages (CVMS) across six-, five-, and three-stage formats.

## Dataset

`project2_CVMS.csv` contains **1,268 de-identified consensus lateral cephalograms** (retained from 1,380 collected radiographs after excluding 112 cases of complete disagreement among the three calibrated raters):

- **`No.`**: Sequential de-identified case index
- **51 candidate features** (`Gender`, `Age month`, and 49 landmark-derived C2–C4 morphological measurements matching Supplementary Table S1, including corrected C3/C4 height-to-width ratios)
- **`Rater_1`, `Rater_2`, `Rater_3`**: Anonymized individual 6-stage ratings (for leave-one-rater-out human benchmarking and inter-rater agreement)
- **Target labels**:
  - `Six_Class`: `Class_1`–`Class_6` (CS1–CS6)
  - `Five_Class`: `Class_1&2` (CS1+CS2 merged), `Class_3`–`Class_6`
  - `Three_Class`: `Pre-pubertal` (CS1–CS2), `Pubertal` (CS3–CS4), `Post-pubertal` (CS5–CS6)

## Pipeline Overview

1. **Data Splitting & Scaling (No Test-Set Leakage)**
   - Load `project2_CVMS.csv` (filters out `Level of dentists agreement == 'Disagreement'` automatically if an unfiltered CSV is supplied).
   - Partition once using an unstratified 70/30 train/test split (`random_state=7`; training `n = 887`, held-out test `n = 381`), reused identically across all three classification formats.
   - Replace `+/-inf` with `0`, fit `StandardScaler` on the **training split only**, apply the fitted scaler unchanged to the test split, and fill `NaN` with `0`.
2. **Feature Selection (Training Split Only)**
   - Fit a 10,000-tree `RandomForestClassifier(random_state=0)` on the training split only and retain features exceeding impurity importance `0.015` via `SelectFromModel(..., prefit=True)`.
   - Yields nested feature subsets of **18** (3-stage), **21** (5-stage), and **26** (6-stage) features.
   - Full-51 vs. selected-feature comparison tunes Random Forest separately on each feature set via `BayesSearchCV`.
3. **Model Training & Hyperparameter Tuning**
   - Four classifiers trained independently per format with fixed seeds:
     - Support Vector Machine (`SVM`) — `BayesSearchCV` (50 iterations, 10-fold `RepeatedKFold`, `random_state=0`)
     - Random Forest (`RForest`) — `BayesSearchCV` (50 iterations, 10-fold `RepeatedKFold`, `random_state=0`)
     - Logistic Regression (`LogReg`) — `BayesSearchCV` (50 iterations, 10-fold `RepeatedKFold`, `random_state=0`)
     - Multi-Layer Perceptron (`MLP`) — `GridSearchCV` (10-fold CV, `random_state=0`)
4. **Evaluation & Summary Results**
   - All metrics, confusion matrices, classification reports, per-case outputs, and statistical tests are computed strictly against ground-truth consensus labels (`y_test`).
   - Summary outputs (mean 10-fold CV: **85.94% / 74.53% / 67.09%**; mean held-out test: **87.1% / 75.4% / 68.0%**; CV-to-test gap: **0.1–2.7 pp**) are stored in `ModelAccuracy/` and `BestParameters/`.
5. **Feature Ablation & Supplementary Analyses**
   - `ablation.py`: Single reproducible pipeline evaluating all feature configurations (`all`, `noage`, `nosex`, `morph`, `age`) and the tuned 51-feature RF baseline across all three formats.
   - `supplementary_analyses.py`: Reproduces Table III (canonical ablation with 2,000-bootstrap 95% CIs and paired contrasts), Table IV (typical vs. atypical age subgroup analysis with 95% CIs), collapsed prediction format comparisons (`6→3`, `5→3` exact McNemar), within-/cross-phase error decomposition, leave-one-rater-out human benchmarks, and 100-bootstrap feature-selection stability (`--run-stability`).

## Annotation Workflow

Landmark annotation and base feature extraction build on [CVMS_Classification](https://github.com/BPK-Benz/CVMS_Classification):

1. Annotate 19 anatomical landmarks on C2–C4 using VGG Image Annotator (VIA): https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html
2. Verify landmark placement with `view.py`.
3. Extract morphometric features from landmark coordinates with `potintTofeature.py`.

## Requirements

Reference environment:
- Python `3.11.16`
- `numpy == 1.26.4`
- `pandas == 2.1.4`
- `scikit-learn == 1.3.2`
- `scikit-optimize == 0.10.2`
- `scipy`, `statsmodels`, `joblib`, `matplotlib`, `seaborn`, `tqdm`

## Usage

```bash
# 1. Primary 3-, 5-, and 6-stage pipeline
python project2_CVMS.py project2_CVMS.csv

# 2. Full reproducible ablation + primary predictions + full-51 RF baseline
python ablation.py project2_CVMS.csv ablation_out

# 3. Supplementary analyses (Tables III–IV, collapsed McNemar, error breakdown, stability)
python supplementary_analyses.py project2_CVMS.csv ablation_out supplementary_out --run-stability
```
