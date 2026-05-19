
# CVMS Classification Project

This project trains and evaluates machine learning classifiers for CVMS stage prediction using a structured dataset.

## Overview

The main script project2_CVMS.py loads data from project2_CVMS.csv and runs a full ML pipeline across three classification formats (3-class, 5-class, and 6-class), 
covering preprocessing, model training, evaluation, and statistical comparison.

Pipeline Steps
1. Data Preparation
- Load structured feature data from project2_CVMS.csv
- Apply feature scaling and preprocessing
- Target labels are taken from the last three columns (one per class format)
2. Feature Selection
- Uses SelectFromModel to identify the most informative features for each classification target
3. Model Training
- Each class format is trained and tuned independently across five models:
  - Support Vector Machine (SVM)
  - Random Forest
  - Logistic Regression
  - Multi-Layer Perceptron (MLP)
4. Output & Evaluation
- Best hyperparameters and trained models saved as .joblib files
- Confusion matrices and classification reports exported per model
- Per-case prediction results exported for review
- Pivot tables generated for cross-model comparison
5. Statistical Testing
- Cochran's Q Test — checks whether multiple models perform significantly differently overall
- McNemar's Test — performs pairwise comparison between individual models

## Annotation Workflow

This project is based on annotation data from [CVMS_Classification](https://github.com/BPK-Benz/CVMS_Classification).

1. Annotator uses VGG software to label 19 points around the cervical vertebrae (C2, C3, and C4).
   - Software: https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html
2. Confirm the position of these 19 points on each image using `view.py`.
3. Convert the labeled points into relevant features such as size, shape, and concavity using `pointTofeature.py`.
4. The original 6-class labels (CS1–CS6) were reclassified into two alternative formats using a majority voting system:
 - 5-class format: CS1 and CS2 were merged into CVMS I, with CS3–CS6 mapped sequentially to CVMS II–V.
- 3-class format: Stages were grouped by growth phase — CS1–CS2 as pre-pubertal, CS3–CS4 as pubertal, and CS5–CS6 as post-pubertal.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- scikit-optimize (`skopt`)
- matplotlib
- seaborn
- tqdm
- scipy
- statsmodels
- joblib

## Usage

Run the main script from the `Project2` directory:

```powershell
python project2_CVMS.py
```

