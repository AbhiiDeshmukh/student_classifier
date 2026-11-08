# Student Outcome Classifier - ML Assignment 2

**Live Streamlit app:** <https://studentclassifier-ivfhgvccnlclkcw3mavfdj.streamlit.app>  
**GitHub repository:** <https://github.com/AbhiiDeshmukh/student_classifier>

---

## a. Problem statement

Higher-education institutions lose a large share of students before graduation. Identifying at-risk students early lets
the institution offer tutoring, financial or counselling support while it can still make a difference.

The goal of this project is to build and compare five classification models that predict, at the end of the
second semester, whether a student will **Drop out**, still be **Enrolled** at the end of the normal course duration, or
**Graduate**. This is a **3-class (multi-class) classification** problem. The trained models are served through an
interactive Streamlit web app that evaluates them on any uploaded test CSV.

## b. Dataset description

| Item | Details |
|---|---|
| Name | Predict Students' Dropout and Academic Success |
| Source | UCI Machine Learning Repository, dataset id 697 - <https://archive.ics.uci.edu/dataset/697> |
| Origin | Polytechnic Institute of Portalegre, Portugal (Realinho et al., 2022), licence CC BY 4.0 |
| Instances | 4,424 students (no duplicates, no missing values) |
| Features | 36 (all numeric / integer-coded) |
| Target | `Target` - `Graduate` (2,209, 49.9%), `Dropout` (1,421, 32.1%), `Enrolled` (794, 17.9%) |

**Feature groups**

- **Demographic:** marital status, nationality, gender, age at enrollment, displaced, international, educational special needs.
- **Socio-economic:** mother's/father's qualification and occupation, debtor, tuition fees up to date, scholarship holder.
- **Academic path at enrollment:** application mode, application order, course, daytime/evening attendance, previous qualification and its grade, admission grade.
- **Academic performance:** curricular units credited / enrolled / evaluated / approved / grade / without evaluation for the 1st and 2nd semesters (12 features).
- **Macro-economic context:** unemployment rate, inflation rate and GDP at the time of enrollment.

**Pre-processing** (done inside each saved scikit-learn `Pipeline`, so the app needs no separate artefacts):

- Stratified 80/20 train-test split (`random_state=5361`) giving 3,539 training rows and 885 test rows. The test split is saved as `test_data.csv`.
- 9 integer-coded **nominal** columns (marital status, application mode, course, previous qualification, nationality, parents' qualification and occupation) are **one-hot encoded**. Categories seen fewer than 15 times are grouped as "infrequent", and unseen categories are handled safely.
- The remaining 27 numeric columns are **standardised** (z-score), which matters for the distance-based (KNN) and margin-based (SVM, Logistic Regression) models.
- Hyper-parameters for every model were chosen by **5-fold stratified `GridSearchCV`** on the training set, optimising weighted F1.

## c. GitHub repository link

<https://github.com/AbhiiDeshmukh/student_classifier>

```
student_classifier/
|-- app.py                     # Streamlit app (loads saved models, no retraining)
|-- train_models.py            # trains + tunes + evaluates the 5 models, saves artefacts
|-- evaluation.py              # shared constants and metric helpers
|-- requirements.txt
|-- README.md
|-- test_data.csv              # held-out test split (885 rows, 36 features + Target)
|-- data/
|   `-- students_dropout_academic_success.csv   # original UCI file (semicolon separated)
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- svm.joblib
|   |-- test_metrics.csv       # comparison table below, as CSV
|   `-- metadata.json          # feature list, classes, best hyper-parameters, versions
`-- .streamlit/config.toml     # app theme
```

**Run locally**

```bash
pip install -r requirements.txt
python train_models.py        # optional: re-creates model/ and test_data.csv (~1-2 min)
streamlit run app.py
```

## d. Models used

All five models were trained on the same training split and evaluated on the same 885-row held-out test set.
Because the problem is multi-class, **Precision, Recall and F1 are support-weighted averages**, **AUC is the
weighted one-vs-rest ROC-AUC**, and **MCC is the multi-class Matthews correlation coefficient**.

| Model | Tuned hyper-parameters |
|---|---|
| Logistic Regression | C = 0.5, class_weight = None (multinomial, lbfgs) |
| Decision Tree | criterion = gini, max_depth = 6, min_samples_leaf = 1 |
| K-Nearest Neighbors | k = 17, Manhattan distance (p = 1), uniform weights |
| Naive Bayes (Gaussian) | var_smoothing = 0.316 |
| Support Vector Machine | RBF kernel, C = 5, gamma = 0.005, sigmoid (Platt) calibrated probabilities |

### Comparison table (held-out test set, n = 885)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7751 | **0.9123** | 0.7607 | 0.7751 | 0.7632 | 0.6270 |
| Decision Tree | 0.7322 | 0.8601 | 0.7203 | 0.7322 | 0.7215 | 0.5559 |
| KNN | 0.7153 | 0.8790 | 0.7053 | 0.7153 | 0.6851 | 0.5289 |
| Naive Bayes | 0.7141 | 0.8392 | 0.7185 | 0.7141 | 0.7105 | 0.5309 |
| Support Vector Machine | **0.7898** | 0.8955 | **0.7810** | **0.7898** | **0.7790** | **0.6527** |

Per-class recall on the test set, to show where the models differ:

| Model | Dropout (284) | Enrolled (159) | Graduate (442) |
|---|---|---|---|
| Logistic Regression | 0.768 | 0.377 | 0.923 |
| Decision Tree | 0.680 | 0.321 | 0.914 |
| KNN | 0.595 | 0.214 | **0.973** |
| Naive Bayes | 0.620 | **0.440** | 0.873 |
| Support Vector Machine | 0.746 | 0.421 | 0.950 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A strong second place (accuracy 0.775, MCC 0.627) and the **highest AUC (0.912)**, so its class probabilities rank students best. The strongest predictors (curricular units approved and grades in each semester, tuition fees up to date) relate to the outcome almost monotonically, so a linear decision boundary over the one-hot and scaled features captures most of the signal. It has the best Dropout recall (0.77) but recognises only 38% of Enrolled students. Tuning chose `class_weight=None`: balanced weights raised Enrolled recall but lowered weighted F1. |
| Decision Tree | Mid-table (accuracy 0.732, MCC 0.556). Cross-validation picked a shallow depth of 6; deeper trees overfit the 3.5k training rows. It is the most interpretable model. Its root split is on 2nd-semester approved units, which alone accounts for 67% of the tree's feature importance, followed by tuition-fee status. Its leaf-level probabilities are coarse, which gives the second-lowest AUC (0.860). It is weakest at separating Enrolled from Dropout. |
| KNN | Lowest F1 (0.685) and MCC (0.529). It has the highest Graduate recall (0.97) but finds only 21% of Enrolled students: with k = 17, neighbourhoods are dominated by the majority Graduate class, and distances lose meaning across 123 one-hot and scaled dimensions. Its AUC (0.879) is still reasonable, so the ranking is better than the hard predictions suggest. Prediction is also the slowest of the five, because every test row is compared with all 3,539 training rows. |
| Naive Bayes | Lowest accuracy (0.714) and lowest AUC (0.839). Its independence assumption is clearly violated here: 1st- and 2nd-semester features are strongly correlated, and one-hot columns are not Gaussian. Grid search needed very heavy variance smoothing (0.316) to compensate. Because it is less biased toward the majority class, it has the **best Enrolled recall (0.44)** and a better F1 than KNN. It is very fast and works well as a simple baseline. |
| Support Vector Machine | **Best on 5 of 6 metrics**: accuracy 0.790, precision 0.781, recall 0.790, F1 0.779 and MCC 0.653. The RBF kernel (C = 5, gamma = 0.005) models smooth non-linear interactions between academic-performance and socio-economic features that the linear model misses. This gives balanced recall across all three classes (0.75 / 0.42 / 0.95). Its AUC (0.896) is slightly below Logistic Regression's because probabilities come from post-hoc Platt calibration rather than being learned directly. |
| **Overall winner for this dataset** | **Support Vector Machine (RBF)**. It has the highest accuracy, F1 and MCC, and MCC is the most reliable single metric for this imbalanced 3-class problem. Logistic Regression is a close and more interpretable runner-up, and the better choice if well-ranked risk probabilities matter most. All five models find the **Enrolled** class hardest (best recall 0.44), because it is an in-between state whose academic profile overlaps with both Dropout and Graduate. |

## Streamlit app features

- **CSV upload** of test data, or one-click use of the bundled `test_data.csv`, plus a button to download the sample file.
- **Model selection dropdown** for the five saved models.
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1 and MCC shown as metric cards.
- **Confusion matrix** heatmap and a full **classification report**.
- **Compare all models** tab with a highlighted comparison table and a grouped bar chart.
- **Row-level predictions** with class probabilities, a misclassified-only filter and CSV download.
- Schema validation with clear error messages if columns or `Target` are missing. The original semicolon-separated UCI file is also accepted.

The app only **loads** the saved pipelines from `model/` and calls `predict` / `predict_proba`. Nothing is retrained on the uploaded data.

