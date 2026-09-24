# Diabetes Prediction System

A machine learning system that predicts whether a person is likely to have diabetes from eight diagnostic measurements. The project covers data cleaning, feature analysis, comparison of six classifiers with hyper-parameter tuning, model evaluation, and a saved model that can be reused for prediction without retraining.

> Educational project. Not a medical diagnosis tool.

## Dataset

**Pima Indians Diabetes Database** (NIDDK, available on Kaggle and the UCI ML Repository).

| Property | Details |
|---|---|
| Samples | 768 patients (women, 21+ years) |
| Features | 8 numeric diagnostic measurements |
| Target | `Outcome`: 1 = diabetic, 0 = non-diabetic |
| Class balance | about 65% non-diabetic, 35% diabetic |

Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.

## Approach

1. **Data cleaning.** In `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI`, a value of 0 is physiologically impossible and really means "not recorded" (5 / 35 / 227 / 374 / 11 rows respectively). These zeros are converted to `NaN` and imputed with the **median inside the model pipeline**, so medians are learned from training data only (no data leakage).
2. **Feature analysis.** Features are ranked by correlation with the target, ANOVA F-score and Random Forest importance (`src/feature_selection.py`). Glucose is the strongest predictor, followed by BMI, Age and DiabetesPedigreeFunction. `train.py --top-k K` can train on only the K best features (`SelectKBest`).
3. **Model comparison.** Six classifiers are tuned with `GridSearchCV` (5-fold stratified CV, F1 scoring) on an 80% training split: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting.
4. **Evaluation.** Accuracy, precision, recall, F1 and ROC-AUC on a held-out 20% stratified test set, plus a confusion matrix and ROC curve.
5. **Model selection.** The best model is chosen by cross-validated F1 on the **training** data, so the test set remains an unbiased final check.
6. **Serialization.** The complete pipeline (imputer, scaler, classifier) is saved with `joblib` to `models/diabetes_model.pkl` and reloaded by `src/predict.py`.

## Results

Held-out test set (154 patients), random_state = 42:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 |
|---|---|---|---|---|---|---|
| **Random Forest** (selected) | 0.740 | 0.603 | 0.759 | 0.672 | 0.814 | 0.707 |
| Support Vector Machine | 0.727 | 0.591 | 0.722 | 0.650 | 0.814 | 0.690 |
| Logistic Regression | 0.708 | 0.571 | 0.667 | 0.615 | 0.810 | 0.682 |
| Decision Tree | 0.747 | 0.627 | 0.685 | 0.655 | 0.804 | 0.669 |
| Gradient Boosting | 0.721 | 0.622 | 0.519 | 0.566 | 0.815 | 0.644 |
| K-Nearest Neighbors | 0.740 | 0.652 | 0.556 | 0.600 | 0.807 | 0.632 |

Random Forest was selected for the highest cross-validated F1 and a recall of about 76% on the diabetic class, which matters in screening because missed cases are costly. With only 154 test samples, differences of a few points between models are within noise, and results can shift slightly across scikit-learn versions.

Figures (confusion matrix, ROC curve, correlation heatmap, feature importance) are written to `reports/figures/`.

## Project structure

```
Diabetes-Prediction-System/
├── data/diabetes.csv            # Pima Indians Diabetes dataset
├── models/diabetes_model.pkl    # saved best pipeline
├── reports/                     # metrics CSVs and figures
├── src/
│   ├── config.py                # paths, constants
│   ├── data_utils.py            # loading, cleaning, splitting
│   ├── eda.py                   # exploratory analysis and plots
│   ├── feature_selection.py     # feature ranking
│   ├── train.py                 # tuning, comparison, selection, saving
│   └── predict.py               # reuse the saved model
├── tests/test_pipeline.py
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/irfank11/diabetes-prediction-system.git
cd diabetes-prediction-system
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

python -m src.eda                 # data quality report + plots
python -m src.feature_selection   # feature ranking
python -m src.train               # train, compare, save best model
python -m src.predict --demo      # predict with the saved model
python -m src.predict             # enter your own values
pytest                            # run tests
```

## Tech stack

Python, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Joblib.

## Credits

Dataset: Pima Indians Diabetes Database, National Institute of Diabetes and Digestive and Kidney Diseases (via Kaggle / UCI).
# Diabetes-Prediction-System
# Diabetes-Prediction-System
