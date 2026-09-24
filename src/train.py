"""Tune and compare 6 classifiers, select the best one and save it.

Usage:
    python -m src.train                # use all 8 features
    python -m src.train --top-k 6      # keep the 6 best features (ANOVA F-test)
"""
import argparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data_utils import get_train_test, load_data

rs = config.RANDOM_STATE


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=rs),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Support Vector Machine": SVC(probability=True, random_state=rs),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=rs),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=rs),
        "Gradient Boosting": GradientBoostingClassifier(random_state=rs),
    }


# Small hyper-parameter grids (searched with 5-fold CV on the training set, scoring F1).
# class_weight="balanced" helps the minority (diabetic) class, which is only ~35% of data.
PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10],
                            "clf__class_weight": [None, "balanced"]},
    "K-Nearest Neighbors": {"clf__n_neighbors": [5, 7, 11, 15, 21],
                            "clf__weights": ["uniform", "distance"]},
    "Support Vector Machine": {"clf__C": [0.1, 1, 10],
                               "clf__gamma": ["scale", 0.01, 0.1],
                               "clf__class_weight": [None, "balanced"]},
    "Decision Tree": {"clf__max_depth": [3, 4, 5, 7],
                      "clf__min_samples_leaf": [5, 10, 20],
                      "clf__class_weight": [None, "balanced"]},
    "Random Forest": {"clf__n_estimators": [200, 400],
                      "clf__max_depth": [4, 6, None],
                      "clf__min_samples_leaf": [2, 5, 10],
                      "clf__class_weight": [None, "balanced"]},
    "Gradient Boosting": {"clf__n_estimators": [100, 200],
                          "clf__learning_rate": [0.03, 0.1],
                          "clf__max_depth": [2, 3]},
}


def build_pipeline(model, top_k=None) -> Pipeline:
    """Impute -> scale -> (optional feature selection) -> classifier.

    Everything is fitted inside the Pipeline, so cross-validation and the test
    set never influence preprocessing (no data leakage).
    """
    steps = [("imputer", SimpleImputer(strategy="median")),
             ("scaler", StandardScaler())]
    if top_k:
        steps.append(("selector", SelectKBest(f_classif, k=top_k)))
    steps.append(("clf", model))
    return Pipeline(steps)


def evaluate(pipe, X_test, y_test) -> dict:
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }


def main(top_k=None):
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = get_train_test(load_data())
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=rs)

    rows, fitted = [], {}
    for name, model in get_models().items():
        pipe = build_pipeline(model, top_k)
        search = GridSearchCV(pipe, PARAM_GRIDS[name], cv=cv, scoring="f1", n_jobs=-1)
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        rows.append({"Model": name, **evaluate(best_pipe, X_test, y_test),
                     "CV F1": search.best_score_})
        fitted[name] = best_pipe
        print(f"  tuned {name}: {search.best_params_}")

    results = pd.DataFrame(rows).set_index("Model").round(3)
    results = results.sort_values("CV F1", ascending=False)
    print("\nModel comparison (test-set metrics + tuned 5-fold CV F1 on train data):")
    print(results.to_string())
    results.to_csv(config.METRICS_PATH)

    # Model selection uses cross-validation on the TRAINING data only,
    # so the test set stays an unbiased final check.
    best_name = results.index[0]
    best = fitted[best_name]
    print(f"\nSelected model: {best_name}")
    print(classification_report(y_test, best.predict(X_test),
                                target_names=["Non-Diabetic", "Diabetic"]))

    ConfusionMatrixDisplay.from_estimator(
        best, X_test, y_test, display_labels=["Non-Diabetic", "Diabetic"], cmap="Blues")
    plt.title(f"Confusion matrix - {best_name}")
    plt.savefig(config.FIGURE_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    RocCurveDisplay.from_estimator(best, X_test, y_test)
    plt.title(f"ROC curve - {best_name}")
    plt.savefig(config.FIGURE_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    joblib.dump(best, config.MODEL_PATH)
    print(f"\nModel saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--top-k", type=int, default=None,
                        help="Keep only the K best features (default: all 8)")
    main(parser.parse_args().top_k)
