"""Feature relevance analysis: correlation, ANOVA F-score and Random Forest importance."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer

from src import config
from src.data_utils import load_data, get_train_test


def analyse_features() -> pd.DataFrame:
    """Rank features using three independent criteria (training data only)."""
    X_train, _, y_train, _ = get_train_test(load_data())
    X_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X_train),
        columns=X_train.columns, index=X_train.index,
    )

    f_scores, _ = f_classif(X_imp, y_train)
    rf = RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_STATE)
    rf.fit(X_imp, y_train)

    ranking = pd.DataFrame({
        "abs_correlation_with_target": X_imp.corrwith(y_train).abs(),
        "anova_f_score": f_scores,
        "rf_importance": rf.feature_importances_,
    }, index=X_imp.columns)
    return ranking.sort_values("rf_importance", ascending=False)


def main():
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ranking = analyse_features()
    print(ranking.round(3))
    ranking.round(4).to_csv(config.REPORT_DIR / "feature_ranking.csv")

    ax = ranking["rf_importance"].sort_values().plot(kind="barh", figsize=(7, 4))
    ax.set_title("Random Forest feature importance")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "feature_importance.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
