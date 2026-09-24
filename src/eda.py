"""Exploratory data analysis: prints a data-quality summary and saves plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.data_utils import load_data, data_quality_report


def main():
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    print(f"Shape: {df.shape}")
    print("\nClass distribution:")
    print(df[config.TARGET].value_counts(normalize=True).round(3))
    print("\nData quality report:")
    print(data_quality_report(df))
    print("\nSummary statistics:")
    print(df.describe().T.round(2))

    plt.figure(figsize=(4, 4))
    sns.countplot(x=config.TARGET, data=df)
    plt.title("Class distribution (0 = Non-diabetic, 1 = Diabetic)")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "class_distribution.png", dpi=150)
    plt.close()

    df[config.FEATURES + [config.TARGET]].hist(figsize=(12, 9), bins=20)
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "feature_histograms.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"\nFigures saved to {config.FIGURE_DIR}")


if __name__ == "__main__":
    main()
