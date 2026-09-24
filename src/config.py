"""Central configuration for the Diabetes Prediction System."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "diabetes.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_PATH = MODEL_DIR / "diabetes_model.pkl"
METRICS_PATH = REPORT_DIR / "model_comparison.csv"

TARGET = "Outcome"
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

# Columns where a value of 0 is physiologically impossible (i.e. a missing value)
ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
