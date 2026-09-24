"""Load the saved model and predict diabetes risk without retraining.

Usage:
    python -m src.predict          # prompts for the 8 values
    python -m src.predict --demo   # runs on a sample patient
"""
import argparse

import joblib
import pandas as pd

from src import config

PROMPTS = {
    "Pregnancies": "Number of pregnancies",
    "Glucose": "Plasma glucose (mg/dL)",
    "BloodPressure": "Diastolic blood pressure (mm Hg)",
    "SkinThickness": "Triceps skin fold thickness (mm)",
    "Insulin": "2-hour serum insulin (mu U/ml)",
    "BMI": "Body mass index",
    "DiabetesPedigreeFunction": "Diabetes pedigree function",
    "Age": "Age (years)",
}

DEMO_PATIENT = {"Pregnancies": 2, "Glucose": 148, "BloodPressure": 72,
                "SkinThickness": 35, "Insulin": 150, "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.63, "Age": 50}


def load_model(path=config.MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.train` first.")
    return joblib.load(path)


def predict(patient: dict, model=None):
    """Return (label, probability of diabetes) for one patient."""
    model = model or load_model()
    row = pd.DataFrame([patient])[config.FEATURES]
    proba = float(model.predict_proba(row)[0, 1])
    return ("Diabetic" if proba >= 0.5 else "Non-Diabetic"), proba


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--demo", action="store_true", help="Use a sample patient")
    args = parser.parse_args()

    patient = DEMO_PATIENT if args.demo else {
        k: float(input(f"{v}: ")) for k, v in PROMPTS.items()}
    label, proba = predict(patient)
    print(f"\nPrediction: {label}  (diabetes probability: {proba:.1%})")
    print("Note: educational project, not a medical diagnosis.")


if __name__ == "__main__":
    main()
