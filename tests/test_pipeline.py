import numpy as np

from src import config
from src.data_utils import get_train_test, load_data, mark_invalid_as_nan
from src.train import build_pipeline, get_models


def test_dataset_schema():
    df = load_data()
    assert df.shape[1] == 9
    assert set(df[config.TARGET].unique()) == {0, 1}


def test_invalid_zeros_become_nan():
    df = load_data()
    df.loc[0, "Glucose"] = 0
    assert np.isnan(mark_invalid_as_nan(df).loc[0, "Glucose"])


def test_split_is_stratified():
    X_train, X_test, y_train, y_test = get_train_test(load_data())
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_all_six_models_fit_and_predict():
    X_train, X_test, y_train, _ = get_train_test(load_data())
    models = get_models()
    assert len(models) == 6
    for model in models.values():
        pipe = build_pipeline(model).fit(X_train, y_train)
        assert pipe.predict(X_test).shape[0] == len(X_test)
