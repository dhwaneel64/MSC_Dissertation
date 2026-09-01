import numpy as np
import pandas as pd

from src.models.baseline import ConstantMeanModel


def test_fit_predict_returns_expected_mean():
    model = ConstantMeanModel()
    y_train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    X_train = pd.DataFrame({"f": [0.0] * 5})
    X_test = pd.DataFrame({"f": [0.0, 0.0]})

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert preds.shape == (2,)
    np.testing.assert_allclose(preds, 3.0)
