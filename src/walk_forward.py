from typing import Protocol, Callable
import numpy as np
import pandas as pd


class Model(Protocol):
    """A model usable by the walk-forward engine.

    fit(X_train, y_train): fits on training features and target.
    predict(X_test): returns a 1-D array of predictions, one per row.

    The engine guarantees X_test is built from data with index <= t.
    The engine never passes t+1 data to predict.
    """
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None: ...
    def predict(self, X_test: pd.DataFrame) -> np.ndarray: ...


def walk_forward(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    model_factory: Callable[[], Model],
    initial_train_end: pd.Timestamp,
    target_col: str = "y",
) -> pd.DataFrame:
    """Expanding-window walk-forward validation.

    At each step:
      1. Train the model on all rows with index <= train_end.
      2. Predict the single row immediately after train_end.
      3. Record the prediction and the realised target.
      4. Advance train_end to the next row. Repeat.

    Args:
        dataset: model-ready DataFrame with feature_cols and target_col.
        feature_cols: list of columns to use as features.
        model_factory: zero-arg callable returning a fresh Model instance per step.
        initial_train_end: timestamp of the last training row in the first iteration.
            The first OOS prediction is for the row immediately after this date.
        target_col: name of the target column. Defaults to "y".

    Returns:
        DataFrame indexed by the OOS dates, with columns:
          - "y_true": realised target value
          - "y_pred": model prediction

    Raises:
        ValueError if initial_train_end is not in dataset.index.
        ValueError if there are no rows after initial_train_end.
        ValueError if dataset.index is not monotonic increasing.
    """
    if not dataset.index.is_monotonic_increasing:
        raise ValueError("dataset.index must be monotonic increasing")

    if initial_train_end not in dataset.index:
        raise ValueError(f"initial_train_end {initial_train_end} is not in dataset.index")

    pos = dataset.index.get_loc(initial_train_end)

    if pos >= len(dataset) - 1:
        raise ValueError(
            f"initial_train_end {initial_train_end} is the last row; no OOS rows available"
        )

    records = []
    for i in range(pos, len(dataset) - 1):
        train = dataset.iloc[: i + 1]
        test = dataset.iloc[i + 1 : i + 2]

        X_train = train[feature_cols]
        y_train = train[target_col]
        X_test = test[feature_cols]

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        records.append(
            {
                "date": test.index[0],
                "y_true": float(test[target_col].iloc[0]),
                "y_pred": float(y_pred[0]),
            }
        )

    result = pd.DataFrame(records).set_index("date")
    result.index.name = dataset.index.name
    return result


def make_model_factory_from_class(model_cls, **kwargs) -> Callable[[], Model]:
    """Helper: returns a zero-arg factory that constructs model_cls(**kwargs) per call.

    Each walk-forward step gets a fresh model instance. No state leaks across steps.
    """
    def factory() -> Model:
        return model_cls(**kwargs)
    return factory
