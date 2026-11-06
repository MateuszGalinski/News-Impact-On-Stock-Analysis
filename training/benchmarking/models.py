"""Model definitions, using the same settings as each track's notebook."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from .data import (
    NEWS_COLUMNS,
    NEWS_WITH_FEATURES,
    NO_NEWS_NO_FEATURES,
    NO_NEWS_WITH_FEATURES,
    Dataset,
)

XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": -1,
}
XGBOOST_V2_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 30,
}
RECURRENT_PARAMS = {"units": 32, "dense_units": 16, "epochs": 20, "batch_size": 32}

# One setup for every track, so the inputs are the only difference between them.
UNIFIED_RECURRENT_PARAMS = {**RECURRENT_PARAMS, "epochs": 200, "early_stopping_patience": 10}
UNIFIED_XGBOOST_PARAMS = XGBOOST_V2_PARAMS


@dataclass(frozen=True)
class ModelSpec:
    """A model to train.

    kind: "rnn", "lstm" or "xgboost".
    params: Keras settings as in RECURRENT_PARAMS, or XGBRegressor arguments.
        Adding early_stopping_patience to the Keras settings stops training
        after that many epochs without improvement and restores the best weights.
    news_last_only: for XGBoost, use the full window for price features but
        only the last day's value for news features.
    """

    name: str
    kind: str
    params: Mapping = field(default_factory=dict)
    news_last_only: bool = False


_RECURRENT_MODELS = (
    ModelSpec("RNN", "rnn", RECURRENT_PARAMS),
    ModelSpec("LSTM", "lstm", RECURRENT_PARAMS),
)
_XGBOOST = ModelSpec("XGBoost", "xgboost", XGBOOST_PARAMS)
_XGBOOST_V2 = ModelSpec("XGBoost (v2)", "xgboost", XGBOOST_V2_PARAMS, news_last_only=True)

UNIFIED_MODELS = (
    ModelSpec("RNN", "rnn", UNIFIED_RECURRENT_PARAMS),
    ModelSpec("LSTM", "lstm", UNIFIED_RECURRENT_PARAMS),
    ModelSpec("XGBoost", "xgboost", UNIFIED_XGBOOST_PARAMS),
)

NOTEBOOK_MODELS = {
    NO_NEWS_NO_FEATURES: (*_RECURRENT_MODELS, _XGBOOST),
    NO_NEWS_WITH_FEATURES: (*_RECURRENT_MODELS, _XGBOOST),
    NEWS_WITH_FEATURES: (*_RECURRENT_MODELS, _XGBOOST_V2),
}


def _tabular_inputs(spec: ModelSpec, X: np.ndarray, feature_columns: tuple[str, ...]) -> np.ndarray:
    if not spec.news_last_only:
        return X.reshape(len(X), -1)
    news = [i for i, column in enumerate(feature_columns) if column in NEWS_COLUMNS]
    price = [i for i in range(len(feature_columns)) if i not in news]
    return np.concatenate([X[:, :, price].reshape(len(X), -1), X[:, -1, news]], axis=1)


class FittedModel:
    """A trained model that predicts one column per horizon.

    best_steps: the epochs or trees whose weights are used, per horizon.
    steps_run: the epochs or trees actually trained, per horizon.
    """

    def __init__(self, spec: ModelSpec, estimators: list, feature_columns: tuple[str, ...]):
        self.spec = spec
        self.estimators = estimators
        self.feature_columns = feature_columns
        self.target_mean = 0.0
        self.target_std = 1.0
        self.best_steps: list[int] = []
        self.steps_run: list[int] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict log returns, with one column per horizon."""
        if self.spec.kind == "xgboost":
            inputs = _tabular_inputs(self.spec, X, self.feature_columns)
            predictions = np.column_stack([estimator.predict(inputs) for estimator in self.estimators])
        else:
            predictions = np.asarray(self.estimators[0].predict(X, verbose=0)).reshape(len(X), -1)
        return predictions * self.target_std + self.target_mean


def _fit_recurrent(spec: ModelSpec, dataset: Dataset, seed: int) -> FittedModel:
    import keras
    from keras import layers

    params = {**RECURRENT_PARAMS, **spec.params}
    recurrent_layers = {"rnn": layers.SimpleRNN, "lstm": layers.LSTM}

    keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            layers.Input(shape=dataset.train.X.shape[1:]),
            recurrent_layers[spec.kind](params["units"]),
            layers.Dense(params["dense_units"], activation="relu"),
            layers.Dense(len(dataset.horizons)),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    patience = params.get("early_stopping_patience")
    callbacks = []
    if patience is not None:
        callbacks.append(
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        )
    stopping = dataset.stopping_split
    history = model.fit(
        dataset.train.X,
        dataset.train.y,
        validation_data=(stopping.X, stopping.y),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=callbacks,
        verbose=0,
    )
    fitted = FittedModel(spec, [model], dataset.feature_columns)
    steps_run = len(history.history["val_loss"])
    best_step = int(np.argmin(history.history["val_loss"])) + 1 if patience is not None else steps_run
    fitted.best_steps = [best_step] * len(dataset.horizons)
    fitted.steps_run = [steps_run] * len(dataset.horizons)
    return fitted


def _fit_xgboost(spec: ModelSpec, dataset: Dataset, seed: int) -> FittedModel:
    from xgboost import XGBRegressor

    X_train = _tabular_inputs(spec, dataset.train.X, dataset.feature_columns)
    stopping = dataset.stopping_split
    X_stopping = _tabular_inputs(spec, stopping.X, dataset.feature_columns)

    estimators = []
    for index in range(len(dataset.horizons)):
        estimator = XGBRegressor(**spec.params, random_state=seed)
        estimator.fit(
            X_train,
            dataset.train.y[:, index],
            eval_set=[(X_stopping, stopping.y[:, index])],
            verbose=False,
        )
        estimators.append(estimator)
    fitted = FittedModel(spec, estimators, dataset.feature_columns)
    fitted.steps_run = [estimator.get_booster().num_boosted_rounds() for estimator in estimators]
    if "early_stopping_rounds" in spec.params:
        fitted.best_steps = [estimator.best_iteration + 1 for estimator in estimators]
    else:
        fitted.best_steps = list(fitted.steps_run)
    return fitted


def fit_model(spec: ModelSpec, dataset: Dataset, seed: int) -> FittedModel:
    """Train one model. A standardized dataset also gets a standardized target."""
    target_mean, target_std = 0.0, 1.0
    if dataset.standardized:
        target_mean = dataset.train.y.mean(axis=0)
        target_std = dataset.train.y.std(axis=0)
        dataset = replace(
            dataset,
            train=replace(dataset.train, y=(dataset.train.y - target_mean) / target_std),
            validation=replace(dataset.validation, y=(dataset.validation.y - target_mean) / target_std),
        )
        if dataset.early_stopping is not None:
            dataset = replace(
                dataset,
                early_stopping=replace(
                    dataset.early_stopping, y=(dataset.early_stopping.y - target_mean) / target_std
                ),
            )

    if spec.kind in ("rnn", "lstm"):
        fitted = _fit_recurrent(spec, dataset, seed)
    elif spec.kind == "xgboost":
        fitted = _fit_xgboost(spec, dataset, seed)
    else:
        raise ValueError(f"Unknown model kind {spec.kind!r}")

    fitted.target_mean = target_mean
    fitted.target_std = target_std
    return fitted
