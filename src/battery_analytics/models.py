"""Fixed model specifications and training-only whole-cell model selection."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ['mean_baseline', 'linear_ridge', 'gradient_boosting', 'random_forest']


def estimator(name, config):
    seed = config['seed']
    if name == 'mean_baseline':
        model = DummyRegressor(strategy='mean')
    elif name == 'linear_ridge':
        model = Ridge(alpha=1.0)
    elif name == 'gradient_boosting':
        model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, learning_rate=.06,
                                              min_samples_leaf=15, l2_regularization=1., early_stopping=False,
                                              random_state=seed)
    elif name == 'random_forest':
        model = RandomForestRegressor(n_estimators=config['validation']['random_forest_trees'], max_depth=8,
                                      min_samples_leaf=6, max_features=1., random_state=seed, n_jobs=2)
    else:
        raise ValueError(name)
    # Fit scaler inside every training split, never on validation or test rows.
    return Pipeline([('scale', StandardScaler()), ('model', model)])


def cell_balanced_weights(frame):
    counts = frame.groupby('group_id').size()
    weights = frame.group_id.map(1 / counts).to_numpy()
    return weights * len(weights) / weights.sum()


def fit_estimator(name, frame, columns, config):
    assert np.isfinite(frame[columns].to_numpy()).all()
    assert np.isfinite(frame.soh).all()
    result = estimator(name, config)
    result.fit(frame[columns], frame.soh, model__sample_weight=cell_balanced_weights(frame))
    return result


def inner_selection(train, columns, config):
    """One separate validation split of whole cells, with predeclared hyperparameters."""
    groups = train.group_id.unique()
    if len(groups) < 3:
        raise ValueError('At least three training cells required for separate validation')
    splitter = GroupShuffleSplit(n_splits=1, test_size=config['validation']['validation_cell_fraction'], random_state=config['seed'])
    ti, vi = next(splitter.split(train, groups=train.group_id))
    inner_train, validation = train.iloc[ti], train.iloc[vi]
    assert set(inner_train.group_id).isdisjoint(validation.group_id)
    scores = {}
    for name in MODEL_NAMES:
        model = fit_estimator(name, inner_train, columns, config)
        p = model.predict(validation[columns])
        absolute = pd.DataFrame({'cell': validation.group_id.to_numpy(), 'error': np.abs(p - validation.soh.to_numpy())})
        scores[name] = float(absolute.groupby('cell').error.mean().mean() * 100)
    # Selecting a family is part of the algorithm and only sees inner validation cells.
    winner = min(scores, key=scores.get)
    return winner, dict(inner_train_cells=sorted(inner_train.group_id.unique()),
                        inner_validation_cells=sorted(validation.group_id.unique()), validation_mae_pp=scores)
