"""Regression test: model optimizers and their param_funcs must be picklable.

The training pipeline ships the predictor (which holds the optimizers and
their YAML-registry param functions) to ProcessPoolExecutor model workers.
Closures cannot be pickled, so the inline search-space DSL must compile to a
module-level callable (``InlineParamFunc``).
"""

import pickle

import optuna

from gpse.models.regression_model_optimizer import RegressionModelOptimizer
from gpse.models.classification_model_optimizer import ClassificationModelOptimizer


def test_regression_optimizer_and_param_funcs_are_picklable():
    opt = RegressionModelOptimizer(random_seed=42, n_threads=2)
    restored = pickle.loads(pickle.dumps(opt))
    assert set(restored.model_configs) == set(opt.model_configs)
    for name in restored.model_configs:
        pickle.dumps(restored.get_param_func(name))


def test_classification_optimizer_and_param_funcs_are_picklable():
    opt = ClassificationModelOptimizer(random_seed=42, n_threads=2, n_classes=3)
    restored = pickle.loads(pickle.dumps(opt))
    assert set(restored.model_configs) == set(opt.model_configs)
    for name in restored.model_configs:
        pickle.dumps(restored.get_param_func(name))


def test_inline_param_func_survives_pickle_roundtrip():
    opt = RegressionModelOptimizer(random_seed=42, n_threads=1)
    param_func = pickle.loads(pickle.dumps(opt.get_param_func("elasticnet_reg")))

    study = optuna.create_study()
    params = param_func(study.ask())

    assert params["random_state"] == 42
    assert 1.0e-8 <= params["alpha"] <= 100.0
    assert 1.0e-5 <= params["l1_ratio"] <= 1.0
