"""Top-level package for DeepRenewal."""

__author__ = """Manu Joseph"""
__email__ = 'manujosephv@gmail.com'
__version__ = '0.1.2'

# Relative imports
from .deeprenewal._estimator import DeepRenewalEstimator
from .croston._estimator import CrostonForecastEstimator
from .croston._predictor import CrostonForecastPredictor
from ._datasets import get_dataset
from ._evaluator import IntermittentEvaluator

__all__ = ["get_dataset","DeepRenewalEstimator","CrostonForecastEstimator","CrostonForecastPredictor", "IntermittentEvaluator"]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)