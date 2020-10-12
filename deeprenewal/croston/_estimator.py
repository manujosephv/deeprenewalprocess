# First-party imports
from gluonts.core.component import validated
from gluonts.model.estimator import DummyEstimator

# Relative imports
from ._predictor import CrostonForecastPredictor


class CrostonForecastEstimator(DummyEstimator):
    @validated(
        getattr(CrostonForecastPredictor.__init__, "Model")
    )  # Reuse the model Predictor model
    def __init__(self, **kwargs) -> None:
        super().__init__(predictor_cls=CrostonForecastPredictor, **kwargs)
