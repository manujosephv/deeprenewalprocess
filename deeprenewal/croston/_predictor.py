# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
import os
from typing import Dict, Iterator, Optional
import logging

# Third-party imports
import numpy as np
import pandas as pd

# First-party imports
from gluonts.core.component import validated
from gluonts.support.pandas import forecast_start
from gluonts.dataset.common import Dataset
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor

from .croston import fit_croston

logger = logging.getLogger("gluonts").getChild("croston")


class CrostonForecastPredictor(RepresentablePredictor):
    """
    Wrapper for calling the `croston forecast

    Parameters
    ----------
    freq
        The granularity of the time series (e.g. '1H')
    prediction_length
        Number of time points to be predicted.
    variant
        The method from rforecast to be used one of
        "original", "sba", "sbj"
    trunc_length
        Maximum history length to feed to the model (some models become slow
        with very long series).
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        variant: str = "original",
        no_of_params: int = 2,
        trunc_length: Optional[int] = None,
    ) -> None:
        super().__init__(freq=freq, prediction_length=prediction_length)

        supported_methods = ["original", "sba", "sbj"]
        assert (
            variant in supported_methods
        ), f"method {variant} is not supported please use one of {supported_methods}"
        self.variant = variant
        assert (
            no_of_params <= 2,
            "Number of parameters should be less than or equal 2",
        )
        self.no_of_params = no_of_params

        self.prediction_length = prediction_length
        self.freq = freq
        self.trunc_length = trunc_length
        self.params = {
            "prediction_length": self.prediction_length,
            "output_types": ["samples"],
        }

    def _unlist(self, l):
        if type(l).__name__.endswith("Vector"):
            return [self._unlist(x) for x in l]
        else:
            return l

    def _run_croston_forecast(self, d, params):
        fit_pred = fit_croston(
            d["target"],
            forecast_length=params["prediction_length"],
            croston_variant=self.variant,
            number_parameters=self.no_of_params,
        )

        return {"samples": fit_pred["croston_forecast"].reshape(1, -1)}

    def predict(
        self,
        dataset: Dataset,
        num_samples: int = 1,
        save_info: bool = False,
        **kwargs,
    ) -> Iterator[SampleForecast]:
        if num_samples != 1:
            num_samples = 1
            logger.warning(
                "num_samples changed to 1 because Croston is non-probabilistic"
            )

        assert num_samples == 1, "Non Probabilistic Method only supports num_samples=1"

        for entry in dataset:
            if isinstance(entry, dict):
                data = entry
            else:
                data = entry.data
                if self.trunc_length:
                    data = data[-self.trunc_length :]

            params = self.params.copy()
            params["num_samples"] = num_samples
            forecast_dict = self._run_croston_forecast(data, params)
            samples = np.array(forecast_dict["samples"])
            expected_shape = (params["num_samples"], self.prediction_length)
            assert (
                samples.shape == expected_shape
            ), f"Expected shape {expected_shape} but found {samples.shape}"

            yield SampleForecast(
                samples, forecast_start(data), self.freq, item_id=data["item_id"]
            )
