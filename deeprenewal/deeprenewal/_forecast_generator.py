import logging
from typing import Any, Callable, Iterator, List, Optional

# Third-party imports
import mxnet as mx
import numpy as np

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.model.forecast import (
    Forecast,
    SampleForecast,
    QuantileForecast,
    DistributionForecast,
)
from gluonts.model.forecast_generator import ForecastGenerator

OutputTransform = Callable[[DataEntry, np.ndarray], np.ndarray]
BlockType = mx.gluon.Block

ALLOWABLE_FORECAST_TYPES = ["flat", "exact", "hybrid"]


class IntermittentSampleForecastGenerator(ForecastGenerator):
    @validated()
    def __init__(self, prediction_length: int, forecast_type: str):
        self.prediction_length = prediction_length
        self.forecast_type = forecast_type
        assert (
            forecast_type in ALLOWABLE_FORECAST_TYPES
        ), "Forecast Type needs to be one of " + "|".join(ALLOWABLE_FORECAST_TYPES)

    def __call__(
        self,
        inference_data_loader: InferenceDataLoader,
        prediction_net: BlockType,
        input_names: List[str],
        freq: str,
        output_transform: Optional[OutputTransform],
        num_samples: Optional[int],
        **kwargs,
    ) -> Iterator[Forecast]:
        for batch in inference_data_loader:
            inputs = [batch[k] for k in input_names]
            outputs = prediction_net(*inputs).asnumpy()
            if output_transform is not None:
                outputs = output_transform(batch, outputs)
            if num_samples:
                num_collected_samples = outputs[0].shape[0]
                collected_samples = [outputs]
                while num_collected_samples < num_samples:
                    outputs = prediction_net(*inputs).asnumpy()
                    if output_transform is not None:
                        outputs = output_transform(batch, outputs)
                    collected_samples.append(outputs)
                    num_collected_samples += outputs[0].shape[0]
                outputs = [
                    np.concatenate(s)[:num_samples] for s in zip(*collected_samples)
                ]
                assert len(outputs[0]) == num_samples
            i = -1

            for i, output in enumerate(outputs):
                # M/ Q
                if self.forecast_type == "flat":
                    output = output[:, :, 0] / output[:, :, 1]
                    mask_nan = np.isnan(output)
                    mask_inf = np.isposinf(output)
                    output[mask_nan] = 0
                    output[mask_inf] = 0
                    output = np.vstack([output[:, 0]] * self.prediction_length).T
                # exact: --> (Q-1) times 0, M, (Q-1) times 0, M, repeat
                # hybrid: --> Q times M/Q, Q times M/Q, repeat
                elif self.forecast_type in ["exact", "hybrid"]:
                    output_list = []
                    for o in output:
                        pred = []
                        for row in o:
                            m = row[0]
                            q = max(1, row[1])
                            if self.forecast_type == "hybrid":
                                f = m / q if m != 0 else 0
                            else:
                                f = 0
                            for j in range(int(q - 1)):
                                pred.append(f)
                            pred.append(
                                m
                            ) if self.forecast_type == "exact" else pred.append(f)
                        output_list.append(pred[: self.prediction_length])
                    output = np.array(output_list)
                else:
                    raise NotImplementedError(
                        f"{self.forecast_type} is not a value choice for forecast_type"
                    )
                yield SampleForecast(
                    output,
                    start_date=batch["forecast_start"][i],
                    freq=freq,
                    item_id=batch[FieldName.ITEM_ID][i]
                    if FieldName.ITEM_ID in batch
                    else None,
                    info=batch["info"][i] if "info" in batch else None,
                )
            assert i + 1 == len(batch["forecast_start"])
