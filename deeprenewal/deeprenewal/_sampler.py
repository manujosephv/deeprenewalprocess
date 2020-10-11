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

from functools import lru_cache
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

from gluonts.core.component import validated
from gluonts.core.exception import GluonTSDateBoundsError
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName

from gluonts.transform._base import FlatMapTransformation
from gluonts.transform.sampler import InstanceSampler, ContinuousTimePointSampler


def shift_timestamp(ts: pd.Timestamp, offset: int) -> pd.Timestamp:
    """
    Computes a shifted timestamp.

    Basic wrapping around pandas ``ts + offset`` with caching and exception
    handling.
    """
    return _shift_timestamp_helper(ts, ts.freq, offset)


@lru_cache(maxsize=10000)
def _shift_timestamp_helper(ts: pd.Timestamp, freq: str, offset: int) -> pd.Timestamp:
    """
    We are using this helper function which explicitly uses the frequency as a
    parameter, because the frequency is not included in the hash of a time
    stamp.

    I.e.
      pd.Timestamp(x, freq='1D')  and pd.Timestamp(x, freq='1min')

    hash to the same value.
    """
    try:
        # this line looks innocent, but can create a date which is out of
        # bounds values over year 9999 raise a ValueError
        # values over 2262-04-11 raise a pandas OutOfBoundsDatetime
        return ts + offset * ts.freq
    except (ValueError, pd._libs.OutOfBoundsDatetime) as ex:
        raise GluonTSDateBoundsError(ex)


class RenewalInstanceSplitter(FlatMapTransformation):
    """
    Selects training instances, by slicing the target and other time series
    like arrays at random points in training mode or at the last time point in
    prediction mode. Assumption is that all time like arrays start at the same
    time point.

    The target and each time_series_field is removed and instead two
    corresponding fields with prefix `past_` and `future_` are included. E.g.

    If the target array is one-dimensional, the resulting instance has shape
    (len_target). In the multi-dimensional case, the instance has shape (dim,
    len_target).

    target -> past_target and future_target

    The transformation also adds a field 'past_is_pad' that indicates whether
    values where padded or not.

    Convention: time axis is always the last axis.

    Parameters
    ----------

    target_field
        field containing the target
    is_pad_field
        output field indicating whether padding happened
    start_field
        field containing the start date of the time series
    forecast_start_field
        output field that will contain the time point where the forecast starts
    train_sampler
        instance sampler that provides sampling indices given a time-series
    past_length
        length of the target seen before making prediction
    future_length
        length of the target that must be predicted
    lead_time
        gap between the past and future windows (default: 0)
    output_NTC
        whether to have time series output in (time, dimension) or in
        (dimension, time) layout (default: True)
    time_series_fields
        fields that contains time-series, they are split in the same interval
        as the target (default: None)
    pick_incomplete
        whether training examples can be sampled with only a part of
        past_length time-units
        present for the time series. This is useful to train models for
        cold-start. In such case, is_pad_out contains an indicator whether
        data is padded or not. (default: True)
    dummy_value
        Value to use for padding. (default: 0.0)
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        is_pad_field: str,
        start_field: str,
        forecast_start_field: str,
        train_sampler: InstanceSampler,
        past_length: int,
        future_length: int,
        lead_time: int = 0,
        output_NTC: bool = True,
        time_series_fields: Optional[List[str]] = None,
        pick_incomplete: bool = True,
        dummy_value: float = 0.0,
    ) -> None:

        assert future_length > 0

        self.train_sampler = train_sampler
        self.past_length = past_length
        self.future_length = future_length
        self.lead_time = lead_time
        self.output_NTC = output_NTC
        self.ts_fields = time_series_fields if time_series_fields is not None else []
        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field
        self.pick_incomplete = pick_incomplete
        self.dummy_value = dummy_value

    def _past(self, col_name):
        return f"past_{col_name}"

    def _future(self, col_name):
        return f"future_{col_name}"

    @staticmethod
    def remove_zero_demand(
        data: DataEntry, target_field, input_fields, pred_length
    ) -> DataEntry:
        target = (
            data[target_field].reshape(1, -1)
            if data[target_field].ndim == 1
            else data[target_field]
        )
        mask = target[0, :] > 0
        data[target_field] = (
            data[target_field][mask]
            if data[target_field].ndim == 1
            else data[target_field][:, mask]
        )
        # Adding Trues for the prediction length. Useful while prediction
        mask = np.append(mask, [True] * pred_length)
        for field in input_fields:
            _mask = mask[: data[field].shape[-1]]
            if data[field].ndim == 1:
                data[field] = data[field][_mask]
            elif data[field].ndim == 2:
                data[field] = data[field][:, _mask]
            else:
                raise NotImplementedError("ndim for {} should be atmost 2")

        return data

    def flatmap_transform(self, data: DataEntry, is_train: bool) -> Iterator[DataEntry]:
        pl = self.future_length
        lt = self.lead_time
        slice_cols = self.ts_fields + [self.target_field]
        true_target = data[self.target_field]
        true_len_target = true_target.shape[-1]
        data = self.remove_zero_demand(
            data,
            target_field=self.target_field,
            input_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES],
            pred_length=self.future_length,
        )

        target = data[self.target_field]

        len_target = target.shape[-1]

        minimum_length = (
            self.future_length
            if self.pick_incomplete
            else self.past_length + self.future_length
        ) + self.lead_time

        if is_train:
            sampling_bounds = (
                (
                    0,
                    len_target - self.future_length - self.lead_time,
                )  # TODO: create parameter lower sampling bound for NBEATS
                if self.pick_incomplete
                else (
                    self.past_length,
                    len_target - self.future_length - self.lead_time,
                )
            )

            # We currently cannot handle time series that are
            # too short during training, so we just skip these.
            # If we want to include them we would need to pad and to
            # mask the loss.
            sampled_indices = (
                np.array([], dtype=int)
                if len_target < minimum_length
                else self.train_sampler(target, *sampling_bounds)
            )
        else:
            assert self.pick_incomplete or len_target >= self.past_length
            sampled_indices = np.array([len_target], dtype=int)
        for i in sampled_indices:
            pad_length = max(self.past_length - i, 0)
            if not self.pick_incomplete:
                assert pad_length == 0, f"pad_length should be zero, got {pad_length}"
            d = data.copy()
            for ts_field in slice_cols:
                if i > self.past_length:
                    # truncate to past_length
                    past_piece = d[ts_field][..., i - self.past_length : i]
                elif i < self.past_length:
                    pad_block = (
                        np.ones(
                            d[ts_field].shape[:-1] + (pad_length,),
                            dtype=d[ts_field].dtype,
                        )
                        * self.dummy_value
                    )
                    past_piece = np.concatenate(
                        [pad_block, d[ts_field][..., :i]], axis=-1
                    )
                else:
                    past_piece = d[ts_field][..., :i]
                d[self._past(ts_field)] = past_piece
                d[self._future(ts_field)] = d[ts_field][..., i + lt : i + lt + pl]
                del d[ts_field]
            pad_indicator = np.zeros(self.past_length)
            if pad_length > 0:
                pad_indicator[:pad_length] = 1

            if self.output_NTC:
                for ts_field in slice_cols:
                    d[self._past(ts_field)] = d[self._past(ts_field)].transpose()
                    d[self._future(ts_field)] = d[self._future(ts_field)].transpose()

            d[self._past(self.is_pad_field)] = pad_indicator
            if is_train:
                pass
                # d[self.forecast_start_field] = shift_timestamp(d[self.start_field], i + lt)
            else:
                d[self.forecast_start_field] = shift_timestamp(d[self.start_field], true_len_target + lt)
            yield d
            