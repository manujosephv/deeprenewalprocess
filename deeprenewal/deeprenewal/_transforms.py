import numpy as np
import pandas as pd
from typing import List
from gluonts.dataset.common import DataEntry
from gluonts.core.component import validated, DType
from gluonts.transform import (
    AddTimeFeatures,
    target_transformation_length,
    SimpleTransformation,
    MapTransformation,
    shift_timestamp,
)

# TODO See if inheritance is possible from add time features
class AddInterDemandPeriodFeature(MapTransformation):
    """
    Calculates the Inter Demand Period feature

    Parameters
    ----------
    start_field
        Field with the start time stamp of the time series
    target_field
        Field with the array containing the time series values
    output_field
        Field name for result.
    pred_length
        Prediction length
    """

    @validated()
    def __init__(
        self,
        start_field: str,
        target_field: str,
        output_field: str,
        pred_length: int,
    ) -> None:
        self.pred_length = pred_length
        self.start_field = start_field
        self.target_field = target_field
        self.output_field = output_field
        self._min_time_point: pd.Timestamp = None
        self._max_time_point: pd.Timestamp = None
        self._full_range_date_features: np.ndarray = None
        self._date_index: pd.DatetimeIndex = None

    def _update_cache(self, start: pd.Timestamp, length: int) -> None:
        end = shift_timestamp(start, length)
        if self._min_time_point is not None:
            if self._min_time_point <= start and end <= self._max_time_point:
                return
        if self._min_time_point is None:
            self._min_time_point = start
            self._max_time_point = end
        self._min_time_point = min(shift_timestamp(start, -50), self._min_time_point)
        self._max_time_point = max(shift_timestamp(end, 50), self._max_time_point)
        self.full_date_range = pd.date_range(
            self._min_time_point, self._max_time_point, freq=start.freq
        )
        # self._full_range_date_features = (
        #     np.vstack(
        #         [feat(self.full_date_range) for feat in self.date_features]
        #     )
        #     if self.date_features
        #     else None
        # )
        self._date_index = pd.Series(
            index=self.full_date_range,
            data=np.arange(len(self.full_date_range)),
        )

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        start = data[self.start_field]
        length = target_transformation_length(
            data[self.target_field], self.pred_length, is_train=is_train
        )
        self._update_cache(start, length)
        i0 = self._date_index[start]
        date_idx = self._date_index.iloc[i0 : i0 + length].index
        # When is_train is false, date_idx has len of target_len + prediction_len
        # which is useful in time feature generation, but we only need target length
        date_idx = date_idx[: len(data[self.target_field])]
        feature = pd.Series(np.ones(len(date_idx)) * np.nan, index=date_idx)
        mask = data[self.target_field] > 0
        feature.loc[mask] = feature.loc[mask].index
        # filling in nan in first row with the corresponding date
        # Assumption: If the frame starts with a zero demand, earliest date in frame is taken as a start
        if len(feature) > 0:
            if pd.isnull(feature[0]):
                feature[0] = feature.index[0]
        feature = feature.ffill().to_frame()
        # feature["diff"] = (feature.index - feature.iloc[:, 0]) / pd.Timedelta(
        #     1, feature.index.freqstr
        # )
        feature["diff"] = feature.index.to_period(feature.index.freqstr).astype(
            int
        ) - pd.DatetimeIndex(feature.iloc[:, 0]).to_period(
            feature.index.freqstr
        ).astype(
            int
        )
        feature["diff"] = feature["diff"].shift(1).round() + 1
        feature["diff"] = feature["diff"].fillna(method="bfill")
        feature = feature["diff"].values
        if self.output_field in data.keys():
            data[self.output_field] = np.vstack([data[self.output_field], feature])
        else:
            data[self.output_field] = feature
        return data


class DropNonZeroTarget(SimpleTransformation):
    """
    Drops NonZero Target Instances for DeepRenewal Training

    Parameters
    ----------
    input_fields
        Fields to drop the instances from
    target_field
        The field which we need to cehck for zeros
    pred_length
        The length of the prediction
    """

    @validated()
    def __init__(
        self, input_fields: List[str], target_field: str, pred_length: int
    ) -> None:
        self.input_fields = input_fields
        self.target_field = target_field
        self.pred_length = pred_length

    def transform(self, data: DataEntry) -> DataEntry:
        target = (
            data[self.target_field].reshape(1, -1)
            if data[self.target_field].ndim == 1
            else data[self.target_field]
        )
        mask = target[0, :] > 0
        data[self.target_field] = (
            data[self.target_field][mask]
            if data[self.target_field].ndim == 1
            else data[self.target_field][:, mask]
        )
        # Adding Trues for the prediction length. Useful while prediction
        mask = np.append(mask, [True] * self.pred_length)
        for field in self.input_fields:
            _mask = mask[: data[field].shape[-1]]
            if data[field].ndim == 1:
                data[field] = data[field][_mask]
            elif data[field].ndim == 2:
                data[field] = data[field][:, _mask]
            else:
                raise NotImplementedError("ndim for {} should be atmost 2")

        return data
