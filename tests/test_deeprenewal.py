#!/usr/bin/env python

"""Tests for `deeprenewal` package."""


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
from typing import Tuple, List

# Third-party imports
import numpy as np
import pandas as pd
import mxnet as mx
import pytest
from random import randint

# First-party imports
import gluonts
from gluonts import time_feature, transform
from gluonts.core import fqname_for
from gluonts.core.serde import dump_code, dump_json, load_code, load_json
from gluonts.dataset.common import ProcessStartField, DataEntry, ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import Evaluator
# from gluonts.evaluation.backtest import backtest_metrics
from gluonts.evaluation.backtest import make_evaluation_predictions 
from gluonts.trainer import Trainer
from deeprenewal import DeepRenewalEstimator, IntermittentEvaluator
from deeprenewal.deeprenewal._transforms import (
    AddInterDemandPeriodFeature,
    DropNonZeroTarget,
)

FREQ = "1D"

INTER_DEMAND_TEST_VALUES = {
    "is_train": [True, False],
    "target": [
        (np.zeros(0), np.array([])),
        (np.zeros(13), np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])),
        (
            np.array([1, 1, 0, 0, 0, 4, 0, 0, 0, 8]),
            np.array([1, 1, 1, 2, 3, 4, 1, 2, 3, 4]),
        ),
    ],
    "start": [
        ProcessStartField.process("2012-01-02", freq="1D"),
    ],
}

ZERO_DEMAND_TEST_VALUES = {
    "is_train": [True, False],
    "target": [
        (np.zeros(0), np.array([])),
        (np.zeros(13), np.array([])),
        (
            np.array([1, 1, 0, 0, 0, 4, 0, 0, 0, 8]),
            np.array([1, 1, 4, 8]),
        ),
    ],
    "start": [
        ProcessStartField.process("2012-01-02", freq="1D"),
    ],
}
#type, hybridize, freq, num_feat_dynamic_real, cardinality
DEEP_RENEWAL_TEST_VALUES = {
    "type":['synthetic','constant'],
    "hybridize": [True, False],
    "freq" :["D","M"],
    "num_feat_dynamic_real": [0,2],
    "cardinality": [[],[2]]
}


@pytest.mark.parametrize("is_train", INTER_DEMAND_TEST_VALUES["is_train"])
@pytest.mark.parametrize("target", INTER_DEMAND_TEST_VALUES["target"])
@pytest.mark.parametrize("start", INTER_DEMAND_TEST_VALUES["start"])
def testAddInterDemandPeriodFeature(start, target, is_train):
    pred_length = 13
    t = AddInterDemandPeriodFeature(
        start_field=FieldName.START,
        target_field=FieldName.TARGET,
        output_field="myout",
        pred_length=pred_length
        # dtype=np.float64,
    )

    assert_serializable(t)

    data = {"start": start, "target": target[0]}
    res = t.map_transform(data, is_train=is_train)
    mat = res["myout"]
    assert np.array_equal(mat, target[1])


@pytest.mark.parametrize("is_train", ZERO_DEMAND_TEST_VALUES["is_train"])
@pytest.mark.parametrize("target", ZERO_DEMAND_TEST_VALUES["target"])
@pytest.mark.parametrize("start", ZERO_DEMAND_TEST_VALUES["start"])
def testDropNonZeroTarget(start, target, is_train):
    pred_length = 13
    input_fields = ["feature1", "feature2"]
    t = DropNonZeroTarget(
        input_fields=input_fields, target_field="target", pred_length=pred_length
    )

    assert_serializable(t)

    data = {
        "start": start,
        "target": target[0],
        "feature1": np.random.rand(*target[0].shape),
        "feature2": np.random.rand(*target[0].shape),
    }
    res = t.map_transform(data, is_train)

    assert np.array_equal(res["target"], target[1])
    assert len(res["feature1"]) == len(res["target"])
    assert len(res["feature2"]) == len(res["target"])
    # assert np.array_equal(res['target'],target[1])


@pytest.mark.parametrize("type", DEEP_RENEWAL_TEST_VALUES["type"])
@pytest.mark.parametrize("hybridize", DEEP_RENEWAL_TEST_VALUES["hybridize"])
@pytest.mark.parametrize("freq", DEEP_RENEWAL_TEST_VALUES["freq"])
@pytest.mark.parametrize("num_feat_dynamic_real", DEEP_RENEWAL_TEST_VALUES["num_feat_dynamic_real"])
@pytest.mark.parametrize("cardinality", DEEP_RENEWAL_TEST_VALUES["cardinality"])
def testDeepRenewal(type, hybridize, freq, num_feat_dynamic_real, cardinality):
    prediction_length = 3
    if type == "synthetic":
        train_ds, test_ds = make_dummy_datasets_with_features(
            prediction_length=prediction_length,
            freq=freq,
            num_feat_dynamic_real=num_feat_dynamic_real,
            cardinality=cardinality,
        )
    else:
        train_ds = make_constant_dataset(train_length=15, freq=freq)
        test_ds = train_ds
    trainer = Trainer(
        ctx="cpu", epochs=1, hybridize=hybridize
    )  # hybridize false for development
    estimator = DeepRenewalEstimator(
        prediction_length=prediction_length,
        freq=freq,
        trainer=trainer,
    )
    predictor = estimator.train(training_data=train_ds)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds, predictor=predictor, num_samples=100
    )
    evaluator = Evaluator(calculate_owa=False, num_workers=0)

    agg_metrics, item_metrics = evaluator(
        ts_it, forecast_it, num_series=len(test_ds)
    )
    if type == "synthetic":
        accuracy = 1.5
    else:
        accuracy = 1.3
    assert agg_metrics['ND']<= accuracy



def make_dummy_datasets_with_features(
    num_ts: int = 5,
    start: str = "2018-01-01",
    freq: str = "D",
    min_length: int = 5,
    max_length: int = 10,
    prediction_length: int = 3,
    cardinality: List[int] = [],
    num_feat_dynamic_real: int = 0,
) -> Tuple[ListDataset, ListDataset]:

    data_iter_train = []
    data_iter_test = []

    for k in range(num_ts):
        ts_length = randint(min_length, max_length)
        perc_zeros = 0.5
        mask = np.random.rand(ts_length) < perc_zeros
        target = np.array([1.0] * ts_length)
        target[mask] = 0
        data_entry_train = {
            FieldName.START: start,
            FieldName.TARGET: target,
        }
        if len(cardinality) > 0:
            data_entry_train[FieldName.FEAT_STATIC_CAT] = [
                randint(0, c) for c in cardinality
            ]
        data_entry_test = data_entry_train.copy()
        if num_feat_dynamic_real > 0:
            data_entry_train[FieldName.FEAT_DYNAMIC_REAL] = [
                [float(1 + k)] * ts_length for k in range(num_feat_dynamic_real)
            ]
            data_entry_test[FieldName.FEAT_DYNAMIC_REAL] = [
                [float(1 + k)] * (ts_length + prediction_length)
                for k in range(num_feat_dynamic_real)
            ]
        data_iter_train.append(data_entry_train)
        data_iter_test.append(data_entry_test)

    return (
        ListDataset(data_iter=data_iter_train, freq=freq),
        ListDataset(data_iter=data_iter_test, freq=freq),
    )


def make_constant_dataset(train_length,N=5, freq="D"):
    # generates 2 ** N - 1 timeseries with constant increasing values
    n = 2 ** N - 1
    perc_zeros = 0.5
    mask = np.random.rand(train_length) < perc_zeros
    targets = np.ones((n, train_length))
    for i in range(0, n):
        targets[i, :] = targets[i, :] * i
        targets[i, mask] = 0

    ds = gluonts.dataset.common.ListDataset(
        data_iter=[{"start": "2012-01-01", "target": targets[i, :]} for i in range(n)],
        freq=freq,
    )

    return ds


def assert_serializable(x: transform.Transformation):
    t = fqname_for(x.__class__)
    y = load_json(dump_json(x))
    z = load_code(dump_code(x))
    assert dump_json(x) == dump_json(
        y
    ), f"Code serialization for transformer {t} does not work"
    assert dump_code(x) == dump_code(
        z
    ), f"JSON serialization for transformer {t} does not work"


def assert_shape(array: np.array, reference_shape: Tuple[int, int]):
    assert (
        array.shape == reference_shape
    ), f"Shape should be {reference_shape} but found {array.shape}."


def assert_padded_array(
    sampled_array: np.array, reference_array: np.array, padding_array: np.array
):
    num_padded = int(np.sum(padding_array))
    sampled_no_padding = sampled_array[:, num_padded:]

    reference_array = np.roll(reference_array, num_padded, axis=1)
    reference_no_padding = reference_array[:, num_padded:]

    # Convert nans to dummy value for assertion because
    # np.nan == np.nan -> False.
    reference_no_padding[np.isnan(reference_no_padding)] = 9999.0
    sampled_no_padding[np.isnan(sampled_no_padding)] = 9999.0

    reference_no_padding = np.array(reference_no_padding, dtype=np.float32)

    assert (sampled_no_padding == reference_no_padding).all(), (
        f"Sampled and reference arrays do not match. '"
        f"Got {sampled_no_padding} but should be {reference_no_padding}."
    )


# testDropNonZeroTarget(
#     start=ProcessStartField.process("2012-01-02", freq="1D"),
#     target=(np.zeros(13), np.array([])),
#     is_train=False,
# )

# testDeepRenewal("constant", True)