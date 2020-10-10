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

"""
Here we reuse the datasets used by LSTNet as the processed url of the datasets
are available on GitHub.
"""
import json
import logging
import os
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from gluonts.dataset.common import TrainDatasets, load_datasets
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository._util import metadata, save_to_file, to_dict
from gluonts.support.util import get_download_path
from gluonts.gluonts_tqdm import tqdm


def _preprocess_retail_data(df, combination):
    df['Country'] = df['Country'].astype("category").cat.codes
    df['StockCode'] = df['StockCode'].astype("category").cat.codes
    df = df.groupby(["StockCode", "Country", "InvoiceDate"]).agg(
        {"Quantity": "sum", "UnitPrice": "mean"}
    )
    counts = df.reset_index().groupby(combination)["StockCode"].count()
    combinations_selected = counts[counts > 10].index
    df = df.reset_index().set_index(combination)
    df = df[df.index.isin(combinations_selected)]
    max_date = df.InvoiceDate.max().replace(hour=0, minute=0, second=0)
    def resample_ds(df):
        df.InvoiceDate = pd.to_datetime(df.InvoiceDate, yearfirst=True)
        df.rename(columns={"InvoiceDate":"date"},inplace=True)
        new_idx = pd.date_range(
            df.date.min().replace(hour=0, minute=0, second=0),
            max_date,
            freq="1D",
            name = 'InvoiceDate'
        )
        df.set_index("date", inplace=True)
        df = (
            df.resample("1D")
            .agg({"Quantity": "sum", "UnitPrice": "mean"})
            .reindex(new_idx)
        )
        df["Quantity"] = df["Quantity"].fillna(0)
        df["UnitPrice"] = df["UnitPrice"].ffill().bfill()
        return df

    df = df.reset_index().groupby(combination).apply(resample_ds)
    df.Quantity = df.Quantity.clip(lower=0)
    return df




def generate_retail_dataset(dataset_path: Path, split: str = "2011-11-24"):
    retail_dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    df = pd.read_excel(retail_dataset_url)
    combination = ["StockCode", "Country"]
    df = _preprocess_retail_data(df, combination)
    df.to_pickle("temp.pkl")
    # df = pd.read_pickle("temp.pkl")
    idx = pd.IndexSlice[:, :, :split]
    train_df = df.loc[idx, :].reset_index()
    idx = pd.IndexSlice[:, :, split:]
    test_df = df.loc[idx, :].reset_index()
    full_df = df.reset_index()
    single_prediction_length = len(test_df["InvoiceDate"].unique())
    feat_static_cat = combination
    feat_dynamic_real = []
    target = 'Quantity'
    date_col = 'InvoiceDate'

    os.makedirs(dataset_path, exist_ok=True)

    uniq_combs = train_df[combination].drop_duplicates().apply(tuple, axis=1)
    dynamic_real_train_l = []
    dynamic_real_test_l = []
    stat_cat_l = []
    start_l = []
    train_target_l = []
    test_target_l = []
    for stock_code, country in tqdm(uniq_combs):
        df = train_df[
            (train_df.StockCode == stock_code) & (train_df.Country == country)
        ]
        _df = full_df[(full_df.StockCode == stock_code) & (full_df.Country == country)]
        train_ts = _df[target].values.ravel()
        if (train_ts>0).sum() > (single_prediction_length+13):
            test_feat_dyn_array = _df.loc[:, feat_dynamic_real].values.T
            train_feat_dyn_array = test_feat_dyn_array[:, :-single_prediction_length]

            test_ts = train_ts.copy()
            train_ts = train_ts[:-single_prediction_length]

            dynamic_real_train_l.append(train_feat_dyn_array)
            dynamic_real_test_l.append(test_feat_dyn_array)
            start_l.append(df[date_col].min())
            train_target_l.append(train_ts)
            test_target_l.append(test_ts)
            stat_cat_l.append(
                np.squeeze(df.loc[:, feat_static_cat].drop_duplicates().values)
            )
    stat_cat_cardinalities = [
            len(full_df[col].unique()) for col in feat_static_cat
        ]

    with open(dataset_path / "metadata.json", "w") as f:
        f.write(
            json.dumps(
                metadata(
                    cardinality=stat_cat_cardinalities,
                    freq="1D",
                    prediction_length=single_prediction_length,
                )
            )
        )

    train_file = dataset_path / "train" / "data.json"
    test_file = dataset_path / "test" / "data.json"
    train_ds = [
        {
            FieldName.ITEM_ID: "|".join(map(str,uniq_comb)),
            FieldName.TARGET: target.tolist(),
            FieldName.START: str(start),
            FieldName.FEAT_STATIC_CAT: fsc.tolist(),
            FieldName.FEAT_DYNAMIC_REAL: fdr.tolist(),
        }
        for uniq_comb, target, start, fdr, fsc in zip(
            uniq_combs, train_target_l, start_l, dynamic_real_train_l, stat_cat_l,
        )
    ]
    save_to_file(train_file, train_ds)
    test_ds = [
        {
            FieldName.ITEM_ID: "|".join(map(str,uniq_comb)),
            FieldName.TARGET: target.tolist(),
            FieldName.START: str(start),
            FieldName.FEAT_STATIC_CAT: fsc.tolist(),
            FieldName.FEAT_DYNAMIC_REAL: fdr.tolist(),
        }
        for uniq_comb, target, start, fdr, fsc in zip(
            uniq_combs, test_target_l, start_l, dynamic_real_test_l, stat_cat_l,
        )
    ]
    save_to_file(test_file, test_ds)


default_dataset_path = get_download_path() / "datasets"

dataset_recipes = OrderedDict(
    {
        # each recipe generates a dataset given a path
        "retail_dataset": partial(generate_retail_dataset, split="2011-11-01")
    }
)


def materialize_dataset(
    dataset_name: str, path: Path = default_dataset_path, regenerate: bool = False,
) -> Path:
    """
    Ensures that the dataset is materialized under the `path / dataset_name`
    path.
    Parameters
    ----------
    dataset_name
        name of the dataset, for instance "m4_hourly"
    regenerate
        whether to regenerate the dataset even if a local file is present.
        If this flag is False and the file is present, the dataset will not
        be downloaded again.
    path
        where the dataset should be saved
    Returns
    -------
        the path where the dataset is materialized
    """

    path.mkdir(parents=True, exist_ok=True)
    dataset_path = path / dataset_name

    dataset_recipe = dataset_recipes[dataset_name]

    if not dataset_path.exists() or regenerate:
        logging.info(f"downloading and processing {dataset_name}")
        dataset_recipe(dataset_path=dataset_path)
    else:
        logging.info(f"using dataset already processed in path {dataset_path}.")

    return dataset_path


def get_dataset(
    dataset_name: str, path: Path = default_dataset_path, regenerate: bool = False,
) -> TrainDatasets:
    """
    Get a repository dataset.
    The datasets that can be obtained through this function have been used
    with different processing over time by several papers (e.g., [SFG17]_,
    [LCY+18]_, and [YRD15]_).
    Parameters
    ----------
    dataset_name
        name of the dataset, for instance "m4_hourly"
    regenerate
        whether to regenerate the dataset even if a local file is present.
        If this flag is False and the file is present, the dataset will not
        be downloaded again.
    path
        where the dataset should be saved
    Returns
    -------
        dataset obtained by either downloading or reloading from local file.
    """
    dataset_path = materialize_dataset(dataset_name, path, regenerate)

    return load_datasets(
        metadata=dataset_path, train=dataset_path / "train", test=dataset_path / "test",
    )
