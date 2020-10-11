from deeprenewal._datasets import get_dataset
from gluonts.dataset.util import to_pandas
from deeprenewal.deeprenewal._estimator import DeepRenewalEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.evaluation import Evaluator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
import mxnet as mx
import ast
from tqdm import tqdm

dataset = get_dataset("retail_dataset", regenerate=False)

print(dataset)

prediction_length = dataset.metadata.prediction_length
freq = dataset.metadata.freq
cardinality = ast.literal_eval(dataset.metadata.feat_static_cat[0].cardinality)
train_ds = dataset.train
test_ds = dataset.test
lr_epoch_map = {1e-5: 3, 1e-4: 25, 1e-3: 3, 1e-2: 3, 1e-1: 3}

# from croston import croston
# import numpy as np
# ts = next(iter(train_ds))['target']
# fit_pred = croston.fit_croston(ts, 10,'original', number_parameters=2)
# yhat = np.concatenate([fit_pred['croston_fittedvalues'], fit_pred['croston_forecast']])

lr = 1e-4
epoch_multiplier = 1

trainer_best_params = {
    "batch_size": 256 ,
    "learning_rate": lr,
    "epochs": 1,
    "num_batches_per_epoch": 100,
    "clip_gradient": "7.71224451133543",
    "weight_decay": 1e-5,
}

estimator_best_params = {
    "context_length": prediction_length,
    "num_layers": 1,
    "num_cells": 32,
    "cell_type": "gru",
    "dropout_rate": 0.3,
    # "distr_output": PiecewiseLinearOutput(num_pieces=4),
}

trainer = Trainer(ctx=mx.context.gpu(), **trainer_best_params,hybridize=True) #hybridize false for development

# estimator = DeepAREstimator(
#     prediction_length=prediction_length,
#     scaling=True,
#     lags_seq=[1],
#     freq=freq,
#     use_feat_dynamic_real=True,
#     use_feat_static_cat=True,
#     cardinality=cardinality,
#     trainer=trainer,
#     **estimator_best_params
# )

estimator = DeepRenewalEstimator(
    prediction_length=prediction_length,
    scaling=True,
    lags_seq=[1],
    freq=freq,
    use_feat_dynamic_real=True,
    use_feat_static_cat=True,
    cardinality=cardinality,
    trainer=trainer,
    forecast_type = 'flat',
    **estimator_best_params
)
predictor = estimator.train(train_ds, test_ds)


# # save the trained model in tmp/
import os

# os.makedirs("model_saved", exist_ok=True)
from pathlib import Path

predictor.serialize(Path("saved_models/deep_renewal_process"))

# loads it back
from gluonts.model.predictor import Predictor

predictor = Predictor.deserialize(Path("saved_models/deep_renewal_process"))
# predictor_deserialized.forecast_generator.forecast_type = "exact"
print("Generating forecasts.......")
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds, predictor=predictor, num_samples=100
)

# tss = list(tqdm(ts_it, total=len(test_ds)))
# forecasts = list(tqdm(forecast_it, total=len(test_ds)))
# print(forecasts[0].mean_ts)
# print(forecasts[0].median)
# print(tss[0].tail())
# from gluonts.model.forecast import SampleForecast
# from croston import croston


# def generate_croston_forecast(train_ds, prediction_length, freq, **kwargs):
#     croston_variant = kwargs.get("croston_variant", "original")
#     number_parameters = kwargs.get("number_parameters", 2)
#     for ts in iter(train_ds):
#         target = ts["target"]
#         fit_pred = croston.fit_croston(
#             target,
#             forecast_length=prediction_length,
#             croston_variant=croston_variant,
#             number_parameters=number_parameters,
#         )
#         yhat = fit_pred["croston_forecast"]
#         sample_fc = SampleForecast(
#             samples=yhat.reshape(1, -1),
#             start_date=ts["start"],
#             freq=freq,
#             item_id=ts["item_id"],
#         )
#         yield sample_fc


# croston_forecast = (
#     generate_croston_forecast(
#         train_ds,
#         prediction_length,
#         freq,
#         croston_variant="original",
#         number_parameters=2,
#     )
# )

from deeprenewal._evaluator import IntermittentEvaluator
evaluator = IntermittentEvaluator(quantiles=[0.25, 0.5, 0.75])

# agg_metrics, item_metrics = evaluator(
#     iter(tss), iter(forecasts), num_series=len(test_ds)
# )
agg_metrics, item_metrics = evaluator(
    ts_it, forecast_it, num_series=len(test_ds)
)
print(agg_metrics)
