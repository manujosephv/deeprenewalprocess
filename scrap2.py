import joblib
from deeprenewal._evaluator import IntermittentEvaluator
from deeprenewal._datasets import get_dataset
import ast

dataset = get_dataset("retail_dataset", regenerate=False)

print(dataset)

prediction_length = dataset.metadata.prediction_length
freq = dataset.metadata.freq
cardinality = ast.literal_eval(dataset.metadata.feat_static_cat[0].cardinality)
train_ds = dataset.train
test_ds = dataset.test
lr_epoch_map = {1e-5: 3, 1e-4: 25, 1e-3: 3, 1e-2: 3, 1e-1: 3}

forecast_dict = joblib.load("tmp/forecast_dict.sav")
tss = forecast_dict['tss']
ets_forecast = forecast_dict['ets']
arima_forecast = forecast_dict['arima']
croston_forecast = forecast_dict['croston']
sba_forecast = forecast_dict['sba']
sbj_forecast = forecast_dict['sbj']
npts_forecast = forecast_dict['npts']
deep_ar_forecasts = forecast_dict['deep_ar']
deep_renewal_flat_forecasts = forecast_dict['deep_renewal_flat']
deep_renewal_exact_forecasts = forecast_dict['deep_renewal_exact']
deep_renewal_hybrid_forecasts = forecast_dict['deep_renewal_hybrid']
evaluator = IntermittentEvaluator(quantiles=[0.25,0.5,0.75], median=False, calculate_spec=True)
#DeepAR
deep_ar_agg_metrics, deep_ar_item_metrics = evaluator(
    iter(tss[:10]), iter(deep_ar_forecasts[:10]), num_series=10
)