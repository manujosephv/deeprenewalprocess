#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import os
import numpy as np
import mxnet as mx

def seed_everything():
    random.seed(42)
    np.random.seed(42)
    mx.random.seed(42)
    
seed_everything()

from models._datasets import get_dataset
from gluonts.dataset.util import to_pandas
from models.deeprenewal._estimator import DeepRenewalEstimator
from models._evaluator import IntermittentEvaluator
from gluonts.model.deepar import DeepAREstimator
from gluonts.distribution.neg_binomial import NegativeBinomialOutput
from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
from gluonts.distribution.student_t import StudentTOutput
from gluonts.model.r_forecast import RForecastPredictor
from gluonts.model.prophet import ProphetPredictor
from gluonts.evaluation import Evaluator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions 

import mxnet as mx
import ast
from tqdm import tqdm
from argparse import ArgumentParser
from gluonts.model.forecast import SampleForecast
# from croston import croston
from fbprophet import Prophet
from pathlib import Path
from gluonts.model.predictor import Predictor
import pandas as pd

import optuna
import wandb


# In[2]:


wandb.init(name="OptunaRunDeepAR", project ="DeepRenewal")


# # Functions

# In[9]:


lr_epoch_map = {1e-5: 25, 1e-4: 20, 1e-3: 15, 1e-2: 10, 1e-1: 5}
# lr_epoch_map = {1e-5: 3, 1e-4: 3, 1e-3: 3, 1e-2: 3, 1e-1: 3}


def get_trainer_from_trial(trial):
    lr = 1e-2
    # epoch_multiplier = 1
    epoch_multiplier = trial.suggest_int("epoch_multiplier", 1, 2)
    epochs = lr_epoch_map[lr] * epoch_multiplier
    # epochs = 2
    bs = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    # bs = 1
    clip_gradient = trial.suggest_uniform("clip_gradient", 5, 10)
    weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3, 1e-2])
    trainer = Trainer(ctx=mx.context.gpu(0),
        learning_rate=lr,
        epochs=epochs,
        num_batches_per_epoch=100,
        batch_size=bs,
        clip_gradient=clip_gradient,
        weight_decay=weight_decay,
        # hybridize=False,
    )
    return trainer

def get_deepar_params_from_trial(self, trial):
        context_length_multiplier = trial.suggest_categorical(
            "deepar_context_length_multiplier", gc.context_length_multiplier_range
        )
        context_length = context_length_multiplier * self.prediction_length
        num_layers = trial.suggest_int("deepar_num_layers", *gc.deepar_num_layers_range)
        num_cells = trial.suggest_categorical("deepar_num_cells", gc.deepar_num_cells_range)
        cell_type = trial.suggest_categorical("deepar_cell_type", gc.deepar_cell_type_range)
        dropout = trial.suggest_int("deepar_dropout", *gc.deepar_dropout_range) * 1e-1
        
        distribution_type = trial.suggest_categorical(
            "deepar_distribution_type", gc.distribution_type_range
        )
        if distribution_type == "piecewise":
            num_pieces = trial.suggest_int("deepar_piecewise_num_pieces", *gc.num_pieces_range)
            dist_output = PiecewiseLinearOutput(num_pieces)
        elif distribution_type == "negative_binomial":
            dist_output = NegativeBinomialOutput()
        elif distribution_type == "student_t":
            dist_output = StudentTOutput()
        else:
            raise NotImplementedError(f"{distribution_type} Not implemented")
        params = dict(
            context_length=context_length,
            num_layers=num_layers,
            num_cells=num_cells,
            cell_type=cell_type,
            dropout_rate=dropout,
            distr_output=dist_output,
        )
        return params

def get_estimator_from_trial(trial, trainer):

    context_length_multiplier = trial.suggest_categorical(
        "context_length_multiplier", [1, 2, 3, 4]
    )
    context_length = context_length_multiplier * prediction_length
    num_layers = trial.suggest_int("num_layers", 2, 5)
    num_lags = trial.suggest_int("num_lags", 0, 12)
    lags_seq = None if num_lags==0 else np.arange(1,num_lags+1).tolist()
    num_cells = trial.suggest_categorical("num_cells", [32, 64, 128])
    cell_type = trial.suggest_categorical("cell_type", ["lstm", "gru"])
    dropout_rate = trial.suggest_int("dropout", 1, 5) * 1e-1
    use_feat_static_cat = trial.suggest_categorical("deeprenewal_use_feat_static_cat", [True, False])
    use_feat_dynamic_real = trial.suggest_categorical("deeprenewal_use_feat_dynamic_real", [True, False])
    
    distribution_type = trial.suggest_categorical(
        "distribution_type", ['piecewise','negative_binomial','student_t']
    )
    if distribution_type == "piecewise":
        num_pieces = trial.suggest_int("deepar_piecewise_num_pieces", *[2,7])
        dist_output = PiecewiseLinearOutput(num_pieces)
    elif distribution_type == "negative_binomial":
        dist_output = NegativeBinomialOutput()
    elif distribution_type == "student_t":
        dist_output = StudentTOutput()
    else:
        raise NotImplementedError(f"{distribution_type} Not implemented")

    estimator = DeepAREstimator(
        prediction_length=prediction_length,
        context_length=context_length,
        num_layers=num_layers,
        num_cells=num_cells,
        cell_type=cell_type,
        dropout_rate=dropout_rate,
        distr_output=dist_output,
        scaling=True,
        lags_seq=lags_seq,
        freq="1D", ##Freq,
        use_feat_dynamic_real=use_feat_dynamic_real,
        use_feat_static_cat=use_feat_static_cat,
        use_feat_static_real=args.use_feat_static_real,
        cardinality=cardinality if use_feat_static_cat else None,
        trainer=trainer,
        )
    return estimator


def objective(trial):
    trainer = get_trainer_from_trial(trial)
    estimator = get_estimator_from_trial(trial, trainer)
    try:
        predictor = estimator.train(train_ds, test_ds)
        # ### Generating forecasts
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds, predictor=predictor, num_samples=100
        )

        tss = list(tqdm(ts_it, total=len(test_ds)))
        forecasts = list(tqdm(forecast_it, total=len(test_ds)))
        # ### Local performance validation
        evaluator = IntermittentEvaluator(quantiles=[0.5], calculate_spec=False)
        agg_metrics, item_metrics = evaluator(
            iter(tss), iter(forecasts), num_series=len(test_ds)
        )
        
        nrmse = agg_metrics["NRMSE"]
        pis = agg_metrics["PIS"]
        mape = agg_metrics["MAPE"]
        maape = agg_metrics["MAAPE"]
        trial.set_user_attr("MAPE",mape)
        trial.set_user_attr("MAAPE",maape)
        trial.set_user_attr("PIS",pis)
        wandb.log({"nrmse": nrmse,
                  "PIS": pis,
                  "MAPE":mape,
                  "MAAPE":maape,
                  "params":trial.params})
    except Exception as e:
        raise e
        trial.set_user_attr("exception", e)
        nrmse = 1e20
        wandb.log({"NRMSE": nrmse})
    return nrmse


# # Config

# In[10]:


parser = ArgumentParser()
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

# add PROGRAM level args
parser.add_argument('--project-name', type=str, default='deep_renewal_processes')
parser.add_argument('--experiment-tag', type=str, default='deep_renewal_process')
parser.add_argument('--use-cuda', type=bool, default=True)
parser.add_argument('--use-wandb', type=bool, default=True)
parser.add_argument('--log-gradients', type=bool, default=True)
parser.add_argument('--run-optuna-sweep', type=bool, default=True)
parser.add_argument('--datasource', type=str, default="retail_dataset")
parser.add_argument('--model-save-dir', type=str, default="models")

# Trainer specific args
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--learning-rate', type=float, default=1e-5)
parser.add_argument('--max-epochs', type=int, default=25)
parser.add_argument('--number-of-batches-per-epoch', type=int, default=100)
parser.add_argument('--clip-gradient', type=float, default=10)
parser.add_argument('--weight-decay', type=float, default=1e-8)


# Model specific args
parser.add_argument('--context-length-multiplier', type=int, default=2)
parser.add_argument('--num-layers', type=int, default=4)
parser.add_argument('--num-cells', type=int, default=128)
parser.add_argument('--cell-type', type=str, default="gru")
#p% are dropped and set to zero
parser.add_argument('--dropout-rate', type=float, default=0.3)
parser.add_argument('--use-feat-dynamic-real', type=bool, default=True)
parser.add_argument('--use-feat-static-cat', type=bool, default=True)
parser.add_argument('--use-feat-static-real', type=bool, default=False)
parser.add_argument('--scaling', type=bool, default=True)
parser.add_argument('--num-parallel-samples', type=int, default=100)
parser.add_argument('--num-lags', type=int, default=1)
#Only for Deep Renewal Processes
parser.add_argument('--forecast-type', type=str, default="flat")
#Only for Deep AR
parser.add_argument('--distr-output', type=str, default="student_t") #neg_binomial


args = parser.parse_args()
is_gpu = mx.context.num_gpus()>0


# # Read in the dataset

# In[11]:


dataset = get_dataset(args.datasource, regenerate=False)


# In[12]:


prediction_length = dataset.metadata.prediction_length
freq = dataset.metadata.freq
cardinality = ast.literal_eval(dataset.metadata.feat_static_cat[0].cardinality)
train_ds = dataset.train
test_ds = dataset.test


# In[13]:


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

df = study.trials_dataframe()
# if len(filters.values())>0:
#     tag = "|".join(filters.values())
# else:
#     tag = 'all'
tag = 'interm_deepar'
df.to_csv(f"{tag}_study_df.csv")


# In[ ]:




