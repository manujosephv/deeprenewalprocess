"""Console script for deeprenewal."""
import random
import os
import numpy as np
import mxnet as mx
from pathlib import Path
from tqdm.autonotebook import tqdm

def seed_everything():
    random.seed(42)
    np.random.seed(42)
    mx.random.seed(42)


seed_everything()

from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from argparse import ArgumentParser
import sys
import ast
from deeprenewal import get_dataset, DeepRenewalEstimator, IntermittentEvaluator


def evaluate_forecast(args):
    is_gpu = mx.context.num_gpus() > 0
    dataset = get_dataset(args.datasource, regenerate=args.regenerate_datasource)
    prediction_length = dataset.metadata.prediction_length
    freq = dataset.metadata.freq
    cardinality = (
        ast.literal_eval(dataset.metadata.feat_static_cat[0].cardinality)
        if args.use_feat_static_cat
        else None
    )
    train_ds = dataset.train
    test_ds = dataset.test
    trainer = Trainer(
        ctx=mx.context.gpu() if is_gpu & args.use_cuda else mx.context.cpu(),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.max_epochs,
        num_batches_per_epoch=args.number_of_batches_per_epoch,
        clip_gradient=args.clip_gradient,
        weight_decay=args.weight_decay,
        hybridize=True,
    )  # hybridize false for development

    estimator = DeepRenewalEstimator(
        prediction_length=prediction_length,
        context_length=prediction_length * args.context_length_multiplier,
        num_layers=args.num_layers,
        num_cells=args.num_cells,
        cell_type=args.cell_type,
        dropout_rate=args.dropout_rate,
        scaling=True,
        lags_seq=np.arange(1, args.num_lags + 1).tolist(),
        freq=freq,
        use_feat_dynamic_real=args.use_feat_dynamic_real,
        use_feat_static_cat=args.use_feat_static_cat,
        use_feat_static_real=args.use_feat_static_real,
        cardinality=cardinality if args.use_feat_static_cat else None,
        trainer=trainer,
    )
    predictor = estimator.train(train_ds, test_ds)
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(Path(args.model_save_dir)/"deep_renewal", exist_ok=True)
    predictor.serialize(Path(args.model_save_dir)/"deep_renewal")
    print("Generating DeepRenewal forecasts.......")
    # predictor = Predictor.deserialize(Path(args.model_save_dir)/"deeprenewal")
    deep_renewal_flat_forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds, predictor=predictor, num_samples=100
    )
    tss = list(tqdm(ts_it, total=len(test_ds)))
    deep_renewal_flat_forecasts = list(tqdm(deep_renewal_flat_forecast_it, total=len(test_ds)))
    evaluator = IntermittentEvaluator(quantiles=[0.25,0.5,0.75], median=args.point_forecast=="median", calculate_spec=args.calculate_spec)
    deep_renewal_flat_agg_metrics, deep_renewal_flat_item_metrics = evaluator(
        iter(tss), iter(deep_renewal_flat_forecasts), num_series=len(test_ds)
    )
    print(deep_renewal_flat_agg_metrics)



def main():
    """Console script for deeprenewal."""
    parser = ArgumentParser()
    # PROGRAM level args
    parser.add_argument("--use-cuda", type=bool, default=True)
    parser.add_argument("--datasource", type=str, default="retail_dataset")
    parser.add_argument("--regenerate-datasource", type=bool, default=False)
    parser.add_argument("--model-save-dir", type=str, default="saved_models")
    parser.add_argument("--point-forecast", type=str, default="mean", choices=['median', 'mean'])
    parser.add_argument("--calculate-spec", type=bool, default=False)

    # Trainer specific args
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--number-of-batches-per-epoch", type=int, default=100)
    parser.add_argument("--clip-gradient", type=float, default=5.170127652392614)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # Model specific args
    parser.add_argument("--context-length-multiplier", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-cells", type=int, default=64)
    parser.add_argument("--cell-type", type=str, default="lstm")
    # p% are dropped and set to zero
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--use-feat-dynamic-real", type=bool, default=False)
    parser.add_argument("--use-feat-static-cat", type=bool, default=False)
    parser.add_argument("--use-feat-static-real", type=bool, default=False)
    parser.add_argument("--scaling", type=bool, default=True)
    parser.add_argument("--num-parallel-samples", type=int, default=100)
    parser.add_argument("--num-lags", type=int, default=1)
    parser.add_argument("--forecast-type", type=str, default="hybrid")
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    evaluate_forecast(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
