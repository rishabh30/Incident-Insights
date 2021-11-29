"""Unearthed Prediction Template"""
import logging
import argparse
from os import getenv
from os.path import join
import pandas as pd

from preprocess import preprocess
from train import model_fn, target_columns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    """Prediction.

    The main function is only used by the Unearthed CLI.

    When a submission is made online AWS SageMaker Processing Jobs are used to perform
    preprocessing and Batch Transform Jobs are used to pass the result of preprocessing
    to the trained model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, default=getenv("SM_MODEL_DIR", "/opt/ml/models")
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/opt/ml/output/public.csv.gz",
    )
    args, _ = parser.parse_known_args()

    # call preprocessing on the data
    df = preprocess(join(args.data_dir, "public.csv.gz"), False)

    # load the model
    model = model_fn(args.model_dir)

    # pass the model the preprocessed data
    logger.info("creating predictions")
    predictions = model.predict(df)
    logger.info(f"predictions have shape of {predictions.shape}")
    logger.info(f"top 1000 predictions are {predictions[:1000]}") #delete

    # save the predictions
    pd.DataFrame(predictions, columns=['incident'], index=df.index).to_csv(args.output, index=False, header=False)
