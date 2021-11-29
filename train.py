"""Unearthed Training Template"""
import argparse
import logging
import pickle
import sys
from io import StringIO
from os import getenv
from os.path import abspath, join
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from preprocess import preprocess, oversample

random_state = 42
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

target_columns = [
    "incident",
]

# Work around for a SageMaker path issue
# (see https://github.com/aws/sagemaker-python-sdk/issues/648)
# WARNING - removing this may cause the submission process to fail
if abspath("/opt/ml/code") not in sys.path:
    sys.path.append(abspath("/opt/ml/code"))


def train(args):
    """Train

    Your model code goes here.
    """
    logger.info("calling training function")

    # If you require any particular preprocessing to create features then this
    # *must* be contained in the preprocessing function for the Unearthed pipeline
    # apply it to the private data
    df = preprocess(join(args.data_dir, "public.csv.gz"))

    # our ExplainableBoostingClassifier requires a balanced training dataset. We can acheive this by oversampling. To speed it up we train on only N samples with the positive and negative incidents, with replacement.
    
    df = oversample(df, "incident", n=6050000)
    

    # the target variable is incident
    #y_train = df[target_columns] 
    y_train = df['incident']
    logger.info(f"training target shape is {y_train}")
    #X_train = df.drop(columns=target_columns)
    X_train = df.drop(columns='incident')
    X_train = X_train.reset_index(drop=True)
    X_train = pd.get_dummies(X_train) 
    X_train.drop(columns='TIME_TYPE_Overtime', inplace = True)
    logger.info(f"training columns is {X_train.columns}")
    logger.info(f"training input shape is {X_train.shape}")

    # we use a glassbox model, and put interactions=0 to avoid combining features
    # see https://interpret.ml/docs/getting-started#train-a-glassbox-model
    model_ebc = ExplainableBoostingClassifier(random_state=random_state, interactions=5)
    model_ebc.fit(X_train, y_train)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate = 0.04, max_features=4, max_depth = 3, random_state = random_state)
    model.fit(X_train, y_train)
    model_rfc = RandomForestClassifier(n_estimators = 100,random_state = random_state)
    model_rfc.fit(X_train, y_train)
    
    from sklearn.tree import DecisionTreeClassifier
    boosting_dtree = DecisionTreeClassifier(class_weight='balanced',
                                        criterion='entropy',
                                        max_depth=1, random_state=42)
    model_ada= AdaBoostClassifier(base_estimator=boosting_dtree, n_estimators=50, learning_rate=.04, random_state=random_state) 
    model_ada.fit(X_train, y_train)
    
    import numpy as np
    cls_weight = (y_train.shape[0] - np.sum(y_train)) / np.sum(y_train)
    from xgboost import XGBClassifier
    model_xgb = XGBClassifier(learning_rate=0.01, n_estimators=132, 
                        max_depth=1,
                        scale_pos_weight=cls_weight,
                        random_state=random_state, n_jobs=-1)
        
    from sklearn.ensemble import VotingClassifier
    
    print(X_train)
    model = VotingClassifier(estimators = [('lr', model_gbc), ('rf', model_ada), ('rfc', model_rfc)], voting ='soft')
    print(model.fit(X_train, y_train))

    # save the model to disk
    save_model(model, args.model_dir)


def save_model(model, model_dir):
    """Save model to a binary file.

    This function must write the model to disk in a format that can
    be loaded from the model_fn.

    WARNING - modifying this function may cause the submission process to fail.
    """
    logger.info(f"saving model to {model_dir}")
    with open(join(model_dir, "model.pkl"), "wb") as model_file:
        pickle.dump(model, model_file)


def model_fn(model_dir):
    """Load model from binary file.

    This function loads the model from disk. It is called by SageMaker.

    WARNING - modifying this function may case the submission process to fail.
    """
    logger.info("loading model")
    with open(join(model_dir, "model.pkl"), "rb") as file:
        return pickle.load(file)


def input_fn(input_data, content_type):
    """Take request data and de-serialize the data into an object for prediction.

    In the Unearthed submission pipeline the data is passed as "text/csv". This
    function reads the CSV into a Pandas dataframe ready to be passed to the model.

    WARNING - modifying this function may cause the submission process to fail.
    """
    logger.info("receiving preprocessed input")

    # this call must result in a dataframe or nparray that matches your model
    input = pd.read_csv(StringIO(input_data), index_col=0, parse_dates=True)
    logger.info(f"preprocessed input has shape {input.shape}")
    return input


if __name__ == "__main__":
    """Training Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed train" command.

    WARNING - modifying this function may cause the submission process to fail.

    The main function must call preprocess, arrange th
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
    train(parser.parse_args())
