# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import logging
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import mlflow
from mlflow.models import infer_signature
from importlib.metadata import version

logging.basicConfig(level=logging.INFO)

# packages the model needs at inference time (pinned to the versions used for training)
SERVING_PACKAGES = ("mlflow", "scikit-learn", "skops", "numpy", "scipy", "pandas")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # MLflow related parameters
    parser.add_argument("--tracking_uri", type=str)
    parser.add_argument("--experiment_name", type=str)
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    # Data directories (SageMaker sets SM_CHANNEL_<NAME> for each input channel)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--target", type=str, default="MedHouseVal")
    args, _ = parser.parse_known_args()

    logging.info("reading data")
    train_df = pd.read_csv(os.path.join(args.train, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.test, "test.csv"))
    X_train, y_train = train_df.drop(columns=[args.target]), train_df[args.target]
    X_test, y_test = test_df.drop(columns=[args.target]), test_df[args.target]

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        mlflow.set_tag("sagemaker_job", os.environ.get("TRAINING_JOB_NAME", "local"))
        mlflow.log_params({"n_estimators": args.n_estimators, "min_samples_leaf": args.min_samples_leaf})
        mlflow.log_input(mlflow.data.from_pandas(train_df, targets=args.target, name="california-train"), context="training")

        logging.info("training model")
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # written to the job logs so SageMaker HPO can read it (see metric_definitions in lab 2)
        rmse = root_mean_squared_error(y_test, model.predict(X_test))
        logging.info(f"test-rmse: {rmse}")

        logging.info("logging model to MLflow")
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_test, model.predict(X_test)),
            input_example=X_test.head(3),
            # MLflow saves sklearn models with skops (safe format); the forest's tree class must be trusted explicitly
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            # pin the serving dependencies explicitly: MLflow's automatic inference picks up stale
            # packages from the SageMaker training image (e.g. typing==3.10.0.0) that fail to install
            pip_requirements=[f"{pkg}=={version(pkg)}" for pkg in SERVING_PACKAGES],
        )

        # logs r2, rmse, mae, mape, ... linked to the run and the model
        mlflow.models.evaluate(
            model_info.model_uri,
            data=test_df,
            targets=args.target,
            model_type="regressor",
        )
