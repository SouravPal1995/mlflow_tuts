import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score, 
    accuracy_score, 
)

import argparse

from datetime import datetime

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("./mlruns_iris")

parser = argparse.ArgumentParser()

parser.add_argument("--solver", type=str, default="lbfgs")
parser.add_argument("--penalty", type=str, default="l2")
parser.add_argument("--multi-class", type=str, default="ovr")
parser.add_argument("--C", type=float, default=1)
parser.add_argument("--max-iter", type=int, default=100)

args = parser.parse_args()

if __name__=="__main__":
    
    data = pd.read_csv("./data/iris.csv")
    y = data["variety"]
    X = data[[column for column in data.columns if column!="variety"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=0.8,
        random_state = 20
    )

    X_train.to_csv("./iris_data_artifacts/X_train.csv")
    X_test.to_csv("./iris_data_artifacts/X_test.csv")
    y_train.to_csv("./iris_data_artifacts/y_train.csv")
    y_test.to_csv("./iris_data_artifacts/y_test.csv")

    experiment = mlflow.set_experiment(
        experiment_name="iris-classification-ml-model"
    )

    with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name = f"iris-run-{datetime.now().strftime('%Y-%m-%d::%H:%M:%S')}",
            nested=False,
            tags = {
                "version":"1",
                "data scientist": "Sourav Pal"
            },
            description='''
                A simple Scikit-learn Model to classify flowers using sepal and petal length/width data.
            '''
        ) as run:

        mlflow.set_tag(key="start", value = datetime.now().strftime("%Y-%m-%d::%H:%M:%S"))
        
        lm = LogisticRegression(solver = args.solver,
                                penalty=args.penalty,
                                multi_class=args.multi_class,
                                C=args.C,
                                max_iter=args.max_iter
        )

        lm.fit(X_train, y_train)

        y_predicted = lm.predict(X_test)
        
        pd.DataFrame(data = {"variety":y_predicted}).to_csv("./iris_data_artifacts/y_predicted.csv")
        
        mlflow.log_artifacts(
            local_dir="./iris_data_artifacts",
            artifact_path="iris_data_artifacts"
        )

        precision_macro = precision_score(y_test, y_predicted, average="macro")
        recall_macro = recall_score(y_test, y_predicted, average="macro")
        f1_macro = f1_score(y_test, y_predicted, average="macro")
        accuracy = accuracy_score(y_test, y_predicted)

        print(f"Precision-Macro: {precision_macro}")
        print(f"Recall-Macro: {recall_macro}")
        print(f"F1-Macro: {f1_macro}")
        print(f"accuracy: {accuracy}")
        
        
        mlflow.log_params({
            "solver": args.solver,
            "penalty": args.penalty,
            "multi_class": args.multi_class,
            "C": args.C,
            "max_iter":args.max_iter
        })
        
        mlflow.log_metrics({
            "Precision-Macro": precision_macro,
            "Recall-Macro": recall_macro,
            "F1-Macro": f1_macro,
            "accuracy":accuracy
        })
        
        mlflow.sklearn.log_model(
            sk_model=lm,
            artifact_path="iris_model_artifacts"
        )

        mlflow.set_tag(key="end", value = datetime.now().strftime("%Y-%m-%d::%H:%M:%S"))
        
    print(f"Last active run: {mlflow.last_active_run().info.run_name}")
        

