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
from mlflow.data import from_pandas
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec

mlflow.set_tracking_uri("")

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--solver", type=str, default="lbfgs")
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
        experiment_name=f"iris-classification-ml-model_{datetime.now().strftime('%Y-%m-%d')}"
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
        
        X_train_complete = pd.concat([X_train, y_train], axis=1)
        X_test_complete = pd.concat([X_test, y_test], axis = 1)
        
        X_train_complete_mlflow = from_pandas(X_train_complete)
        X_test_complete_mlflow = from_pandas(X_test_complete)
        
        mlflow.log_input(
            dataset = X_train_complete_mlflow,
            context = "training"
        )
        
        mlflow.log_input(
            dataset = X_test_complete_mlflow,
            context = "testing"
        )
        
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
        
        # input_data =[
        #     {"type": "double", "name": "sepal.length"}, 
        #     {"type":"double", "name": "sepal.width"}, 
        #     {"type": "double", "name":"petal.length"}, 
        #     {"type": "double", "name": "petal.width"}
        # ]
        
        # output_data = [
        #     {"type": "string"}
        # ]
        
        # input_schema = Schema(
        #     [ColSpec(col["type"], col["name"]) for col in input_data]
        # )
        
        # output_schema = Schema(
        #     [ColSpec(col["type"]) for col in output_data]
        # )
        
        # input_example = {
        #     "sepal.length":5.7,
        #     "sepal.width":3.0,
        #     "petal.length":4.2,
        #     "petal.width":1.2
        # }
        
        # #output_example = []
        
        # signature = ModelSignature(
        #     inputs = input_schema,
        #     outputs=output_schema
        # )
        
        signature = infer_signature(
            X_test,
            y_predicted
        )
        
        input_example = {
            "columns": np.array(X_test.columns),
            "data": X_test.to_numpy()
        }
        
        mlflow.sklearn.log_model(
            sk_model=lm,
            artifact_path="iris_model_artifacts",
            signature=signature,
            input_example=input_example
        )

        mlflow.set_tag(key="end", value = datetime.now().strftime("%Y-%m-%d::%H:%M:%S"))
        
    print(f"Last active run: {mlflow.last_active_run().info.run_name}")
        

