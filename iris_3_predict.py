import mlflow
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score, 
    accuracy_score, 
)

if __name__=="__main__":
    
    X_test = pd.read_csv("./iris_data_artifacts/X_test.csv", index_col=0)
    y_test = pd.read_csv("./iris_data_artifacts/y_test.csv", index_col=0)
    
    lm = mlflow.pyfunc.load_model(
        "./mlruns_custom_iris_track/680743165053945005/610b33b9f20f4eb085df64a4deeb3649/artifacts/pyfunc_artifact_loc/artifacts\iris.pkl"
    )
    y_predicted = lm.predict(X_test)
    
    precision_macro = precision_score(y_test, y_predicted, average="macro")
    recall_macro = recall_score(y_test, y_predicted, average="macro")
    f1_macro = f1_score(y_test, y_predicted, average="macro")
    accuracy = accuracy_score(y_test, y_predicted)

    print(f"Precision-Macro: {precision_macro}")
    print(f"Recall-Macro: {recall_macro}")
    print(f"F1-Macro: {f1_macro}")
    print(f"accuracy: {accuracy}")
