from mage_ai.io.file import FileIO
from pandas import DataFrame
import mlflow
import pickle

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

mlflow.set_tracking_uri("file:///home/src/homework_03/mlruns")
mlflow.set_experiment("yellow-taxi-data")

@data_exporter
def export_data(output, **kwargs) -> None:
    """
    Template for exporting data to filesystem.

    Docs: https://docs.mage.ai/design/data-loading#fileio
    """
    
    with mlflow.start_run():
        
        count = 0
        for dv in output[:2]:
            with open('preprocessor_{}.b'.format(count), "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact("preprocessor_{}.b".format(count), artifact_path="dv")
            count += 1

        mlflow.sklearn.log_model(sk_model= output[2], artifact_path="artifacts")
