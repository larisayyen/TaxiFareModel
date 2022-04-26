from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
import mlflow

class Mlf():
    MLFLOW_URI = "https://mlflow.lewagon.ai/"

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    @memoized_property
    def ml_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def experiment_id(self):
        try:
            experiment_id = self.ml_client.create_experiment(self.experiment_name)
        except BaseException:
            experiment_id = self.ml_client.get_experiment_by_name(self.experiment_name).experiment_id

        return experiment_id

    def create_run(self):
        self.run = self.ml_client.create_run(self.experiment_id)

    def log_param(self, key, value):
        self.ml_client.log_param(self.run.info.run_id, key, value)

    def log_metric(self, key, value):
        self.ml_client.log_metric(self.run.info.run_id, key, value)
