# imports

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


from TaxiFareModel.encoders import DistanceTransformer,TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data_get import get_data,clean_data

from memoized_property import memoized_property
import mlflow
from  mlflow.tracking import MlflowClient
from TaxiFareModel.mlflow_tooling import Mlf

import joblib

class Trainer():

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearSVR())
        ])


    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X,self.y)
        #print(self.pipeline._get_param_names)

    def grid_search(self):
        params = [
            {"linear_model__C": [1, 10, 100], "linear_model__tol": [0.001,0.01,0.1]}
        ]
        self.cv = GridSearchCV(self.pipeline,params)
        self.cv.fit(self.X,self.y)
        return self.cv.best_params_

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""

        y_pred = self.pipeline.predict(X_test)

        return compute_rmse(y_pred, y_test)

    def track_on_mlf(self, X_test, y_test, cv_params, **kwargs):
        ## Adding MLFlow tracking ##
        mlf_client = Mlf('[CN][SH][login]Taxi03')
        mlf_client.create_run()
        mlf_client.log_metric('rmse', self.evaluate(X_test, y_test))
        kwargs.update(cv_params)
        for key, value in kwargs.items():
            mlf_client.log_param(key, value)

    def save_model(self):
        joblib.dump(self.pipeline, 'model.joblib')
        print("Model saved")

if __name__ == "__main__":
    # get data
    N = 10000
    df = get_data(nrows=N)
    df = clean_data(df)

    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # build pipeline
    trainer = Trainer(X_train,y_train)

    trainer.set_pipeline()

    # train the pipeline
    trainer.run()

    best_params = trainer.grid_search()
    #print(best_params)

    # evaluate the pipeline
    rmse = trainer.track_on_mlf(
        X_val,y_val,best_params,
        n_rows=N,test_split=0.2,
        model='linear_regression', ohe_unknown='ignored'
        )
    rmse = trainer.evaluate(X_val, y_val)
    print(rmse)

    trainer.save_model()
