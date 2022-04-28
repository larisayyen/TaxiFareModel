
FROM python:3.8.6-buster

WORKDIR /fare_amount_predictor


# the trained model
COPY model.joblib /fare_amount_predictor/model.joblib
# the code of the project which is required in order to load the model
COPY TaxiFareModel /fare_amount_predictor/TaxiFareModel
# the code of our API
COPY api /fare_amount_predictor/api
# the list of requirements
COPY requirements.txt /fare_amount_predictor/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
