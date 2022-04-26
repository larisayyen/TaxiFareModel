import pandas as pd

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"
LOCAL_PATH = 'raw_data/train_10k.csv'

DIST_ARGS = dict(start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude")


def get_data(nrows=10000,local=True,**kwargs):
    '''returns a DataFrame with nrows from s3 bucket'''
    if local:
        path = LOCAL_PATH
    else:
        path = AWS_BUCKET_PATH
    df = pd.read_csv(path,nrows=nrows)
    return df

def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


if __name__ == '__main__':
    params = dict(nrows=1000,local=True)
    df = get_data(**params)
