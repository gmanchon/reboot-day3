import pathlib

import pandas as pd

from TaxiFareModel.utils import simple_time_tracker

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"
LOCAL_PATH = pathlib.Path(__file__).parent.parent.absolute() / 'data' / 'taxi-fare-train.csv'

DIST_ARGS = dict(start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude")


@simple_time_tracker
def get_data(nrows=10_000, local=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    if local:
        path = LOCAL_PATH
    else:
        path = AWS_BUCKET_PATH
    df = pd.read_csv(path, nrows=nrows)

    # write csv to disk
    if not local:
        df.to_csv(LOCAL_PATH)

    return df


def clean_df(df, test=False):
    """ Cleaning Data based on Kaggle test sample
    - remove high fare amount data points
    - keep samples where coordinate wihtin test range
    """
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


if __name__ == "__main__":
    params = dict(nrows=1_000,
                  local=False,  # set to False to get data from GCP (Storage or BigQuery)
                  )
    df = get_data(**params)
