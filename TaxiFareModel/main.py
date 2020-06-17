import warnings
from copy import copy

from termcolor import colored

from TaxiFareModel.data import clean_df, get_data
from TaxiFareModel.trainer import Trainer

####################################
#  generation liste de parametres  #
####################################
DEFAULT_PARAMS = dict(nrows=100_000,
                      local=True,  # set to False to get data from GCP (Storage or BigQuery)
                      optimize=True,
                      estimator="xgboost",
                      mlflow=True,  # set to True to log params to mlflow
                      pipeline_memory=None,
                      distance_type="manhattan",
                      feateng=["distance_to_center", "direction", "distance", "time_features", "geohash"])


def feat_eng_experiment(default_params=DEFAULT_PARAMS):
    new_params = copy(default_params)
    new_params["experiment_name"] = "feateng"
    new_params["estimator"] = "RandomForest"
    l_params = []
    for feateng in ["distance_to_center", "direction", "distance", "time_features", "geohash"]:
        params = copy(new_params)
        params["feateng"] = [feateng]
        l_params.append(params)
    return l_params


def model_experiment(default_params=DEFAULT_PARAMS):
    """
    Beware here, depending on number of row in DEFAULT_PARAMS,
    It might take a while to run
    """
    new_params = copy(default_params)
    new_params["experiment_name"] = "model"
    l_params = []
    for model in ["Lasso", "Ridge", "Linear", "GBM", "RandomForest", "xgboost"]:
        params = copy(new_params)
        params["estimator"] = model
        l_params.append(params)
    return l_params


def distance_experiment(default_params=DEFAULT_PARAMS):
    new_params = copy(default_params)
    new_params["experiment_name"] = "distance"
    new_params["estimator"] = "xgboost"
    l_params = []
    for distance_type in ["haversine", "euclidian", "manhattan"]:
        params = copy(new_params)
        params["distance_type"] = distance_type
        l_params.append(params)
    return l_params


####################################
#        main functions            #
####################################
def load_data(params):
    print("############   Loading Data   ############")
    df = get_data(**params)
    df = clean_df(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    print("shape: {}".format(X.shape))
    print("size: {} Mb".format(X.memory_usage().sum() / 1e6))
    return X, y


def workflow(X, y, params):
    t = Trainer(X=X, y=y, **params)
    del X, y
    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()
    print(colored("############   Saving model    ############", "green"))
    t.save_model()


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get params list
    l_params = [distance_experiment(), model_experiment(), feat_eng_experiment()]
    # Get data
    X, y = load_data(l_params[0][0])
    for l_param in l_params:
        for params in l_param:
            workflow(X, y, params)
