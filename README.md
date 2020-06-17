# Iterate with MlFlow

Now that we have already built a simple model, we want to make it better! The ultimate goal is to have a model that makes more accurate predictions on the test set, hence getting a `RMSE` as low as possible.

**So what can we do?**

There are many different things that make models better:
- Build and try to use different or more features
- Test with different estimators (linear, non linear, etc.)
- Test influence of distance definition

## Installation and MlFlow setup

```bash
make install
```

1. If you want to run mlflow locally:
- Open a new terminal window and run `mlflow ui` at the root of this repository
- See the results [in your browser](http://localhost:5000/)

2. If you want to log parameters to remote instance
- modify `MLFLOW_URI` inside `trainer.py`

## Parameters to set

- Inspect how parameters are passed to the `Trainer()` instance inside `TaxiFareModel/trainer.py`
- We will play with these parameters to run different experiments

## Experiments

Inside `main.py`, we define 3 experiments, for each experiment we generate a list of parameters, i.e:

```python
def distance_experiment(default_params=DEFAULT_PARAMS):
    new_params = copy(default_params)
    new_params["experiment_name"] = "distance"
    new_params["estimator"] = "RandomForest"
    l_params = []
    for distance_type in ["haversine", "euclidian", "manhattan"]:
        params = copy(new_params)
        params["distance_type"] = distance_type
        l_params.append(params)
    return l_params
```

## How to use

- Choose experiments amongst `distance_experiment`, `model_experiment` and `feat_eng_experiment`
- generate a list of parameters
- run the workflow for each set of parameters
