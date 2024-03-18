# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Example Notebook
# MAGIC This notebook walks through some examples of using mlflow to log sklearn models on databricks
# MAGIC 
# MAGIC ### Cluster Requirements
# MAGIC Installation:
# MAGIC **mssql-jdbc**

# COMMAND ----------

# MAGIC %md
# MAGIC MLflow does not come pre-installed for the notebook. We need to run `pip install` using the `%` command to enable terminal commands. Anytime we install new packages, the kernel restarts and you will need to re-run your notebook cells.

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
import time
import pickle
import mlflow
import pathlib
import re

# COMMAND ----------

# Notebook parameters
seed = 31337

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 1:  Grab Some Data
# MAGIC First, we need soem data to work with. Let's create a jdbc connection and grab some data from the Product Rollover database

# COMMAND ----------

server = "azrpmsqln003.database.windows.net"
database = "ProductRollover"
port = 1443
# TODO: extract into secrets manager
username = 'PythonUser'
password = 'un?Ma^vaa_6_!g5X'
jdbcUrl = f"jdbc:sqlserver://{server};databaseName={database}"
connectionProperties = {
  "user" : username,
  "password" : password,
  "driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}
query = """(select * from STDClaimRouting.vLowInvolvementTraining) a"""

# COMMAND ----------

pyspark_df = spark.read.jdbc(url=jdbcUrl, table=query, properties=connectionProperties)

# COMMAND ----------

# MAGIC %md
# MAGIC ### PysparkSQL
# MAGIC Notice that the spark.read.jdbc() function returned a PysparkSQL dataframe. Pyspark is a python-like spark language that is designed to perform efficient operations using cluster resources. Pyspark objects have their own packages for manipulating them. Pyspark does not work with pandas or sklearn.
# MAGIC * https://databricks.com/glossary/pyspark
# MAGIC * https://spark.apache.org/docs/2.4.0/api/python/index.html

# COMMAND ----------

type(pyspark_df)

# COMMAND ----------

display(pyspark_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### PysparkSQL to Pandas
# MAGIC Let's convert the pyspark dataframe into a pandas dataframe

# COMMAND ----------

df = pyspark_df.toPandas()

# COMMAND ----------

# set the column types
df = df.astype(dtype={'low_involvement': float,
            'n_closures': float,
            'notified_until_assignment': float,
            'work_related': str,
            'complexity': str,
            'age_at_dod': float,
            'special_handling': str,
            'assignment_until_decision': float,
            'created_date': str,
            'notified_date': str,
            'initial_assignment_date': str,
            'initial_decision_date': str,
            'group_id': str,
            'clmno': str,
            'ltd_coverage_flag': str,
            'record_type': str,
            'special_assignment': str,
            'job_classification': str,
            'situs_state': str,
            'job_title': str,
            'performance_guarantee': str,
            'cause': str,
            'diagnosis': str,
            'claim_decision': str})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split the Dataset
# MAGIC Now that we have a pandas dataframe to work with, let's go ahead and perform train/test split on the dataset and set 20% of the data aside of evaluation.

# COMMAND ----------

target = 'low_involvement'
X = df.drop(target, axis=1)
y = df[[target]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2:  Train and Log a SKlearn Model to MLflow
# MAGIC We're ready to train a model using our training dataset (X_train and y_train). Let's start by training a simple random forest model with default parameters.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply Column Transformations
# MAGIC Before we can train our model we need to handle some missing values and transform our categorical data. Most sklearn estimators require that missing values are imputed and categorical variables are encoded.

# COMMAND ----------

numeric = ['notified_until_assignment', 'age_at_dod']
categorical = ['job_classification', 'situs_state', 'cause', 'job_title', 'diagnosis', 'special_handling', 'work_related', 'special_assignment', 'performance_guarantee', 'complexity']

# COMMAND ----------

numeric_df = X_train[numeric]
categorical_df = X_train[categorical]

# COMMAND ----------

# Let's instantiate our transformation strategies
# For numeric variables:
median_imputer = SimpleImputer(strategy='median')
scaler = MinMaxScaler()
# For categorical variables:
constant_imputer = SimpleImputer(strategy='constant')
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# COMMAND ----------

# Let's apply transformer.fit to our data
# For numeric variables:
median_imputer.fit(numeric_df)
scaler.fit(numeric_df)
# For categorical variables:
constant_imputer.fit(categorical_df)
encoder.fit(categorical_df)

# COMMAND ----------

# Let's transform to our data
# For numeric variables:
numeric_df = median_imputer.transform(numeric_df)
numeric_df = scaler.transform(numeric_df)
# For categorical variables:
categorical_df = constant_imputer.transform(categorical_df)
categorical_df = encoder.transform(categorical_df)

# COMMAND ----------

# We need to combine the dataframes for training. Transformers return numpy arrays so let's convert them to dataframes to concatenate them
X_train_transformed = pd.concat([pd.DataFrame(numeric_df), 
                                 pd.DataFrame(categorical_df)], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train and Log Model
# MAGIC We are now ready to input our X_train data into the model. 
# MAGIC 
# MAGIC Note: future data that we can predict on must have the same transformations applied in order for the model to make predicitons. This can be made easier with a Pipeline (disucssed in the next section).

# COMMAND ----------

rf = RandomForestClassifier()

# COMMAND ----------

# MAGIC %md
# MAGIC MLflow has an API for sklearn models. If we enable autologging prior to starting the run, we can automatically log metrics to MLflow.
# MAGIC * https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html
# MAGIC 
# MAGIC Note: there are limitations to autologging (explained in above link), so I have created a custom function with similar functionality. Relevant things to know are that it only logs the top 5 models unless otherwise set, it cannot rescore custom scoring metrics, and you cannot modify which metrics and parameters get logged from cv_results

# COMMAND ----------

mlflow.sklearn.autolog()

# COMMAND ----------

with mlflow.start_run() as run:
    rf.fit(X_train_transformed, np.ravel(y_train))

# COMMAND ----------

def fetch_logged_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts

# COMMAND ----------

params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)

# COMMAND ----------

print(metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Manual Logging to MLflow
# MAGIC We can also manually log our own metrics if we would like. The mlflow auto logger is a convenience function that takes care of some of this automatically and performs these operations under the hood.

# COMMAND ----------

def my_pipeline(X, numeric, categorical):
    numeric_df = X[numeric]
    categorical_df = X[categorical]
    # for numeric variables:
    numeric_df = median_imputer.transform(numeric_df)
    numeric_df = scaler.transform(numeric_df)
    # For categorical variables:
    categorical_df = constant_imputer.transform(categorical_df)
    categorical_df = encoder.transform(categorical_df)
    return pd.concat([pd.DataFrame(numeric_df), 
                      pd.DataFrame(categorical_df)], axis=1)

# COMMAND ----------

transformed_train_df = my_pipeline(X_train, numeric, categorical)
train_preds = rf.predict(transformed_train_df)
train_score = f1_score(y_train, train_preds)

# COMMAND ----------

with mlflow.start_run() as run:  
    mlflow.sklearn.log_model(rf, 'rf_manual_model_logged')

    mlflow.log_param(str(100), 'n_estimators')
        
    # score on the training set - I think this is what mlflow does
    transformed_train_df = my_pipeline(X_train, numeric, categorical)
    train_preds = rf.predict(transformed_train_df)
    train_score = f1_score(y_train, train_preds)
    mlflow.log_metric('training_f1_score', train_score)

    # evaluate on the test set
    transformed_test_df = my_pipeline(X_test, numeric, categorical)
    test_preds = rf.predict(transformed_test_df)
    test_score = f1_score(y_test, test_preds)
    mlflow.log_metric('validation_f1_score', test_score, 4)

    mlflow.set_tags({'example':'tag'}) 

    print(f"Done logging.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 3:  Train and Log Multiple Models
# MAGIC Applying transformations can be done in a cleaner and more consistent way in sklearn by using a Pipeline. This is particularly useful when we want to train multiple model types and/or multiple model parameters.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set Up a Model Pipeline
# MAGIC We're goign to set up the same transformations from above in a model pipeline.

# COMMAND ----------

numeric = ['notified_until_assignment', 'age_at_dod']
categorical = ['job_classification', 'situs_state', 'cause', 'job_title', 'diagnosis', 'special_handling', 'work_related', 'special_assignment', 'performance_guarantee', 'complexity']

# COMMAND ----------

numeric_transformer = Pipeline(steps=[
    # impute the median of numeric features
    ('impute', SimpleImputer(strategy='median')),
    # normalize all numeric features
    ('scale', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    # impute missing values with "MISSING"
    ('impute', SimpleImputer(strategy='constant')),
    # encode categorical variables
    ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# COMMAND ----------

preprocessor = ColumnTransformer(
    transformers = [
        ('numeric', numeric_transformer, numeric),
        ('categorical', categorical_transformer, categorical),
    ],
)

# COMMAND ----------

rf_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('random_forest', RandomForestClassifier())
])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set Up the Grid Search
# MAGIC Here we are going to set up the parameters to search over our parameter space.

# COMMAND ----------

param_grid = [
  {'random_forest__n_estimators': [50, 100],
   'random_forest__max_depth': [5, 7, 12],
   'random_forest__min_samples_leaf': [20]},
 ]

# COMMAND ----------

grid_search = GridSearchCV(estimator=rf_pipeline,
                           param_grid=param_grid,
                           scoring=['f1','precision', 'recall', 'accuracy'],
                           refit='f1',
                           cv=5,
                           n_jobs=4,
                           verbose=1,
                           error_score="raise")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Models
# MAGIC One advantage of this method is that we can pass our raw X_train dataframe and transformations are applied in a consistent manner. Additionally, when we call ``.predict()`` on the `best_estimator`, we do not need to apply any transformations prior.

# COMMAND ----------

with mlflow.start_run() as run:
    grid_search.fit(X_train, np.ravel(y_train))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Manual Logging Grid Search to MLflow
# MAGIC We can also manually log our own metrics from grid search cv, but it's a litte more complicated. The mlflow auto logger is a convenience function that takes care of some of this automatically and performs these operations under the hood. However, you cannot select the metrics that are logged from cv_results or alter the scoring functions that come out of the box. We can use a custom function to score if we would like to for more flexibility:

# COMMAND ----------

def log_model(cv_results, params, scores, folds: int, tags={}, model=None):
    """Logging one row of an SKLearn gridsearch cv results to mlflow
    
    Parameters
    ----------
    cv_results : pandas series
        The row of gridsearch.cv_results_ to log to mlflow.
    params : list of strings
        List of parameter names to iterate over for logging to MLFlow.
    scores : list of strings
        List of score names to iterate over for logging to MLFlow.
    folds : int
        The number of cross-validation folds the model
        was trained over.
    model : sklearn.estimator, default=None
        The estimator object to log to mlflow.
    tags : dictionary, default={}
        Additional information to log to mlflow.

    Note
    --------
    Make sure you set your experiment and tracking URI prior to calling 
    this function
        
    """
    
    print(f"Logging to mlflow...")
    with mlflow.start_run(nested=True) as child_run:
        if model:
            print("Logging best model")
            mlflow.sklearn.log_model(model, 'model')
        
        print("Logging CV folds")
        mlflow.log_param("cv_folds", folds)
        
        print("Logging model parameters")
        for param in params:
            mlflow.log_param(param, str(cv_results[f"param_{param}"]))

        print("Logging scoring metrics")
        # Only log the scores on the out of sample
        for score_name in scores:
            mlflow.log_metric(score_name.replace("mean_test_", ""), cv_results[score_name])
        
        print("Logging tags")
        mlflow.set_tags(tags) 

        print(f"Done logging model.")

# COMMAND ----------

def log_cv_results(gridsearch, tags={}):
    """Logging of cross validation results to mlflow tracking server
    Parameters
    ----------
    gridsearch : sklearn.GridSearchCV
        The sklearn grid search object that contains the trained
        results you would like to log to MLFlow.
    
    Note
    --------
    Your experiment will be set to your notebook path
    """
    
    with mlflow.start_run() as run:  
        # sklearn grid search metrics
        cv_results = gridsearch.cv_results_
        num_models = len(gridsearch.cv_results_['params'])
        num_folds = gridsearch.cv
        model_param_names = params = list(grid_search.param_grid[0].keys())
        mean_test_scores = [score for score in cv_results if "mean_test" in score]
        if type(grid_search.refit)==str:
            tags["eval_metric"]= grid_search.refit
        # log each model result to mlflow
        for i in range(num_models):
            # Model attributes
            model_class = re.search("^(.*)__", params[0]).group(1)
            cv_results_row = pd.DataFrame(cv_results).iloc[i]
            # if it is the best model, then save the model object and tag as 'best_estimator'
            if i == gridsearch.best_index_:
                tags["model_object"]= 'best_estimator'
                best_model = gridsearch.best_estimator_
                log_model(cv_results_row, model_param_names, mean_test_scores, run, num_folds, tags, best_model)
            # else, log the cv results
            else:
                log_model(cv_results_row, model_param_names, mean_test_scores, run, num_folds, tags)

# COMMAND ----------

log_cv_results(grid_search)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4:  Score the Test Set and Log the Model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score on the Test Set

# COMMAND ----------

preds = grid_search.best_estimator_.predict(X_test)
score = f1_score(y_test, preds)

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Train Over the Entire Dataset
# MAGIC Combine the training and test set and retrain the model using both. The score on the test set is our best estimate for how this model would perform on unseen data. We will use the parameters selected in cross validation.

# COMMAND ----------

X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test])

# COMMAND ----------

grid_search.best_params_

# COMMAND ----------

model_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('random_forest', RandomForestClassifier(n_estimators=100, min_samples_leaf=20, max_depth=12))
])

# COMMAND ----------

mlflow.autolog(disable=True)

# COMMAND ----------

model = model_pipeline.fit(X_all, np.ravel(y_all))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the Final Model to MLFlow
# MAGIC If we hadn't already called auto logging, we could log one model using a mlflow along with its performance metric.
# MAGIC 
# MAGIC Note: we can specify a conda enironment for model serving but I am not doing that here

# COMMAND ----------

with mlflow.start_run() as run:  
    mlflow.sklearn.log_model(model, 'final_model')

    mlflow.log_param(str(100), 'n_estimators')
    mlflow.log_param(str(20), 'min_samples_leaf')
    mlflow.log_param(str(12), 'max_depth')
        
    mlflow.log_metric('test_score', score)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 5:  Read in the Model and Score
# MAGIC Now that we have saved our model, let's try to read it in and make predictions using it. We can get the model path from the "Experiments" tab.

# COMMAND ----------

logged_model = 'runs:/3f139513bd4f44b19c01989b4df894a8/final_model'

# COMMAND ----------

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

# Predict 
loaded_model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 5:  Advanced

# COMMAND ----------

# TODO: log conda env for model serving
# conda_env =  _mlflow_conda_env(
#         additional_conda_deps=None,
#         additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
#         additional_conda_channels=None,
#     )

# COMMAND ----------

# TODO: log signature
# Log the model with a signature that defines the schema of the model's inputs and outputs. 
  # When the model is deployed, this signature will be used to validate inputs.
#   signature = infer_signature(X_train, wrappedModel.predict(None, X_train))

# COMMAND ----------

# NOTE: can't log CV results to mlflow
# print("Logging CV results matrix")
#         tempdir = tempfile.TemporaryDirectory().name
#         os.mkdir(tempdir)
#         timestamp = datetime.now().isoformat().split(".")[0].replace(":", ".")
#         filename = "%s-%s-cv_results.csv" % (model_name, timestamp)
#         csv = os.path.join(tempdir, filename)
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             pd.DataFrame(cv_results).to_csv(csv, index=False)
        
#         mlflow.log_artifact(csv, "cv_results")

