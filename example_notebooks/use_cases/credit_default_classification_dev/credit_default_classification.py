from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from pyspark.sql import SparkSession
from sklearn.metrics import average_precision_score, log_loss, accuracy_score, precision_score

# Method to create the spark session.
def createSparkSession(application_name):
    spark = SparkSession                                      \
        .builder                                              \
        .appName(application_name)                            \
        .config("spark.sql.catalogImplementation", "hive")    \
        .config('spark.yarn.queue', 'sparkjob')               \
        .config('spark.driver.cores', 10)                     \
        .config('spark.yarn.containerLauncherMaxThreads', 10) \
        .config('spark.executor.memory', '8g')                \
        .config('spark.rpc.numRetries', '10')                 \
        .config('spark.network.timeout', '600s')              \
        .config('spark.executor.instances', 50)               \
        .config('spark.executor.cores', 5)                    \
        .config('spark.driver.memory', '15g')                 \
        .enableHiveSupport()                                  \
        .getOrCreate()
    return spark


# # Set the parameters for the run
model_name = 'credit_default.joblib'
MLFLOW_EXPERIMENT_NAME = 'kenney_sandbox|credit_default'
MLFLOW_TRACKING_URI = 'http://lx8527:5000/'


# # Get the data
df = pd.read_csv('cs-training.csv')
df.rename(columns={'Unnamed: 0':'id'}, inplace=True)

# # Train Test Split
y = df['SeriousDlqin2yrs']
X = df.drop('SeriousDlqin2yrs', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# # Set up the model pipeline
gbt_pipeline = Pipeline(steps=[
    ('impute', ColumnTransformer(transformers=[
                        ('scalar imputing mean', SimpleImputer(), X_train.columns),
                        ], remainder='drop')),
    ('scale', ColumnTransformer(transformers=[
                        ('scalar scaling', MinMaxScaler(feature_range=(0, 1)), np.arange(0, len(X_train.columns))),
                        ], remainder='drop')),
    ('GBT', GradientBoostingClassifier())
    ])

rf_pipeline = Pipeline(steps=[
    ('impute', ColumnTransformer(transformers=[
                        ('scalar imputing mean', SimpleImputer(), X_train.columns),
                        ], remainder='drop')),
    ('scale', ColumnTransformer(transformers=[
                        ('scalar scaling', MinMaxScaler(feature_range=(0, 1)), np.arange(0, len(X_train.columns))),
                        ], remainder='drop')),
    ('RF', RandomForestClassifier())
    ])

lr_pipeline = Pipeline(steps=[
    ('impute', ColumnTransformer(transformers=[
                        ('scalar imputing mean', SimpleImputer(), X_train.columns),
                        ], remainder='drop')),
    ('scale', ColumnTransformer(transformers=[
                        ('scalar scaling', MinMaxScaler(feature_range=(0, 1)), np.arange(0, len(X_train.columns))),
                        ], remainder='drop')),
    ('LR', LogisticRegression())
    ])


# # Set up the grid Search
# Set the parameters for a grid search over the selected family of models.

gbt_grid = {
    'impute__scalar imputing mean__strategy' : ['mean', 'median'],
    'GBT__n_estimators' : [25, 50, 100, 250],
    'GBT__max_depth'    : [2, 5, 9, 15],
    'GBT__learning_rate': [0.1, 0.5],
    'GBT__loss': ['deviance', 'exponential']
}

rf_grid = {
    'impute__scalar imputing mean__strategy' : ['mean', 'median'],
    'RF__n_estimators' : [25, 50, 100, 250],
    'RF__max_depth'    : [2, 5, 9, 15],
    'RF__min_samples_split' : [2, 20]
}

lr_grid = {
    'impute__scalar imputing mean__strategy' : ['mean', 'median'],
    'LR__penalty' : ['l1', 'l2'],
    'LR__C'    : np.logspace(0, 4, 10)
}


# ## Train the Models
gbt_grid_search = GridSearchCV(gbt_pipeline, gbt_grid, cv=5, return_train_score=False
                   , scoring=['accuracy', 'precision', 'average_precision', 'neg_log_loss']
                   , refit='average_precision', n_jobs=-1 )

rf_grid_search = GridSearchCV(rf_pipeline, rf_grid, cv=5, return_train_score=False
                   , scoring=['accuracy', 'precision', 'average_precision', 'neg_log_loss']
                   , refit='average_precision', n_jobs=-1 )

lr_grid_search = GridSearchCV(lr_pipeline, lr_grid, cv=5, return_train_score=False
                   , scoring=['accuracy', 'precision', 'average_precision', 'neg_log_loss']
                   , refit='average_precision', n_jobs=-1 )

gbt_model = gbt_grid_search.fit(X_train, y_train)
rf_model = rf_grid_search.fit(X_train, y_train)
lr_model = lr_grid_search.fit(X_train, y_train)

# Candidate Model Evaluation: Log the results to MLFlow
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment(experiment_name)
with mlflow.start_run(nested = True) as r:
    mlflow.log_param('model_type', 'gbt')
    for i in range(0, len(gbt_model.cv_results_['params'])):
        with mlflow.start_run(nested = True) as inner_run:
            for param in gbt_grid:
                mlflow.log_param(param, gbt_model.cv_results_['params'][i][param])
            mlflow.log_metric('total_fit_minutes', (gbt_model.cv_results_['mean_fit_time'][i]*5)/60)
            mlflow.log_metric('mean_test_accuracy', gbt_model.cv_results_['mean_test_accuracy'][i])
            mlflow.log_metric('mean_test_average_precision', gbt_model.cv_results_['mean_test_average_precision'][i])
            mlflow.log_metric('mean_test_neg_log_loss', gbt_model.cv_results_['mean_test_neg_log_loss'][i])
            mlflow.log_metric('mean_test_precision', gbt_model.cv_results_['mean_test_precision'][i])
            
with mlflow.start_run(nested = True) as r:
    mlflow.log_param('model_type', 'lr')
    for i in range(0, len(lr_model.cv_results_['params'])):
        with mlflow.start_run(nested = True) as inner_run:
            for param in lr_grid:
                mlflow.log_param(param, lr_model.cv_results_['params'][i][param])
            mlflow.log_metric('total_fit_minutes', (lr_model.cv_results_['mean_fit_time'][i]*5)/60)
            mlflow.log_metric('mean_test_accuracy', lr_model.cv_results_['mean_test_accuracy'][i])
            mlflow.log_metric('mean_test_average_precision', lr_model.cv_results_['mean_test_average_precision'][i])
            mlflow.log_metric('mean_test_neg_log_loss', lr_model.cv_results_['mean_test_neg_log_loss'][i])
            mlflow.log_metric('mean_test_precision', lr_model.cv_results_['mean_test_precision'][i])

with mlflow.start_run(nested = True) as r:
    mlflow.log_param('model_type', 'rf')
    for i in range(0, len(rf_model.cv_results_['params'])):
        with mlflow.start_run(nested = True) as inner_run:
            for param in rf_grid:
                mlflow.log_param(param, rf_model.cv_results_['params'][i][param])
            mlflow.log_metric('total_fit_minutes', (rf_model.cv_results_['mean_fit_time'][i]*5)/60)
            mlflow.log_metric('mean_test_accuracy', rf_model.cv_results_['mean_test_accuracy'][i])
            mlflow.log_metric('mean_test_average_precision', rf_model.cv_results_['mean_test_average_precision'][i])
            mlflow.log_metric('mean_test_neg_log_loss', rf_model.cv_results_['mean_test_neg_log_loss'][i])
            mlflow.log_metric('mean_test_precision', rf_model.cv_results_['mean_test_precision'][i])
