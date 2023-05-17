
# coding: utf-8

# # Credit Default Classification

# * SeriousDlqin2yrs
# * RevolvingUtilizationOfUnsecuredLines
# * age
# * NumberOfTime30-59DaysPastDueNotWorse
# * DebtRatio
# * MonthlyIncome
# * NumberOfOpenCreditLinesAndLoans
# * NumberOfTimes90DaysLate
# * NumberRealEstateLoansOrLines
# * NumberOfTime60-89DaysPastDueNotWorse
# * NumberOfDependents

# Source: https://www.kaggle.com/c/GiveMeSomeCredit/data

# # Import required packages

# In[20]:


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
from sklearn.metrics import average_precision_score, log_loss, accuracy_score, precision_score


# # Set the parameters for the run

# In[2]:


model_name = 'credit_default.joblib'
MLFLOW_EXPERIMENT_NAME = 'kenney_sandbox|credit_default'
MLFLOW_TRACKING_URI = 'http://lx8527:5000/'


# # Get the data

# In[3]:


df = pd.read_csv('cs-training.csv')


# In[4]:


df.columns


# In[5]:


df.rename(columns={'Unnamed: 0':'id'}, inplace=True)


# In[6]:


df.head()


# In[7]:


df.dtypes


# In[8]:


len(df)


# # Train Test Split

# In[9]:


y = df['SeriousDlqin2yrs']
X = df.drop('SeriousDlqin2yrs', axis=1)


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# # Exploratory Data Analysis

# In[11]:


y_train.head()


# In[12]:


type(y_train[1])


# In[13]:


y_train.describe()


# In[14]:


len(X_train)


# # Set up the model pipeline

# In[ ]:


gbt_pipeline = Pipeline(steps=[
    ('impute', ColumnTransformer(transformers=[
                        ('scalar imputing mean', SimpleImputer(), X_train.columns),
                        ], remainder='drop')),
    ('scale', ColumnTransformer(transformers=[
                        ('scalar scaling', MinMaxScaler(feature_range=(0, 1)), np.arange(0, len(X_train.columns))),
                        ], remainder='drop')),
    ('GBT', GradientBoostingClassifier())
    ])


# In[ ]:


rf_pipeline = Pipeline(steps=[
    ('impute', ColumnTransformer(transformers=[
                        ('scalar imputing mean', SimpleImputer(), X_train.columns),
                        ], remainder='drop')),
    ('scale', ColumnTransformer(transformers=[
                        ('scalar scaling', MinMaxScaler(feature_range=(0, 1)), np.arange(0, len(X_train.columns))),
                        ], remainder='drop')),
    ('RF', RandomForestClassifier())
    ])


# In[ ]:


lr_pipeline = Pipeline(steps=[
    ('impute', ColumnTransformer(transformers=[
                        ('scalar imputing mean', SimpleImputer(), X_train.columns),
                        ], remainder='drop')),
    ('scale', ColumnTransformer(transformers=[
                        ('scalar scaling', MinMaxScaler(feature_range=(0, 1)), np.arange(0, len(X_train.columns))),
                        ], remainder='drop')),
    ('LR', LogisticRegression())
    ])


# In[ ]:


rf_pipeline.get_params().keys()


# # Set up the grid Search

# Set the parameters for a grid search over the selected family of models.

# In[ ]:


# This trains 360 GBTs with 5 fold CV
gbt_grid = {
    'impute__scalar imputing mean__strategy' : ['mean', 'median'],
    'GBT__n_estimators' : [25, 50, 100],
    'GBT__max_depth'    : [2, 5, 9],
    'GBT__learning_rate': [0.1, 0.5],
    'GBT__loss': ['deviance', 'exponential']
}


# In[ ]:


rf_grid = {
    'impute__scalar imputing mean__strategy' : ['mean', 'median'],
    'RF__n_estimators' : [25, 50, 100],
    'RF__max_depth'    : [2, 5, 9],
    'RF__min_samples_split' : [2, 20]
}


# In[ ]:


lr_grid = {
    'impute__scalar imputing mean__strategy' : ['mean', 'median'],
    'LR__penalty' : ['l1', 'l2'],
    'LR__C'    : np.logspace(0, 4, 10)
}


# ## Train the Models

# In[ ]:


gbt_grid_search = GridSearchCV(gbt_pipeline, gbt_grid, cv=5, return_train_score=False
                   , scoring=['accuracy', 'precision', 'average_precision', 'neg_log_loss']
                   , refit='average_precision', n_jobs=-1 )
gbt_model = gbt_grid_search.fit(X_train, y_train)


# In[ ]:


rf_grid_search = GridSearchCV(rf_pipeline, rf_grid, cv=5, return_train_score=False
                   , scoring=['accuracy', 'precision', 'average_precision', 'neg_log_loss']
                   , refit='average_precision', n_jobs=-1 )
rf_model = rf_grid_search.fit(X_train, y_train)


# In[ ]:


lr_grid_search = GridSearchCV(lr_pipeline, lr_grid, cv=5, return_train_score=False
                   , scoring=['accuracy', 'precision', 'average_precision', 'neg_log_loss']
                   , refit='average_precision', n_jobs=-1 )
lr_model = lr_grid_search.fit(X_train, y_train)


# ## Candidate Model Evaluation

# In[ ]:


# Estimation of performance of GBT on the Validation Set:
average_precision_score(y_test, gbt_model.predict_proba(X_test)[:, 1])


# In[ ]:


# Estimation of performance of RF on the Validation Set:
average_precision_score(y_test, rf_model.predict_proba(X_test)[:, 1])


# In[ ]:


# Estimation of performance of LR on the Validation Set:
average_precision_score(y_test, lr_model.predict_proba(X_test)[:, 1])


# # Time of Run

# In[ ]:


pd.DataFrame(grid_search.cv_results_)['mean_fit_time']


# In[ ]:


time_of_run_in_hours = (pd.DataFrame(grid_search.cv_results_)['mean_fit_time'] * 10).sum() / 60 / 60
print('time of run in hours: {}'.format(time_of_run_in_hours))
hours_per_record = time_of_run_in_hours / 20000
print('hours per record: {}'.format(hours_per_record))
records_in_an_hour = 1 / hours_per_record
print('number of records in 1 hour {}'.format(records_in_an_hour))


# # Use Dimensionality Reduction to Reduce Training Time

# In[15]:


pca_pipeline = Pipeline(steps=[
    ('impute', ColumnTransformer(transformers=[
                        ('scalar imputing mean', SimpleImputer(), X_train.columns),
                        ], remainder='drop')),
    ('scale', ColumnTransformer(transformers=[
                        ('scalar scaling', MinMaxScaler(feature_range=(0, 1)), np.arange(0, len(X_train.columns))),
                        ], remainder='drop')),
    ('PCA', PCA()),
    ('GBT', GradientBoostingClassifier())
    ])


# In[16]:


pca_pipeline.get_params().keys()


# In[17]:


pca_grid = {
    'PCA__n_components' : [2, 3, 5],
    'GBT__n_estimators' : [25, 50, 100],
    'GBT__max_depth'    : [2, 5, 9],
    'GBT__loss': ['deviance', 'exponential']
}


# In[18]:


pca_grid_search = GridSearchCV(pca_pipeline, pca_grid, cv=5, return_train_score=False
                   , scoring=['accuracy', 'precision', 'average_precision', 'neg_log_loss']
                   , refit='average_precision', n_jobs=-1 )
pca_model = pca_grid_search.fit(X_train, y_train)


# In[21]:


# Estimation of performance of PCA on the Validation Set:
average_precision_score(y_test, pca_model.predict_proba(X_test)[:, 1])


# # Analyze the Model Results

# In[22]:


pd.DataFrame(pca_model.cv_results_).T


# In[ ]:


pd.DataFrame(grid_search.cv_results_)[['mean_fit_time', 'param_GBT__max_depth', 'param_GBT__n_estimators',
                                      'mean_test_accuracy', 'mean_test_precision', 'mean_test_average_precision',
                                      'mean_test_neg_log_loss']].T#.to_csv('grid_search_cv_results.csv')


# In[ ]:


#pd.DataFrame(grid_sesarch.cv_results_).columns


# In[ ]:


grid_search.best_estimator_


# # Feature Importance

# In[ ]:


feature_imp = pd.DataFrame({'column': X_train.columns,
                            'feature_importance': grid_search.best_estimator_.named_steps["GBT"].feature_importances_})
feature_imp = feature_imp.sort_values('feature_importance', ascending=False)
#feature_imp.to_csv('feature_importance_90k_lapse.csv')


# In[ ]:


feature_imp.head(15)


# # Save the Model Output

# In[ ]:


dump(grid_search, '{}.joblib'.format(model_name))


# # Write the Predictions

# Evaluate on the Test set

# In[ ]:


from sklearn.metrics import average_precision_score, log_loss, accuracy_score, precision_score
print('Test ({} samples) performance metrics'.format(len(y_test)))
print('average precision: {}'.format(average_precision_score(y_test, 
                                                            grid_search.predict_proba(X_test)[:, 1])))
print('log loss: {}'.format(log_loss(y_test, grid_search.predict_proba(X_test)[:, 1])))
print('accuracy: {}'.format(accuracy_score(y_test, grid_search.predict(X_test))))
print('precision: {}'.format(precision_score(y_test, grid_search.predict(X_test))))


# Save the results

# In[ ]:


test_scores = pd.DataFrame({'id': X_test['id'],
                            'probability': grid_search.predict_proba(X_test)[:, 1],
                            'prediction': grid_search.predict(X_test),
                             'actual': y_test})


# In[ ]:


test_scores.head()


# In[ ]:


#test_scores_with_data = pd.concat([test_scores, X_train.drop('id', axis=1)], axis=1)


# In[ ]:


#test_scores_with_data.head()


# In[ ]:


#test_scores_with_data.to_csv('{}_scores_with_data.csv'.format(model_name))


# # Predcitions Confusion Matrix

# In[ ]:


cm = confusion_matrix(test_scores['actual'], test_scores['prediction'])
cm


# In[ ]:


total = len(test_scores)
perc_vals = [[cm[0][0]/total*100, cm[0][1]/total*100], 
             [cm[1][0]/total*100, cm[1][1]/total*100]]
perc_vals

