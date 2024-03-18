numeric_transformer = Pipeline(steps=[
    # impute the median of numeric features
    ('impute', SimpleImputer(strategy='median')),
    # normalize all numeric features
    ('scale', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    # impute missing values with "MISSING"
    ('impute', SimpleImputer(strategy='constant')),
    # classify all categories that represent less than 1% of the data as "other"
    ('bin_other', ThresholdBinner(threshold=0.01, bin_label='other')),
    # encode categorical variables
    ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('numeric', numeric_transformer, numeric),
        ('categorical', categorical_transformer, categorical),
        # ('binary', binary_transformer, binary)
    ],
)

rf_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('random_forest', RandomForestClassifier())
])

#%%
param_grid = [
  {'random_forest__n_estimators': [50, 100],
   'random_forest__max_depth': [5, 7, 12],
   'random_forest__min_samples_leaf': [20]},
 ]

grid_search = GridSearchCV(estimator=rf_pipeline,
                           param_grid=param_grid,
                           scoring='f1',
                           cv=5,
                           # n_jobs=-1,
                           verbose=1,
                           error_score="raise")

grid_search.fit(X_train, np.ravel(y_train))

grid_search.best_params_

best_estimator = grid_search.best_estimator_
