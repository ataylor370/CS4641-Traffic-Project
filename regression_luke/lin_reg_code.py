import pandas as pd
from pathlib import Path
import csv
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn import set_config
from sklearn.metrics import mean_squared_error, r2_score

#set_config(transform_output="pandas")

df = pd.read_csv("data_points.csv")
X = df[["road_type", "days_of_week", "time", "through_lanes"]]
y = df["hourly_throughput"]
groups = df["point"]

splitter = GroupShuffleSplit(n_splits=1, test_size = 0.2, random_state = 0)


train_idx, test_idx = next(splitter.split(X, y, groups=groups))


X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


df_train = df.iloc[train_idx]
df_test  = df.iloc[test_idx]

'''
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(train_idx)
print(test_idx)
'''

column_prep = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(drop='first'), ['road_type', 'days_of_week']),
    ('poly',   PolynomialFeatures(degree=4, include_bias=False), ['time']),
    ('scale',  StandardScaler(), ['through_lanes']),
], remainder='passthrough')

add_cross_terms = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

pipeline = Pipeline(steps=[
  ('prep', column_prep),
  #('cross', add_cross_terms),
  ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)

model = pipeline.named_steps['model']

coefficients = model.coef_
intercept    = model.intercept_

# save to csv

# save metrics
metrics = {
    'r2_train': r2_score(y_train, pipeline.predict(X_train)),
    'r2_test':  r2_score(y_test,  pipeline.predict(X_test)),
    'rmse_train': np.sqrt(mean_squared_error(y_train, pipeline.predict(X_train))),
    'rmse_test':  np.sqrt(mean_squared_error(y_test,  pipeline.predict(X_test))),
    'intercept':  intercept
}
coef_df = pd.DataFrame({'coefficient': coefficients})
coef_df.to_csv('L1_coefficients.csv', index=False)
pd.DataFrame([metrics]).to_csv('L1_metrics.csv', index=False)
print(metrics)