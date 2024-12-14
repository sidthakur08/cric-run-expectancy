import pandas as pd
import numpy as np
import json
import math
import joblib
import random

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

delivery = pd.read_csv('../data/delivery.csv')

# Get sum of runs per over
delivery['remaining_overs'] = delivery['remaining_overs'].apply(lambda r:int(r))
over_data = delivery.groupby(['matchid', 'team', 'innings', 'remaining_overs'], sort=False)['runs_on_ball'].sum().reset_index(name='runs_this_over')
# since questions is asking for runs per over, not runs per over with wickets in hand -- below code is commented out
# over_runs_df = final_balls_df.groupby(['matchid', 'team', 'innings', 'remaining_overs','remaining_wickets'], sort=False)['runs_on_ball'].sum().reset_index(name='runs_this_over')

# Example: split by matchid
match_ids = over_data['matchid'].unique()
random.seed(123)
np.random.shuffle(match_ids)
train_ids = match_ids[:int(0.8*len(match_ids))]
test_ids = match_ids[int(0.8*len(match_ids)):]

train_df = over_data[over_data['matchid'].isin(train_ids)]
test_df = over_data[over_data['matchid'].isin(test_ids)]

# train_df, test_df = train_test_split(over_data, test_size=0.3, random_state=42)

features = ['team','innings','remaining_overs']

X_train = train_df[features]
y_train = train_df['runs_this_over']

X_test = test_df[features]
y_test = test_df['runs_this_over']

categ_features = ['team']
num_features = ['innings','remaining_overs']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categ_features),
    ],
    remainder='passthrough'  # leaves numeric features as-is
)

# Build a pipeline: first transform (preprocessor), then fit a model
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5

print(f"MAE: {mae}, RMSE: {rmse}")

joblib.dump(model_pipeline, 'linear_model.pkl')
print("Model saved!")