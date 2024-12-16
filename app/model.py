import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_model(input_path = 'data/delivery.csv', output_path = 'models/model.pkl'):
    '''
    Trains a Linear Regression model using delivery data and saves the model.
    
    Args:
    input_path (str): Path to the input CSV file.
    output_path (str): Path to save the trained model.
    '''

    print("Loading input data...")
    delivery = pd.read_csv(input_path)

    # Check required columns
    req_cols = {'matchid', 'team', 'innings', 'remaining_overs', 'runs_on_ball'}
    if not req_cols.issubset(delivery.columns):
        raise KeyError(f"Missing required columns: {req_cols - set(delivery.columns)}")

    if delivery.empty:
        raise ValueError("The dataset is empty")

    # Preprocess data
    print("Preprocessing data...")
    delivery['remaining_overs'] = delivery['remaining_overs'].apply(lambda r: int(r))
    over_data = delivery.groupby(['matchid', 'team', 'innings', 'remaining_overs'], sort=False)['runs_on_ball'].sum().reset_index(name='runs_this_over')

    # Split data into training and testing
    print("Splitting data into train and test sets...")
    match_ids = over_data['matchid'].unique()
    np.random.seed(123)
    np.random.shuffle(match_ids)
    train_ids = match_ids[:int(0.8 * len(match_ids))]
    test_ids = match_ids[int(0.8 * len(match_ids)):]

    train_df = over_data[over_data['matchid'].isin(train_ids)]
    test_df = over_data[over_data['matchid'].isin(test_ids)]

    features = ['team', 'innings', 'remaining_overs']
    X_train = train_df[features]
    y_train = train_df['runs_this_over']
    X_test = test_df[features]
    y_test = test_df['runs_this_over']

    # Define preprocessor for categorical features
    categ_features = ['team']
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categ_features)
    ], remainder='passthrough')

    # Build pipeline with preprocessor and Linear Regression model
    print("Training Linear Regression model...")
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    model_pipeline.fit(X_train, y_train)

    # Evaluate model
    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MAE: {mae}, RMSE: {mse**0.5}")

    # Save model
    joblib.dump(model_pipeline, output_path)
    print(f"Model saved to {output_path}")

if __name__ == '__main__':
    train_model()