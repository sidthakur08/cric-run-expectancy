import pandas as pd
import numpy as np
import json
import math
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# loading in the JSON data
with open('match_results.json','r') as f:
    match_res = json.load(f)

match_df = pd.DataFrame(match_res)
match_df = match_df[match_df['gender']=='male']
match_df = match_df.iloc[:,:21] # dropping columns containing supersub info since not relevant

# Filter matches: Men’s ODI with a definitive winner
filtered_matches = match_df[
    (match_df['result'] != 'no result')
]

valid_match_ids = filtered_matches['matchid'].unique().tolist()
# Load innings data
with open('innings_results.json', 'r') as f:
    innings_res = json.load(f)

innings_df = pd.DataFrame(innings_res)
# Filter by valid matches
innings_df = innings_df[innings_df['matchid'].isin(valid_match_ids)]
# keeping select columns
select_cols = ['runs.batsman','runs.extras','runs.total','over','team','innings','matchid','wides','wicket.kind','wicket.player_out','wicket.fielders','legbyes','noballs','byes']
innings_df = innings_df.loc[:,select_cols]

# Convert over to numeric for sorting and calculations
innings_df['over_numeric'] = innings_df['over'].astype(float)
innings_df = innings_df.sort_values(by=['matchid', 'innings', 'over_numeric']).reset_index(drop=True)

def process_innings_data(df):
    wickets_fallen = 0
    deliveries_processed = []
    wicket_tags = ['caught', 'bowled', 'lbw', 'run out', 'caught and bowled', 'stumped', 'hit wicket', 'obstructing the field']
    total_overs = 50
    total_balls = total_overs * 6
    legal_deliveries_bowled = 0
    
    for i, row in df.iterrows():
        runs_batsman = row.get('runs.batsman', 0)
        runs_extras = row.get('runs.extras', 0)
        runs_on_ball = runs_batsman + runs_extras

        wicket_on_ball = 1 if (row['wicket.kind'] in wicket_tags) else 0
        wickets_remaining = 10 - wickets_fallen

        # Determine if this ball is legal
        # If it's a wide or no-ball, do not increment legal_deliveries_bowled
        legal = True if (math.isnan(row['wides']) & math.isnan(row['noballs'])) else False
        if legal:
            legal_deliveries_bowled += 1

        # Calculate remaining overs in standard cricket format (O.B)
        balls_remaining = total_balls - legal_deliveries_bowled
        overs_remaining_int = balls_remaining // 6
        balls_remaining_in_over = balls_remaining % 6
        remaining_overs = overs_remaining_int + (balls_remaining_in_over * 0.1)

        deliveries_processed.append({
            'matchid': row['matchid'],
            'team': row['team'],
            'innings': row['innings'],
            'runs_on_ball': runs_on_ball,
            'wicket_on_ball': wicket_on_ball,
            'remaining_overs': remaining_overs,
            'remaining_wickets': wickets_remaining,
            'over_numeric': row['over_numeric']
        })

        if wicket_on_ball:
            wickets_fallen += 1

    return pd.DataFrame(deliveries_processed)

# Apply to each innings
final_balls_df = innings_df.groupby(['matchid', 'innings'], group_keys=False).apply(process_innings_data).reset_index(drop=True)
# Assuming final_balls_df is the dataset you currently have
# Extract the integer over number from over_numeric
final_balls_df['over_number'] = final_balls_df['over_numeric'].apply(lambda x: int(str(x).split('.')[0]))

# Get sum of runs per over
over_runs_df = final_balls_df.groupby(['matchid', 'innings', 'over_number'])['runs_on_ball'].sum().reset_index(name='runs_this_over')

# Get the state at the start of the over from the first legal delivery of that over
over_state_df = final_balls_df.groupby(['matchid', 'innings', 'over_number']).first().reset_index()
over_state_df['remaining_overs'] = 49 - over_state_df['over_number']
over_data = over_state_df[['matchid', 'innings', 'over_number','remaining_overs', 'team', 'remaining_wickets']].merge(
    over_runs_df, on=['matchid', 'innings', 'over_number'], how='inner'
)

# Now over_data contains one row per over with:
# matchid, innings, over_number, team, remaining_overs, remaining_wickets, runs_this_over
# Example: split by matchid
match_ids = over_data['matchid'].unique()
np.random.shuffle(match_ids)
train_ids = match_ids[:int(0.8*len(match_ids))]
test_ids = match_ids[int(0.8*len(match_ids)):]

train_df = over_data[over_data['matchid'].isin(train_ids)]
test_df = over_data[over_data['matchid'].isin(test_ids)]
features = ['remaining_overs', 'remaining_wickets', 'innings']

X_train = train_df[features]
y_train = train_df['runs_this_over']

X_test = test_df[features]
y_test = test_df['runs_this_over']

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5

print(f"MAE: {mae}, RMSE: {rmse}")

joblib.dump(rf_model, 'rf_model.pkl')
print("Model saved!")