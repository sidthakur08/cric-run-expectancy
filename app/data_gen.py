import pandas as pd
import json
import numpy as np
import math

# loading in the JSON data
print("Loading in Match Results JSON...")
with open('data/match_results.json','r') as f:
    match_res = json.load(f)

match_df = pd.DataFrame(match_res)
match_df = match_df[match_df['gender']=='male']
match_df = match_df.iloc[:,:21] # dropping columns containing supersub info since not relevant
print("Complete Match Results Dataframe Shape for Men's matches ", match_df.shape)

# Filter matches: Men’s ODI with a definitive winner
filtered_matches = match_df[
    (match_df['result'] != 'no result')
]
valid_match_ids = filtered_matches['matchid'].unique().tolist()

# Load innings data
print("Loading in Innings JSON...")
with open('data/innings_results.json', 'r') as f:
    innings_res = json.load(f)

innings_df = pd.DataFrame(innings_res)
# Filter by valid matches
innings_df = innings_df[innings_df['matchid'].isin(valid_match_ids)]
# keeping select columns
select_cols = ['runs.batsman','runs.extras','runs.total','over','team','innings','matchid','wides','wicket.kind','wicket.player_out','wicket.fielders','legbyes','noballs','byes']
innings_df = innings_df.loc[:,select_cols]
print("Dimensions after only keeping relevant matchids and cleaning columns", innings_df.shape)

# Convert over to numeric for sorting and calculations
innings_df['over_numeric'] = innings_df['over'].astype(float)
innings_df = innings_df.sort_values(by=['matchid', 'innings', 'over_numeric']).reset_index(drop=True)

# process data
def process_innings_data(data):
    wickets_fallen = 0
    innings_processed = []
    wicket_tags = ['caught', 'bowled', 'lbw', 'run out', 'caught and bowled', 'stumped', 'hit wicket', 'obstructing the field']
    total_overs = 50
    total_balls = total_overs * 6
    legal_deliveries_bowled = 0
    
    for i, row in data.iterrows():
        runs_on_ball = row.get('runs.total', 0)

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

        innings_processed.append({
            'matchid': row['matchid'],
            'team': row['team'],
            'innings': row['innings'],
            'remaining_overs': remaining_overs,
            'remaining_wickets': wickets_remaining,
            'runs_on_ball': runs_on_ball,
            'wicket_on_ball': wicket_on_ball,
        })

        if wicket_on_ball:
            wickets_fallen += 1

    return pd.DataFrame(innings_processed)

# Apply to each innings
final_balls_df = innings_df.groupby(['matchid', 'team', 'innings'], group_keys=False).apply(process_innings_data).reset_index(drop=True)
final_balls_df.to_csv('data/delivery.csv', index=False)