import pandas as pd
import json
import numpy as np
import math
import os

def generate_data():

    '''
    Generates a delivery-level dataset by processing match results and innings data.
    
    Steps:
    - Load match_results.json and innings_results.json.
    - Filter relevant matches (e.g., men's ODI matches with definitive results).
    - Process deliveries to compute remaining overs, wickets, and runs per ball.
    - Save the cleaned data as delivery.csv.
    '''

    # Load match results
    print("Loading in Match Results JSON...")
    with open('data/match_results.json', 'r') as f:
        match_res = json.load(f)

    # Convert match results to DataFrame
    match_df = pd.DataFrame(match_res)
    match_df = match_df[match_df['gender'] == 'male']  # Filter for men's matches
    match_df = match_df.iloc[:, :21]  # Dropping columns containing supersub info since not relevant
    print("Complete Match Results DataFrame Shape for Men's matches ", match_df.shape)

    # Filter matches with definitive results
    filtered_matches = match_df[(match_df['result'] != 'no result')]
    valid_match_ids = filtered_matches['matchid'].unique().tolist()

    # Load innings data
    print("Loading in Innings JSON...")
    with open('data/innings_results.json', 'r') as f:
        innings_res = json.load(f)

    # Convert innings results to DataFrame and filter by valid matches
    innings_df = pd.DataFrame(innings_res)
    innings_df = innings_df[innings_df['matchid'].isin(valid_match_ids)]
    print("Dimensions after filtering valid match IDs:", innings_df.shape)

    # Keep only selected columns
    select_cols = ['runs.batsman', 'runs.extras', 'runs.total', 'over', 'team', 'innings',
                   'matchid', 'wides', 'wicket.kind', 'wicket.player_out', 'wicket.fielders',
                   'legbyes', 'noballs', 'byes']
    innings_df = innings_df.loc[:, select_cols]

    # Convert 'over' to numeric for sorting and calculations
    innings_df['over_numeric'] = innings_df['over'].astype(float)
    innings_df = innings_df.sort_values(by=['matchid', 'innings', 'over_numeric']).reset_index(drop=True)

    def process_innings_data(data):
        '''
        Processes each innings to compute deliveries with remaining overs, wickets, and runs per ball.
        
        Args:
        data (DataFrame): Input DataFrame containing delivery-level data.

        Returns:
        DataFrame: Processed data with calculated features.
        '''

        wickets_fallen = 0
        innings_processed = []
        wicket_tags = ['caught', 'bowled', 'lbw', 'run out', 'caught and bowled', 'stumped', 'hit wicket', 'obstructing the field']
        total_overs = 50
        total_balls = total_overs * 6
        legal_deliveries_bowled = 0
        
        for _, row in data.iterrows():
            runs_on_ball = row.get('runs.total', 0)
            wicket_on_ball = 1 if row['wicket.kind'] in wicket_tags else 0

            # Determine if the ball is legal (exclude wides and no-balls)
            legal = math.isnan(row['wides']) and math.isnan(row['noballs'])
            if legal:
                legal_deliveries_bowled += 1

            # Calculate remaining overs in standard format (O.B)
            balls_remaining = total_balls - legal_deliveries_bowled
            overs_remaining_int = balls_remaining // 6
            balls_remaining_in_over = balls_remaining % 6
            remaining_overs = overs_remaining_int + (balls_remaining_in_over * 0.1)

            # Update wickets fallen
            if wicket_on_ball:
                wickets_fallen += 1
            
            wickets_remaining = 10 - wickets_fallen

            # Append processed row
            innings_processed.append({
                'matchid': row['matchid'],
                'team': row['team'],
                'innings': row['innings'],
                'remaining_overs': remaining_overs,
                'remaining_wickets': wickets_remaining,
                'runs_on_ball': runs_on_ball,
                'wicket_on_ball': wicket_on_ball,
            })

        return pd.DataFrame(innings_processed)

    # Apply processing to each innings group
    print("Processing innings data...")
    final_balls_df = innings_df.groupby(['matchid', 'team', 'innings'])[list(innings_df.columns)].apply(process_innings_data).reset_index(drop=True)

    # Save the processed data to CSV
    final_balls_df.to_csv('data/delivery.csv', index=False)
    print("Delivery data saved to data/delivery.csv")

if __name__ == "__main__":
    generate_data()