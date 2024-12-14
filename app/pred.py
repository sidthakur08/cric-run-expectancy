import argparse
import pandas as pd
import joblib

def get_preds(model_path, csv_path, team="Ireland", start_over = 1, end_over = 5):
    model = joblib.load(model_path)
    team_data = pd.read_csv(csv_path)
    team_data = team_data[(team_data['team']==team)]

    # Get sum of runs per over
    team_data['remaining_overs'] = team_data['remaining_overs'].apply(lambda r:int(r))
    over_data = team_data.groupby(['matchid', 'team', 'innings', 'remaining_overs'], sort=False)['runs_on_ball'].sum().reset_index(name='runs_this_over')
    over_data['overs'] = 50 - team_data['remaining_overs']
    over_data = over_data[(over_data['overs'] >= start_over) & (over_data['overs'] <= end_over)]

    return over_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--csv", required=True, help="Path to CSV containing data to predict on")
    parser.add_argument("--team", default="Ireland", help="Team name for filtering (default Ireland)")
    parser.add_argument("--start_over", type=int, default=10, help="Starting range of overs to be filtered (default 1, inclusive)")
    parser.add_argument("--end_over", type=int, default=10, help="Ending range of overs to be filtered (default 5, inclusive)")
    args = parser.parse_args()
    get_preds(args.model, args.csv, args.team, args.start_over, args.end_over)