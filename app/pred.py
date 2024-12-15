import argparse
import pandas as pd
import joblib

def get_preds(model_path, csv_path, team="Ireland", start_over = 1, end_over = 5):
    model = joblib.load(model_path)
    team_data = pd.read_csv(csv_path)

    select_cols = ['team','innings','remaining_overs']
    team_data = team_data.loc[:,select_cols]
    team_data = team_data[team_data['team']==team]

    team_data['remaining_overs'] = team_data['remaining_overs'].apply(lambda r:int(r))
    team_data['overs'] = team_data['remaining_overs'].apply(lambda r: 50 - r)
    team_data = team_data[(team_data['overs'] >= start_over) & (team_data['overs'] <= end_over)].reset_index(drop=True).drop_duplicates()

    # Check if the DataFrame is empty after filtering
    if team_data.empty:
        print("No data available for predictions. Check your team name and overs range.")
        return

    team_data['pred_runs_in_over'] = model.predict(team_data.loc[:,['team','innings','remaining_overs']])

    # Print the result
    print(team_data.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--csv", required=True, help="Path to CSV containing data to predict on")
    parser.add_argument("--team", default="Ireland", help="Team name for filtering (default Ireland)")
    parser.add_argument("--start_over", type=int, default=10, help="Starting range of overs to be filtered (default 1, inclusive)")
    parser.add_argument("--end_over", type=int, default=10, help="Ending range of overs to be filtered (default 5, inclusive)")
    args = parser.parse_args()
    get_preds(args.model, args.csv, args.team, args.start_over, args.end_over)