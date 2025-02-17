import pandas as pd
from nba_api.stats.endpoints import ScoreboardV2
from datetime import datetime, timedelta
from XGBoost import FUNC_PREDICT

# VARIABLES
team_data = pd.read_csv('teams.csv')
season = "2024-25"

# Fetch future game schedule
future_date = (datetime.today() + timedelta(days=3)).strftime('%Y-%m-%d')
scoreboard = ScoreboardV2(game_date=future_date)
games = scoreboard.get_dict()['resultSets'][0]['rowSet']

# Create a DataFrame for future games
future_games_df = pd.DataFrame(
    [row[:8] for row in games],  # Select only the first 8 columns
    columns=['Game_ID', 'Game_Status', 'Game_Code', 'Home_Team_ID', 
             'Away_Team_ID', 'Game_Time', 'Home_Team', 'Away_Team']
)

future_game_team_IDs = future_games_df.iloc[:, [6, 7]]
future_games = future_game_team_IDs.merge(team_data, left_on='Home_Team', right_on='Team_ID')
future_games = future_games.merge(team_data, left_on='Away_Team', right_on='Team_ID')

teams_df = future_games.iloc[:, [6, 3]]
for i in teams_df:
    team = teams_df.iloc[i, 0]
    opponent = teams_df.iloc[i, 1]
    predicted_pts, predicted_opp_pts, predicted_plus_minus = FUNC_PREDICT(team, opponent, season)
    print(f"{team} vs. {opponent}: {predicted_pts} - {predicted_opp_pts} ({predicted_plus_minus})")
