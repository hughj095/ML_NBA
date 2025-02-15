import time
import pandas as pd
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
from requests.exceptions import ReadTimeout

# Set season
season = "2024-25"  # Adjust for the current season

# Function to get team ID by name
def get_team_id(team_name):
    nba_teams = teams.get_teams()
    for team in nba_teams:
        if team_name.lower() in team['full_name'].lower():
            return team['id']
    return None

# Load teams from CSV
df_teams = pd.read_csv('teams.csv')

# Ensure 'team_name' column exists
if 'team_name' not in df_teams.columns:
    raise ValueError("The CSV must have a 'team_name' column!")

teams_list = df_teams['team_name'].tolist()

# Function to fetch game logs with retry logic
def fetch_game_log(team_id, retries=3, delay=10):
    for attempt in range(retries):
        try:
            game_log = teamgamelog.TeamGameLog(team_id=team_id, season=season)
            return game_log.get_data_frames()[0]
        except ReadTimeout:
            print(f"Timeout error for team ID {team_id}. Retrying ({attempt+1}/{retries})...")
            time.sleep(delay)  # Wait before retrying
    print(f"Failed to fetch data for team ID {team_id} after {retries} retries.")
    return None  # Return None if all retries fail

# Loop through teams and fetch game logs
for team in teams_list:
    team_id = get_team_id(team)
    if not team_id:
        print(f"Team not found: {team}")
        continue  # Skip to the next team

    df = fetch_game_log(team_id)
    
    if df is not None:
        filename = f"{season}/{team}_games_{season}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved: {filename}")
