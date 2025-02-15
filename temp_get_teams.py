import pandas as pd
from nba_api.stats.static import teams

# Get the list of NBA teams
nba_teams = teams.get_teams()

# Create DataFrame
df = pd.DataFrame(nba_teams)[['id', 'full_name', 'abbreviation']]
df.columns = ['Team_ID', 'team_name', 'team_abbr']

# Save to CSV
df.to_csv("teams.csv", index=False)

print("CSV file 'nba_teams.csv' saved successfully!")

