import pandas as pd
import json
from nba_api.live.nba.endpoints import scoreboard

# Get today's scoreboard data
games = scoreboard.ScoreBoard()

# Parse JSON
data = json.loads(games.get_json())

# Extract games
all_games = data["scoreboard"]["games"]

# Filter for future games (gameStatus == 1)
future_games = [
    {
        "gameId": game["gameId"],
        "gameTimeUTC": game["gameTimeUTC"],
        "homeTeam": f"{game['homeTeam']['teamCity']} {game['homeTeam']['teamName']}",
        "awayTeam": f"{game['awayTeam']['teamCity']} {game['awayTeam']['teamName']}"
    }
    for game in all_games if game["gameStatus"] == 1
]

# Convert to pandas DataFrame
df = pd.DataFrame(future_games)

# Print the DataFrame
print(df)



