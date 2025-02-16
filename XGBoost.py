import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

team = "Boston Celtics"
opponent = "Milwaukee Bucks"  # Specify upcoming opponent
season = "2024-25"

# Load data
team_df = pd.read_csv(f"{season}/{team}_games_2024-25.csv")
opponent_df = pd.read_csv(f"{season}/{opponent}_games_2024-25.csv")

# Rename opponent stats
opponent_df = opponent_df.rename(columns={
    "PTS": "OPP_PTS",
    "FGM": "OPP_FGM",
    "FGA": "OPP_FGA",
    "FG3M": "OPP_FG3M",
    "FG3A": "OPP_FG3A",
    "FTM": "OPP_FTM",
    "FTA": "OPP_FTA",
    "REB": "OPP_REB",
    "OREB": "OPP_OREB",
    "AST": "OPP_AST",
    "PLUS_MINUS": "OPP_PLUS_MINUS"
})

# Merge team and opponent data on GAME_ID
df = team_df.merge(opponent_df, on="GAME_ID", suffixes=("", "_OPP"))

# Calculate team & opponent shooting percentages
df["FG_PCT"] = df["FGM"] / df["FGA"]
df["3P_PCT"] = df["FG3M"] / df["FG3A"]
df["FT_PCT"] = df["FTM"] / df["FTA"]
df["REB_MARGIN"] = df["REB"] - df["OREB"]
df["ASSIST_RATIO"] = df["AST"] / df["FGM"]

df["OPP_FG_PCT"] = df["OPP_FGM"] / df["OPP_FGA"]
df["OPP_3P_PCT"] = df["OPP_FG3M"] / df["OPP_FG3A"]
df["OPP_FT_PCT"] = df["OPP_FTM"] / df["OPP_FTA"]
df["OPP_REB_MARGIN"] = df["OPP_REB"] - df["OPP_OREB"]
df["OPP_ASSIST_RATIO"] = df["OPP_AST"] / df["OPP_FGM"]

# Rolling averages for last 5 games (team and opponent)
df["LAST_5_PTS"] = df["PTS"].rolling(window=5).mean().shift(1)
df["LAST_5_OPP_PTS"] = df["OPP_PTS"].rolling(window=5).mean().shift(1)
df["LAST_5_PLUS_MINUS"] = df["PLUS_MINUS"].rolling(window=5).mean().shift(1)

# Historical performance vs. opponent
df["VS_OPP_PTS"] = df.groupby("MATCHUP")["PTS"].transform(lambda x: x.rolling(window=3).mean().shift(1))
df["VS_OPP_PLUS_MINUS"] = df.groupby("MATCHUP")["PLUS_MINUS"].transform(lambda x: x.rolling(window=3).mean().shift(1))

# Drop NaN rows caused by rolling averages
df = df.dropna()

# Define features and target
features = [
    "FG_PCT", "3P_PCT", "FT_PCT", "REB_MARGIN", "ASSIST_RATIO",
    "OPP_FG_PCT", "OPP_3P_PCT", "OPP_FT_PCT", "OPP_REB_MARGIN", "OPP_ASSIST_RATIO",
    "LAST_5_PTS", "LAST_5_OPP_PTS", "LAST_5_PLUS_MINUS", "VS_OPP_PTS", "VS_OPP_PLUS_MINUS"
]
target = ["PTS", "OPP_PTS", "PLUS_MINUS"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# **Predict the next game against the known opponent**
latest_team_stats = team_df.iloc[-1][["FG_PCT", "3P_PCT", "FT_PCT", "REB_MARGIN", "ASSIST_RATIO"]]
latest_opp_stats = opponent_df.iloc[-1][["OPP_FG_PCT", "OPP_3P_PCT", "OPP_FT_PCT", "OPP_REB_MARGIN", "OPP_ASSIST_RATIO"]]

# Get last 5 game averages
latest_team_rolling = df.iloc[-1][["LAST_5_PTS", "LAST_5_OPP_PTS", "LAST_5_PLUS_MINUS"]]
latest_vs_opp = df[df["MATCHUP"] == f"{team} vs {opponent}"].iloc[-1][["VS_OPP_PTS", "VS_OPP_PLUS_MINUS"]]

# Prepare input for prediction
next_game = pd.DataFrame([{
    **latest_team_stats, **latest_opp_stats, **latest_team_rolling, **latest_vs_opp
}])

# Predict
predicted_score = model.predict(next_game)

# Round & display
predicted_pts = round(predicted_score[0][0])
predicted_opp_pts = round(predicted_score[0][1])
predicted_plus_minus = round(predicted_score[0][2])

print(f"Predicted Score: {team} {predicted_pts} - {opponent} {predicted_opp_pts}")
print(f"Predicted PLUS/MINUS: {predicted_plus_minus}")

