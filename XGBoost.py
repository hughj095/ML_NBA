import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

team = "Boston Celtics"

# Load data
df = pd.read_csv(f"{team}_games_2024-25.csv")

# Feature Engineering
df['FG_PCT'] = df['FGM'] / df['FGA']  # Field goal percentage
df['3P_PCT'] = df['FG3M'] / df['FG3A']  # Three-point percentage
df['FT_PCT'] = df['FTM'] / df['FTA']  # Free throw percentage
df['REB_MARGIN'] = df['REB'] - df['OREB']  # Rebound margin
df['ASSIST_RATIO'] = df['AST'] / df['FGM']  # Assist to FG ratio

# Rolling averages (last 5 games)
df['LAST_5_PTS'] = df['PTS'].rolling(window=5).mean().shift(1)
df['LAST_5_OPP_PTS'] = df['PLUS_MINUS'].rolling(window=5).mean().shift(1)

# Add weighted history vs. the same opponent
df['VS_OPP_PTS'] = df.groupby('MATCHUP')['PTS'].transform(lambda x: x.rolling(window=3).mean().shift(1))

# Drop rows with NaN (first few games may lack rolling averages)
df = df.dropna()

# Define features and target
features = ['FG_PCT', '3P_PCT', 'FT_PCT', 'REB_MARGIN', 'ASSIST_RATIO', 'LAST_5_PTS', 'LAST_5_OPP_PTS', 'VS_OPP_PTS']
target = ['PTS', 'PLUS_MINUS']  # Predicting both team's score

# Split data
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Predict next game
next_game = X_test.iloc[-1:]  # Simulate next game
predicted_score = model.predict(next_game)

print(f"Predicted Score: {predicted_score}")
