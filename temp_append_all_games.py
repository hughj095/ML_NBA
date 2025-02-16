import pandas as pd
import os

# Directory containing the CSV files
directory = '2024-25'

# List to store DataFrames
df_list = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):  # Ensure it's a CSV file
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)  # Read the file
        df_list.append(df)  # Append to the list

# Concatenate all DataFrames into one
all_teams_df = pd.concat(df_list, ignore_index=True)

# Save the combined DataFrame as a CSV file
all_teams_df.to_csv('all_teams_games_2024-25.csv', index=False)

print("Merged DataFrame saved as 'all_teams_games_2024-25.csv'")
