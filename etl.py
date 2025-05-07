import pandas as pd
import numpy as np
import os

df = pd.read_csv('data/adp_merged_7_17.csv')
df = df.dropna(subset=['name', 'position', 'team', 'adp'])
df = df[['Year', 'name', 'position', 'team', 'adp']]
df = df[df['position'].isin(['RB', 'WR', 'QB', 'TE'])]
os.makedirs('clean_data', exist_ok=True)
df.to_csv('clean_data/cleaned_players.csv', index=False)
print("✅ Cleaned data saved to clean_data/cleaned_players.csv")
