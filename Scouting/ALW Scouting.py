# %% 
import pandas as pd
from sklearn import preprocessing as pre
import numpy as np
from functools import reduce

# Reload spreadsheet using new upload
file_path = "C:/Users/blaks/OneDrive/Documents/Work/CCM Analysis/ALW 24_25 All Stats (WyScout).xlsx"
xls = pd.ExcelFile(file_path)

# Load relevant performance sheets
relevant_sheets = ['Attacking', 'Defending', 'Passing', 'Key Passing', 'Goalkeeping']
dfs = [xls.parse(sheet) for sheet in relevant_sheets]

# Merge all performance sheets on 'Player'
from functools import reduce
merged_df = reduce(lambda left, right: pd.merge(left, right, on='Player', how='outer'), dfs)

# Load 'General' for position and minutes played
general_df = xls.parse('General')[['Player', 'Position', 'Minutes played']]
full_df = pd.merge(merged_df, general_df, on='Player', how='left')

# Drop clearly non-performance columns
columns_to_exclude = [
    col for col in full_df.columns
    if any(keyword in col.lower() for keyword in ['team', 'age', 'passport', 'height', 'weight', 'matches played'])
    or col.endswith('_x') or col.endswith('_y')
]
id_cols = ['Player', 'Position', 'Minutes played']
stats_df = full_df.drop(columns=columns_to_exclude, errors='ignore')

# Separate identifier columns
player_info = stats_df[id_cols]
performance_data = stats_df.drop(columns=id_cols, errors='ignore')

# Convert to numeric and fill missing values
performance_data = performance_data.apply(pd.to_numeric, errors='coerce').fillna(0)

# Standardize using Z-scores
scaler = pre.StandardScaler()
scaled_data = scaler.fit_transform(performance_data)

# Return processed info
scaled_data.shape, performance_data.columns[:10], player_info.head()



