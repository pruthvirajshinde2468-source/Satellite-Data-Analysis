# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:28:35 2025

@author: pruth
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, timedelta

plt.style.use('default')
plt.figure(figsize=(12,6))



df = pd.read_csv(r"C:\Users\pruth\Downloads\open-meteo-52.52N13.42E38m.csv")
print("Data shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))

## data cleaning and preparation

# check data type and missing values
print("Data type and missing values:")
print(df.info())
print("Missind value summery:")
print(df.isnull().sum())

#Create a clean dataframe starting from the time series data
# Find where the time column starts (after the metadata row)
time_start_index = df[df.iloc[:,0] == 'time'].index[0]
#df.iloc[:, 0]
#What it does: Gets all rows (:) from the first column (0) of the DataFrame.
#Translation: "Give me the entire first column of the DataFrame."

#. df.iloc[:, 0] == 'time'
#What it does: Creates a Boolean mask - compares each value in the first column to the string 'time'.
#Result: Returns True where the value equals 'time', False everywhere else.

print(f"\nTime series data starts at row:  {time_start_index}")

# create new data frame
headers = df.iloc[time_start_index]

df_clean = df.iloc[time_start_index + 1:].copy()
#df.iloc[time_start_index + 1:] gets all rows starting from the NEXT row after the header
#time_start_index + 1 means: "Start from the row after the header row"
#: means: "Go all the way to the end"
#.copy() creates a new independent DataFrame so changes don't affect the original

df_clean.columns = headers
df_clean = df_clean.reset_index(drop=True)
print("n\Cleaned DataFrame shape:", df_clean.shape)
print(df_clean.head())

## Convert data types and handle Timestamp

df_clean['time'] = pd.to_datetime(df_clean['time'])

# Convert temp and radiation to numeric, handling missing values
df_clean['temperature_2m (°C)'] = pd.to_numeric(df_clean['temperature_2m (°C)'], errors='coerce')
df_clean['direct_radiation (W/m²)'] = pd.to_numeric(df_clean['direct_radiation (W/m²)'], errors='coerce')

df_clean = df_clean.set_index("time")
df_clean = df_clean.sort_index()
print("Data after cleaning and conversion:")
print(df_clean.info())


### Exploratory Data analysis

print("Basic Statistics")
print(df_clean.describe())
# gives stastictical  values like count mean max min

# check time period coverd
time_span = df_clean.index.max() - df_clean.index.min()
print(f'\nData covers {time_span.days} days')

# Check data completeness
print('\nData Completeness:')
print(f"Temperature: {df_clean['temperature_2m (°C)'].notna().sum()}/{len(df_clean)} ({df_clean['temperature_2m (°C)'].notna().mean()*100:.1f}%)")

# df_clean['temperature_2m (°C)'].notna() Creates a True/False mask for each row True = Temperature value is present (not NaN/missing) False = Temperature value is missing (NaN)
# .sum() Counts how many True values are in the mask
# .notna().mean()*100 Calculates the percentage of non-missing data
# {...:.1f}% Formats the percentage to 1 decimal place

print(f"Radiation: {df_clean['direct_radiation (W/m²)'].notna().sum()}/{len(df_clean)} ({df_clean['direct_radiation (W/m²)'].notna().mean()*100:.1f}%)")

### Data Visualization 

fig, axes = plt.subplots(2,2, figsize=(15, 10))
fig.suptitle('Berlin wheather Data Analysis (52.52°N, 13.42°E)', fontsize=16, fontweight='bold')

# Plot 1: Temperature over time

axes[0,0].plot(df_clean.index, df_clean['temperature_2m (°C)'], color='red', alpha=0.7, linewidth=1)
#df_clean.index = X-axis values (the timestamps)
#df_clean['temperature_2m (°C)'] = Y-axis values (temperature data)
#alpha=0.7 = 70% opaque, 30% transparent (lets you see through the line a bit)

axes[0,0].set_title('Temperature Trend')
axes[0,0].set_ylabel('Temperature')
axes[0,0].grid(True, alpha=0.3)
#alpha=0.3 makes the grid lines very faint so they don't distract from the data.
axes[0,0].tick_params(axis='x', rotation=45)
#Rotates the X-axis labels (dates) by 45 degrees.

## Radiation over time

axes[0,1].plot(df_clean.index, df_clean['direct_radiation (W/m²)'], color='green', alpha=0.7, linewidth=1)
axes[0,1].set_title('Radiation Chnages')
axes[0,1].set_ylabel("Radiation")
axes[0,1].grid(True, alpha=0.3)
axes[0,1].tick_params(axis='x', rotation=45)

# Plot 3: Temperature distribution
axes[1, 0].hist(df_clean['temperature_2m (°C)'].dropna(), bins=50, color='red', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Temperature Distribution')
axes[1, 0].set_xlabel('Temperature (°C)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Radiation distribution
axes[1, 1].hist(df_clean['direct_radiation (W/m²)'].dropna(), bins=50, color='orange', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Radiation Distribution')
axes[1, 1].set_xlabel('Radiation (W/m²)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

#### Advance analysis- Correlation and Trends 


#Correlation measures how two variables move together:
#positive correlation: When one goes up, the other tends to go up
#Negative correlation: When one goes up, the other tends to go down
#No correlation: No relationship between them
#The correlation coefficient ranges from -1 to +1:

#+1.0 = Perfect positive correlation
#+0.7 = Strong positive correlation
#+0.3 = Weak positive correlation
#0.0 = No correlation
#-0.3 = Weak negative correlation
#-0.7 = Strong negative correlation
#-1.0 = Perfect negative correlation

#print(f"Correlation: {correlation:.3f}") This displays the result formatted to 3 decimal places.

# scatter plot to show relationship 

# Correlation between temperature and radiation
correlation = df_clean['temperature_2m (°C)'].corr(df_clean['direct_radiation (W/m²)'])
print(f"Correlation between temperature and radiation: {correlation:.3f}")

# Scatter plot to show relationship
plt.figure(figsize=(10, 6))
plt.scatter(df_clean['direct_radiation (W/m²)'], df_clean['temperature_2m (°C)'], 
            alpha=0.5, s=10, color='purple')
plt.xlabel('Direct Radiation (W/m²)')
plt.ylabel('Temperature (°C)')
plt.title(f'Temperature vs Radiation (Correlation: {correlation:.3f})')
plt.grid(True, alpha=0.3)
plt.show()

# Monthly analysis
df_clean['month'] = df_clean.index.month
monthly_stats = df_clean.groupby('month').agg({
    'temperature_2m (°C)': ['mean', 'max', 'min'],
    'direct_radiation (W/m²)': ['mean', 'max']
}).round(2)

print("\nMonthly Statistics:")
print(monthly_stats)