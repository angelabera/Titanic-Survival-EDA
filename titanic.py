import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("titanic-survival-dataset.csv")

print("Dataset loaded successfully!")

# Basic inspection
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns)

print("\n--- Dataset Information ---")
df.info()

print("\n--- Statistical Summary ---")
print(df.describe())

# Missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Duplicate rows
print("\n--- Duplicate Rows ---")
duplicates = df.duplicated().sum()
print("Number of duplicate rows:", duplicates)