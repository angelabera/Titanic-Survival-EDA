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


print("\n--- Handling Missing Age Values ---")
median_age = df["Age"].median()
df["Age"] = df["Age"].fillna(median_age)
print("Missing Age values after cleaning:", df["Age"].isnull().sum())


print("\n--- Handling Missing Embarked Values ---")
most_common_embarked = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(most_common_embarked)
print("Missing Embarked values after cleaning:",
      df["Embarked"].isnull().sum())


print("\n--- Handling Cabin Values ---")
df["Cabin_Available"] = df["Cabin"].notna().astype(int)
print(df[["Cabin", "Cabin_Available"]].head())


df.drop("Cabin", axis=1, inplace=True)


print("\n--- Overall Survival ---")
survival_count = df["Survived"].value_counts()
print(survival_count)