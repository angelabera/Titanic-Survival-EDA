import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("titanic-survival-dataset.csv")

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 2. VISUALIZATION SETTINGS
# ============================================================

sns.set_theme(style="whitegrid")

# Custom color palette
custom_palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]


# ============================================================
# 3. HISTOGRAM - AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Age",
    bins=20,
    color=custom_palette[0],
    edgecolor="black"
)

plt.title("Age Distribution of Titanic Passengers", fontsize=14)
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()

plt.show()


# ============================================================
# 4. BAR CHART - SURVIVAL BY GENDER
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Sex",
    hue="Survived",
    palette=custom_palette[:2]
)

plt.title("Survival Count by Gender", fontsize=14)
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.legend(
    title="Survived",
    labels=["No", "Yes"]
)

plt.tight_layout()

plt.show()


# ============================================================
# 5. BAR CHART - SURVIVAL BY PASSENGER CLASS
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Pclass",
    hue="Survived",
    palette=custom_palette[:2]
)

plt.title("Survival Count by Passenger Class", fontsize=14)
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.legend(
    title="Survived",
    labels=["No", "Yes"]
)

plt.tight_layout()

plt.show()


# ============================================================
# 6. SCATTER PLOT - AGE VS FARE
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Age",
    y="Fare",
    hue="Survived",
    palette=custom_palette[:2],
    alpha=0.7
)

plt.title("Age vs Fare by Survival Status", fontsize=14)
plt.xlabel("Age")
plt.ylabel("Fare")

plt.legend(
    title="Survived",
    labels=["No", "Yes"]
)

plt.tight_layout()

plt.show()


# ============================================================
# 7. BOX PLOT - FARE BY PASSENGER CLASS
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Pclass",
    y="Fare",
    hue="Pclass",
    palette=custom_palette[:3],
    legend=False
)

plt.title("Fare Distribution by Passenger Class", fontsize=14)
plt.xlabel("Passenger Class")
plt.ylabel("Fare")

plt.tight_layout()

plt.show()


# ============================================================
# 8. BOX PLOT - AGE BY SURVIVAL STATUS
# ============================================================

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Survived",
    y="Age",
    hue="Survived",
    palette=custom_palette[:2],
    legend=False
)

plt.title("Age Distribution by Survival Status", fontsize=14)
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")

plt.tight_layout()

plt.show()


# ============================================================
# 9. CORRELATION MATRIX
# ============================================================

# Select numerical columns
numeric_data = df.select_dtypes(include="number")

correlation_matrix = numeric_data.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)


# ============================================================
# 10. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Titanic Dataset", fontsize=15)
plt.tight_layout()

plt.show()


# ============================================================
# 11. SURVIVAL RATE BY GENDER
# ============================================================

survival_gender = df.groupby("Sex")["Survived"].mean() * 100

print("\nSurvival Rate by Gender:")
print(survival_gender)


# ============================================================
# 12. SURVIVAL RATE BY PASSENGER CLASS
# ============================================================

survival_class = df.groupby("Pclass")["Survived"].mean() * 100

print("\nSurvival Rate by Passenger Class:")
print(survival_class)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("      DATA VISUALIZATION COMPLETED")
print("========================================")

print("\nVisualizations created:")
print("1. Age Distribution Histogram")
print("2. Survival by Gender Bar Chart")
print("3. Survival by Passenger Class Bar Chart")
print("4. Age vs Fare Scatter Plot")
print("5. Fare by Passenger Class Box Plot")
print("6. Age by Survival Status Box Plot")
print("7. Correlation Matrix")
print("8. Correlation Heatmap")