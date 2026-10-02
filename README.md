# Titanic Survival EDA

## Overview

This project performs Exploratory Data Analysis (EDA) on the classic Titanic dataset using Python and Pandas.

The objective is to understand the structure of the dataset, identify and handle missing values and duplicate records, and analyze passenger survival patterns using statistical operations and grouping.

## Dataset

The dataset is the `titanic-survival-dataset.csv` file from Kaggle's Titanic: Machine Learning from Disaster competition.

It contains information about 891 passengers and 12 attributes, including:

- Passenger ID
- Survival status
- Passenger class
- Name
- Sex
- Age
- Number of siblings/spouses aboard
- Number of parents/children aboard
- Ticket
- Fare
- Cabin
- Port of embarkation

## Technologies Used

- Python
- Pandas
- NumPy

## EDA Performed

### 1. Dataset Inspection

The following Pandas operations were used:

- `head()` to view the first five records
- `shape` to determine the number of rows and columns
- `columns` to inspect column names
- `info()` to examine data types and non-null values
- `describe()` to obtain statistical summaries

The dataset contains:

- **891 rows**
- **12 columns**

### 2. Missing Value Analysis

Missing values were identified using:

```python
df.isnull().sum()
```

Missing values were found mainly in:

- `Age`
- `Cabin`
- `Embarked`

### 3. Data Cleaning

The following cleaning operations were performed:

- Missing `Age` values were replaced with the median age.
- Missing `Embarked` values were replaced with the most frequently occurring embarkation port.
- A new `Cabin_Available` feature was created to indicate whether cabin information was available.
- The original `Cabin` column was removed because a large proportion of its values were missing.
- Duplicate rows were checked.

No duplicate rows were found in the dataset.

### 4. Survival Analysis

The overall survival rate was calculated using the `Survived` column.

- Total passengers: **891**
- Survivors: **342**
- Non-survivors: **549**
- Overall survival rate: **38.38%**

### 5. Survival by Gender

The survival rate was analyzed using `groupby()`.

| Gender | Passengers | Survivors | Survival Rate |
|---|---:|---:|---:|
| Female | 314 | 233 | 74.20% |
| Male | 577 | 109 | 18.89% |

### 6. Survival by Gender and Passenger Class

Survival rates were further analyzed by combining gender and passenger class.

| Gender | Class | Survival Rate |
|---|---:|---:|
| Female | 1st | 96.81% |
| Female | 2nd | 92.11% |
| Female | 3rd | 50.00% |
| Male | 1st | 36.89% |
| Male | 2nd | 15.74% |
| Male | 3rd | 13.54% |

## Key Findings

1. The dataset contains 891 passenger records.
2. The overall survival rate was 38.38%.
3. Survival rates differed substantially between male and female passengers in this dataset.
4. Passenger class was also associated with different survival rates.
5. The combination of gender and passenger class revealed additional differences in survival patterns.
6. Pandas `groupby()` and aggregation functions were useful for identifying these patterns.
7. Missing values were successfully identified and handled during the data-cleaning process.

## How to Run

Clone or download this repository and make sure `titanic-survival-dataset.csv` and `titanic.py` are in the same directory.

Run:

```bash
python titanic.py
```

The analysis results will be displayed in the terminal.

## Project Structure

```text
Titanic-Survival-EDA/
│
├── titanic-survival-dataset.csv
├── titanic.py
└── README.md
```

## Learning Outcome

This project demonstrates the use of Pandas for:

- DataFrame operations
- Data inspection
- Statistical analysis
- Missing-value handling
- Duplicate detection
- Grouping and aggregation
- Exploratory data analysis