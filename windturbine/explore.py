import numpy as np
import pandas as pd

# ---- Exp 1: NumPy and Pandas basics ----
arr = np.array([1, 2, 3, 4, 5])
print("NumPy sum:", arr.sum(), "| mean:", arr.mean(), "| x2:", arr * 2)

df = pd.read_csv("wind_scada.csv")
print("\nFirst 5 rows:\n", df.head())
print("\nIndexing - row 0:\n", df.iloc[0])
print("\nSlicing - rows 0-4, 2 columns:\n", df.loc[0:4, ["WindSpeed", "ActivePower"]])
print("\nFiltering - Turbine T5 only:\n", df[df["Turbine_ID"] == "T5"].head())

# ---- Exp 2: Exploratory Data Analysis ----
print("\nShape (rows, columns):", df.shape)
print("\nColumn types:\n", df.dtypes)
print("\nDescriptive statistics:\n", df.describe())
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nRows per turbine:\n", df["Turbine_ID"].value_counts())