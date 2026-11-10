import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("charts", exist_ok=True)
df = pd.read_csv("wind_scada.csv", parse_dates=["Timestamp"])

# ---- Exp 4: Data preprocessing ----
df = df.drop_duplicates()
df = df.sort_values(["Turbine_ID", "Timestamp"]).reset_index(drop=True)
print("Rows after removing duplicates:", len(df))

# Compare 3 imputation methods on Rotor_RPM
rpm_mean = df["Rotor_RPM"].fillna(df["Rotor_RPM"].mean())
rpm_median = df["Rotor_RPM"].fillna(df["Rotor_RPM"].median())
rpm_ffill = df.groupby("Turbine_ID")["Rotor_RPM"].ffill()
print("RPM average -> mean fill: %.3f | median fill: %.3f | ffill: %.3f"
      % (rpm_mean.mean(), rpm_median.mean(), rpm_ffill.mean()))

# Final choice: forward fill per turbine, median for any leftovers
for col in ["Rotor_RPM", "Generator_Temp"]:
    df[col] = df.groupby("Turbine_ID")[col].ffill()
    df[col] = df[col].fillna(df[col].median())
print("Missing values left:", df.isnull().sum().sum())

# Encoding categorical variable
print(pd.get_dummies(df["Turbine_ID"]).head())
df["Turbine_Code"] = df["Turbine_ID"].astype("category").cat.codes

# ---- Exp 5: Outlier detection and treatment ----
df["Power_Diff"] = df["ActivePower"] - df["Theoretical_Power"]
q1, q3 = df["Power_Diff"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
iqr_out = (df["Power_Diff"] < low) | (df["Power_Diff"] > high)
z = (df["Power_Diff"] - df["Power_Diff"].mean()) / df["Power_Diff"].std()
z_out = z.abs() > 3
print("IQR outliers:", iqr_out.sum(), "| Z-score outliers:", z_out.sum())

# Treatment: remove only impossible readings (power ABOVE theoretical).
# Low readings are real faults - we must keep them!
spikes = df["Power_Diff"] > high
df = df[~spikes].reset_index(drop=True)
print("Spike rows removed:", spikes.sum(), "| Final rows:", len(df))

df.to_csv("wind_clean.csv", index=False)

# ---- Exp 3: Visualization ----
cols = ["WindSpeed", "Theoretical_Power", "ActivePower", "Rotor_RPM", "Generator_Temp"]

plt.figure(figsize=(7, 4))
sns.histplot(df["WindSpeed"], bins=40, kde=True)
plt.title("Wind Speed Distribution")
plt.savefig("charts/1_histogram.png", dpi=120, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x="Turbine_ID", y="Power_Diff")
plt.title("Power Loss vs Theoretical (kW) per Turbine")
plt.savefig("charts/2_boxplot.png", dpi=120, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 5))
sub = df[df["Turbine_ID"].isin(["T1", "T7"])]
sns.scatterplot(data=sub, x="WindSpeed", y="ActivePower", hue="Turbine_ID", s=8)
plt.title("Healthy T1 vs Faulty T7")
plt.savefig("charts/3_scatter.png", dpi=120, bbox_inches="tight")
plt.close()

sns.pairplot(df[cols].sample(1500, random_state=1), corner=True)
plt.savefig("charts/4_pairplot.png", dpi=100, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 5))
sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("charts/5_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()

print("Charts saved in the 'charts' folder")