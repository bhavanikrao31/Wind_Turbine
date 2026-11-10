import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv("wind_clean.csv")
rank = pd.read_csv("ranking.csv", index_col=0)

# Only look at times when the turbine should be producing well
df = df[df["Theoretical_Power"] > 500].copy()

# Baseline = the 4 best turbines (lowest loss)
healthy_ids = list(rank.index[-4:])
print("Healthy baseline turbines:", healthy_ids)

df["ws_bin"] = df["WindSpeed"].round()
df["pw_bin"] = df["ActivePower"] // 200
base = df[df["Turbine_ID"].isin(healthy_ids)]
rpm_base = base.groupby("ws_bin")["Rotor_RPM"].median()
temp_base = base.groupby("pw_bin")["Generator_Temp"].median()

# How different is this reading from a healthy turbine?
df["rpm_ratio"] = (df["Rotor_RPM"] / df["ws_bin"].map(rpm_base)).fillna(1)
df["temp_excess"] = (df["Generator_Temp"] - df["pw_bin"].map(temp_base)).fillna(0)

# Underperforming = produces >10% less than the theoretical curve
df["Loss_ratio"] = (df["Theoretical_Power"] - df["ActivePower"]) / df["Theoretical_Power"]
df["Underperforming"] = (df["Loss_ratio"] > 0.10).astype(int)

# ---- Rule-based probable cause engine ----
def diagnose(r):
    if r["Underperforming"] == 0:
        return "Normal"
    if r["temp_excess"] > 8:
        return "Gearbox Friction / Overheating"
    if r["rpm_ratio"] < 0.85:
        return "Pitch Fault / Icing"
    return "Sensor Drift / Other"

df["Cause"] = df.apply(diagnose, axis=1)

summary = df.groupby("Turbine_ID").agg(
    Underperf_Time_pct=("Underperforming", lambda s: round(s.mean() * 100, 1)))
summary["Probable_Cause"] = (df[df.Underperforming == 1]
    .groupby("Turbine_ID")["Cause"].agg(lambda s: s.value_counts().idxmax()))
summary = summary.join(rank["Loss_%"]).sort_values("Loss_%", ascending=False)
summary.loc[summary["Loss_%"] < 3, "Probable_Cause"] = "Healthy"
print("\nDIAGNOSIS PER TURBINE:\n", summary)

# ---- Exp 8: Logistic regression ----
feats = ["WindSpeed", "Rotor_RPM", "Generator_Temp", "rpm_ratio", "temp_excess"]
X = StandardScaler().fit_transform(df[feats])
y = df["Underperforming"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
pred = model.predict(Xte)

print("\nAccuracy:", round(accuracy_score(yte, pred), 3))
print("Confusion matrix:\n", confusion_matrix(yte, pred))
print(classification_report(yte, pred, target_names=["Normal", "Underperforming"]))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(yte, pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Under"], yticklabels=["Normal", "Under"])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.savefig("charts/8_confusion.png", dpi=120, bbox_inches="tight")
plt.close()

df.drop(columns=["ws_bin", "pw_bin"]).to_csv("wind_diag.csv", index=False)
summary.to_csv("diagnosis.csv")
print("Saved diagnosis.csv and wind_diag.csv")