import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("wind_clean.csv")

# ---- Underperformance ranking (core feature) ----
run = df[df["Theoretical_Power"] > 0]
rank = run.groupby("Turbine_ID").agg(
    Theoretical_kWh=("Theoretical_Power", "sum"),
    Actual_kWh=("ActivePower", "sum"))
rank["Loss_%"] = ((rank.Theoretical_kWh - rank.Actual_kWh)
                  / rank.Theoretical_kWh * 100).round(2)
rank = rank.sort_values("Loss_%", ascending=False)
print("\nUNDERPERFORMANCE RANKING:\n", rank)
rank.to_csv("ranking.csv")

# ---- Exp 6: Feature engineering ----
feat = ["WindSpeed", "Rotor_RPM", "Generator_Temp", "ActivePower"]
mm = MinMaxScaler().fit_transform(df[feat])
print("\nNormalized min/max:", mm.min(), mm.max())
Xs = StandardScaler().fit_transform(df[feat])
print("Standardized mean:", round(Xs.mean(), 3), "| std:", round(Xs.std(), 3))

pca = PCA(n_components=2)
pcs = pca.fit_transform(Xs)
print("PCA explained variance:", pca.explained_variance_ratio_.round(3))
idx = np.random.default_rng(1).choice(len(df), 3000, replace=False)
plt.figure(figsize=(7, 5))
plt.scatter(pcs[idx, 0], pcs[idx, 1], s=5,
            c=df["Turbine_ID"].astype("category").cat.codes.iloc[idx], cmap="tab10")
plt.title("PCA (2 components)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.savefig("charts/6_pca.png", dpi=120, bbox_inches="tight")
plt.close()

# ---- Exp 7: Linear regression on HEALTHY turbines ----
h = df[df["Turbine_ID"].isin(["T1", "T2", "T3", "T8"]) & (df["WindSpeed"] > 3)].copy()
h["WS2"] = h["WindSpeed"] ** 2
h["WS3"] = h["WindSpeed"] ** 3

def fit(cols, name):
    Xtr, Xte, ytr, yte = train_test_split(h[cols], h["ActivePower"],
                                          test_size=0.2, random_state=1)
    m = LinearRegression().fit(Xtr, ytr)
    pred = m.predict(Xte)
    print(f"{name}: R2 = {r2_score(yte, pred):.3f} | "
          f"RMSE = {np.sqrt(mean_squared_error(yte, pred)):.1f} kW")
    return yte, pred

fit(["WindSpeed"], "Simple regression")
yte, pred = fit(["WindSpeed", "WS2", "WS3"], "Multiple regression")

# Checking assumptions: residuals
res = yte - pred
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(pred, res, s=3)
ax[0].axhline(0, color="red")
ax[0].set_title("Residuals vs Predicted (should look random)")
ax[1].hist(res, bins=40)
ax[1].set_title("Residual distribution (should be bell-shaped)")
plt.savefig("charts/7_residuals.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved ranking.csv and charts")