import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

df = pd.read_csv("wind_diag.csv")
feats = ["WindSpeed", "Theoretical_Power", "ActivePower",
         "Rotor_RPM", "Generator_Temp", "Loss_ratio"]
X, y = df[feats], df["Cause"]
print("Class counts:\n", y.value_counts())

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1),
}

rows = []
for name, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    cv = cross_val_score(m, Xtr, ytr, cv=5, scoring="f1_macro").mean()
    rows.append({"Model": name,
                 "Accuracy": round(accuracy_score(yte, p), 4),
                 "F1_macro": round(f1_score(yte, p, average="macro"), 4),
                 "CV_F1": round(cv, 4)})
res = pd.DataFrame(rows).sort_values("F1_macro", ascending=False)
print("\nMODEL COMPARISON:\n", res.to_string(index=False))

best_name = res.iloc[0]["Model"]
best = models[best_name]
pred = best.predict(Xte)
print("\nBEST MODEL:", best_name)
print(classification_report(yte, pred))

# Chart 9: model comparison
res.set_index("Model")[["Accuracy", "F1_macro", "CV_F1"]].plot(
    kind="bar", figsize=(8, 4), ylim=(0.5, 1.02))
plt.title("Model Comparison")
plt.xticks(rotation=15)
plt.savefig("charts/9_model_comparison.png", dpi=120, bbox_inches="tight")
plt.close()

# Chart 10: feature importance (Random Forest)
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=feats).sort_values()
plt.figure(figsize=(7, 4))
imp.plot(kind="barh", color="#00A8CC")
plt.title("Feature Importance (Random Forest)")
plt.savefig("charts/10_feature_importance.png", dpi=120, bbox_inches="tight")
plt.close()

# Chart 11: confusion matrix of best model
labels = sorted(y.unique())
plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix(yte, pred, labels=labels), annot=True, fmt="d",
            cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.xticks(rotation=20, ha="right")
plt.savefig("charts/11_cause_confusion.png", dpi=120, bbox_inches="tight")
plt.close()

joblib.dump({"model": best, "features": feats, "name": best_name,
             "accuracy": float(res.iloc[0]["Accuracy"]),
             "f1": float(res.iloc[0]["F1_macro"])}, "model.pkl")
print("\nModel saved as model.pkl")