

import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

 
# 1. Load Data
 
df = pd.read_csv("data/processed/final_data.csv")
print("✅ Data loaded:", df.shape)

 
# 2. Prepare Features & Target
 
# Encode categorical columns so the model can use them
le = LabelEncoder()
if "land_cover" in df.columns:
    df["land_cover"] = le.fit_transform(df["land_cover"].astype(str))
if "soil_type" in df.columns:
    df["soil_type"] = le.fit_transform(df["soil_type"].astype(str))

# Features the model will learn from
FEATURE_COLS = [
    "rainfall", "temperature", "humidity",
    "river_discharge", "water_level", "elevation",
    "land_cover", "soil_type",
    "population_density", "infrastructure",
    "past_disasters", "risk_index"
]

# Target: what we want to predict (0=Low, 1=Medium, 2=High)
TARGET_COL = "risk_label"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

 
# 3. Train / Test Split  (80% train, 20% test)
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Train size: {len(X_train)}  |  Test size: {len(X_test)}")

 
# 4. Train Models
 

# --- Model A: Logistic Regression ---
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_preds)
print(f"\n📊 Logistic Regression Accuracy : {lr_acc * 100:.2f}%")
print("   Detailed Report:")
print(classification_report(y_test, lr_preds, target_names=["Low", "Medium", "High"]))

# --- Model B: Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"📊 Random Forest Accuracy        : {rf_acc * 100:.2f}%")
print("   Detailed Report:")
print(classification_report(y_test, rf_preds, target_names=["Low", "Medium", "High"]))

# ── Side-by-side comparison ──────────────────────────────
print("=" * 45)
print(f"  Logistic Regression : {lr_acc * 100:.2f}%")
print(f"  Random Forest       : {rf_acc * 100:.2f}%")
print("=" * 45)

 
# 5. Pick the Best Model
 
if rf_acc >= lr_acc:
    best_model = rf_model
    best_preds = rf_preds
    print("\n🏆 Best Model: Random Forest")
else:
    best_model = lr_model
    best_preds = lr_preds
    print("\n🏆 Best Model: Logistic Regression")

 
# 6. Add Predictions Column to Full Dataset
 
# Run the best model on ALL rows (not just test) so visualization has predictions for every point
all_preds = best_model.predict(X)
df["predicted_risk"] = all_preds

# Save updated dataset for Member 4 to use
df.to_csv("data/processed/final_data.csv", index=False)
print("\nPredictions added → data/processed/final_data.csv")

 
# 7. Save the Model as model.pkl
 
os.makedirs("outputs/models", exist_ok=True)
with open("outputs/models/model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Model saved  → outputs/models/model.pkl")