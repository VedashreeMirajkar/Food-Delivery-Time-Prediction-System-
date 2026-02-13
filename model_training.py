# =========================================
# Food Delivery Time Prediction Model
# Gradient Boosting + Stacking (FINAL)
# =========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge


# =========================================
# 1. LOAD DATA
# =========================================

df = pd.read_csv("Dataset.csv")

print("Original shape:", df.shape)


# =========================================
# 2. CLEANING
# =========================================

# Rename target column (IMPORTANT FIX)
df.rename(columns={"Delivery Time_taken(min)": "Delivery_Time"}, inplace=True)

# Drop useless columns
df.drop(columns=["ID", "Delivery_person_ID"], errors="ignore", inplace=True)

# Drop missing
df.dropna(inplace=True)

print("After cleaning:", df.shape)


# =========================================
# 3. DISTANCE FEATURE (Haversine)
# =========================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) *
         np.cos(np.radians(lat2)) *
         np.sin(dlon/2)**2)

    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


df["distance_km"] = haversine(
    df["Restaurant_latitude"],
    df["Restaurant_longitude"],
    df["Delivery_location_latitude"],
    df["Delivery_location_longitude"]
)


# =========================================
# 4. FEATURE ENGINEERING (UNIQUE FEATURES)
# =========================================

df["efficiency_score"] = df["Delivery_person_Ratings"] / (df["distance_km"] + 1)
df["age_experience_factor"] = df["Delivery_person_Age"] * df["Delivery_person_Ratings"]
df["rating_distance"] = df["Delivery_person_Ratings"] * df["distance_km"]
df["vehicle_distance"] = df["distance_km"] * 1


# =========================================
# 5. ENCODING
# =========================================

le_order = LabelEncoder()
le_vehicle = LabelEncoder()

df["Type_of_order"] = le_order.fit_transform(df["Type_of_order"])
df["Type_of_vehicle"] = le_vehicle.fit_transform(df["Type_of_vehicle"])


# =========================================
# 6. REMOVE OUTLIERS (optional boost accuracy)
# =========================================

df = df[df["Delivery_Time"] < df["Delivery_Time"].quantile(0.99)]


# =========================================
# 7. SELECT FEATURES
# =========================================

features = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Type_of_order",
    "Type_of_vehicle",
    "distance_km",
    "efficiency_score",
    "age_experience_factor",
    "rating_distance",
    "vehicle_distance"
]

X = df[features]
y = df["Delivery_Time"]


# =========================================
# 8. TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train:", X_train.shape, " Test:", X_test.shape)


# =========================================
# 9. SCALING
# =========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================================
# 10. STACKING MODEL (FAST + ACCURATE)
# =========================================

gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1)
rf = RandomForestRegressor(n_estimators=80, max_depth=10, n_jobs=-1)

stack_model = StackingRegressor(
    estimators=[
        ("gb", gb),
        ("rf", rf)
    ],
    final_estimator=Ridge()
)

print("Training model...")
stack_model.fit(X_train, y_train)


# =========================================
# 11. EVALUATION
# =========================================

pred = stack_model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nModel Performance")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 3))


# =========================================
# 12. SAVE MODEL
# =========================================

joblib.dump((stack_model, scaler), "delivery_model.pkl")

print("\nModel saved as delivery_model.pkl ✅")
