# train_models.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

DATA = Path("data/dataset.csv")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

def load_and_prepare():
    df = pd.read_csv(DATA, parse_dates=["date"])
    # create lag features: previous 7-day avg of steps, active, calories_in
    df = df.sort_values(["user_id","date"])
    df['steps_7d_avg'] = df.groupby('user_id')['steps'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['active_7d_avg'] = df.groupby('user_id')['active_min'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['cal_in_7d_avg'] = df.groupby('user_id')['calories_in'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    # target: calories_burned (same day) for calories model
    # For BMI change model: predict bmi change after 14 days relative to current bmi
    df['bmi_future_14d'] = df.groupby('user_id')['bmi'].shift(-14)
    df['bmi_change_14d'] = df['bmi_future_14d'] - df['bmi']
    df = df.dropna(subset=['bmi_change_14d'])
    return df

def train_calories_model(df):
    features = ['age','sex','height_cm','weight_kg','steps','active_min','sleep_h']
    # encode sex
    df['sex_m'] = (df['sex']=="M").astype(int)
    X = df[['age','sex_m','height_cm','weight_kg','steps','active_min','sleep_h']]
    y = df['calories_burned']
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42)
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    print("Calories model RMSE:", np.sqrt(mean_squared_error(y_val, preds)))
    joblib.dump(rf, MODELS/"calories_model.joblib")
    print("Saved calories model.")
    return rf

def train_bmi_change_model(df):
    # predict bmi_change_14d using current habits and 7-day averages
    df['sex_m'] = (df['sex']=="M").astype(int)
    feats = ['age','sex_m','height_cm','weight_kg','steps_7d_avg','active_7d_avg','cal_in_7d_avg','sleep_h','drank_alcohol','smoked']
    X = df[feats]
    y = df['bmi_change_14d']
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42)
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    print("BMI-change model RMSE:", np.sqrt(mean_squared_error(y_val, preds)))
    joblib.dump(rf, MODELS/"bmi_change_14d_model.joblib")
    print("Saved bmi change model.")
    return rf

def train_anomaly_detector(df):
    # use numeric features to detect anomalies in daily logs
    df_num = df[['steps','active_min','sleep_h','calories_in','calories_burned','weight_kg']]
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    iso.fit(df_num)
    joblib.dump(iso, MODELS/"anomaly_detector.joblib")
    print("Saved anomaly detector.")
    return iso

if __name__ == "__main__":
    df = load_and_prepare()
    print("Rows available for training:", len(df))
    train_calories_model(df)
    train_bmi_change_model(df)
    train_anomaly_detector(df)
