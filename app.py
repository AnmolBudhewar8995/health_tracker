# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import time

MODELS = Path("models")
DATA = Path("data/dataset.csv")

@st.cache_data
def load_models():
    calories = joblib.load(MODELS/"calories_model.joblib")
    bmi_change = joblib.load(MODELS/"bmi_change_14d_model.joblib")
    iso = joblib.load(MODELS/"anomaly_detector.joblib")
    return calories, bmi_change, iso

@st.cache_data
def load_data():
    return pd.read_csv(DATA, parse_dates=["date"])

calories_model, bmi_model, anomaly_model = load_models()
df_all = load_data()

# Custom CSS for styling
st.markdown("""
<style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Button effects */
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: bold;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #4ECDC4, #FF6B6B);
    }
    
    .stButton > button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover:before {
        left: 100%;
    }
    
    /* Mirror effect for cards/containers */
    .mirror-effect {
        position: relative;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .mirror-effect::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        border-radius: 15px;
        z-index: -1;
    }
    
    .mirror-effect::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        height: 50%;
        background: linear-gradient(to bottom, rgba(255,255,255,0.1), transparent);
        transform: scaleY(-1);
        opacity: 0.3;
        z-index: -1;
    }
    
    /* Progress bars styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        border-radius: 10px;
    }
    
    /* Metrics styling */
    .metric-container {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 15px;
        margin: 5px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
    }
    
    /* Text color adjustments */
    .stMarkdown, .stText, .stHeader, .stSubheader {
        color: white !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

st.title("🏃‍♂️ Personal Health & Fitness Tracker — AI Insights")

# Sidebar: pick a user
users = df_all['user_id'].unique().tolist()
user = st.sidebar.selectbox("👤 Select user", users)
st.sidebar.markdown("📤 Or upload your CSV with same schema (date,user_id,weight_kg,steps,active_min,sleep_h,calories_in)")

uploaded = st.sidebar.file_uploader("📤 Upload CSV (optional)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["date"])
else:
    df = df_all[df_all['user_id']==user].sort_values("date")

st.header(f"📊 Dashboard for {user} — {len(df)} records")

# Add tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🎯 Goals", "🔍 Analysis", "🤖 Predictions"])

with tab1:
    st.markdown('<div class="mirror-effect">', unsafe_allow_html=True)
    # show recent metrics
    latest = df.sort_values("date").iloc[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        col1.metric("Latest weight (kg)", latest['weight_kg'], delta=f"{latest['weight_kg'] - df['weight_kg'].iloc[-2]:.2f}" if len(df) > 1 else None)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        col2.metric("Latest BMI", latest['bmi'], delta=f"{latest['bmi'] - df['bmi'].iloc[-2]:.2f}" if len(df) > 1 else None)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        col3.metric("Latest steps (today)", int(latest['steps']), delta=f"{int(latest['steps'] - df['steps'].iloc[-2])}" if len(df) > 1 else None)
        st.markdown('</div>', unsafe_allow_html=True)

    # Quick stats
    st.subheader("📊 Quick Stats")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        avg_steps = df['steps'].tail(7).mean()
        st.metric("7-Day Avg Steps", f"{int(avg_steps)}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        avg_active = df['active_min'].tail(7).mean()
        st.metric("7-Day Avg Active Min", f"{int(avg_active)}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col6:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        avg_sleep = df['sleep_h'].tail(7).mean()
        st.metric("7-Day Avg Sleep", f"{avg_sleep:.1f}h")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    # Daily Goals Progress with Animation
    st.subheader("🎯 Daily Goals Progress")
    goals_col1, goals_col2, goals_col3 = st.columns(3)

    # Steps goal (assuming 10,000 steps)
    steps_goal = 10000
    steps_progress = min(latest['steps'] / steps_goal, 1.0)
    with goals_col1:
        st.metric("Steps Goal", f"{int(latest['steps'])}/{steps_goal}")
        progress_bar = st.progress(0)
        for percent_complete in range(int(steps_progress * 100) + 1):
            progress_bar.progress(percent_complete)
            time.sleep(0.01)  # Animation effect
        if steps_progress >= 1.0:
            st.success("🎉 Steps goal achieved!")

    # Active minutes goal (assuming 30 minutes)
    active_goal = 30
    active_progress = min(latest['active_min'] / active_goal, 1.0)
    with goals_col2:
        st.metric("Active Minutes Goal", f"{int(latest['active_min'])}/{active_goal}")
        progress_bar = st.progress(0)
        for percent_complete in range(int(active_progress * 100) + 1):
            progress_bar.progress(percent_complete)
            time.sleep(0.01)  # Animation effect
        if active_progress >= 1.0:
            st.success("🎉 Active minutes goal achieved!")

    # Sleep goal (assuming 8 hours)
    sleep_goal = 8
    sleep_progress = min(latest['sleep_h'] / sleep_goal, 1.0)
    with goals_col3:
        st.metric("Sleep Goal", f"{latest['sleep_h']:.1f}/{sleep_goal}")
        progress_bar = st.progress(0)
        for percent_complete in range(int(sleep_progress * 100) + 1):
            progress_bar.progress(percent_complete)
            time.sleep(0.01)  # Animation effect
        if sleep_progress >= 1.0:
            st.success("🎉 Sleep goal achieved!")

with tab3:
    # time series plots
    st.subheader("📈 Trends")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['weight_kg'], mode='lines+markers', name='Weight (kg)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['date'], y=df['bmi'], mode='lines+markers', name='BMI', line=dict(color='red')))
    fig.update_layout(title="Weight and BMI Trends", xaxis_title="Date", yaxis_title="Value", updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode='immediate')]), dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])])])
    frames = [go.Frame(data=[go.Scatter(x=df['date'][:k+1], y=df['weight_kg'][:k+1], mode='lines+markers'), go.Scatter(x=df['date'][:k+1], y=df['bmi'][:k+1], mode='lines+markers')]) for k in range(len(df))]
    fig.frames = frames
    st.plotly_chart(fig, width='stretch')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['date'], y=df['steps'], mode='lines+markers', name='Steps', line=dict(color='green')))
    fig2.add_trace(go.Scatter(x=df['date'], y=df['active_min'], mode='lines+markers', name='Active Minutes', line=dict(color='orange')))
    fig2.update_layout(title="Activity Trends", xaxis_title="Date", yaxis_title="Count", updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode='immediate')]), dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])])])
    frames2 = [go.Frame(data=[go.Scatter(x=df['date'][:k+1], y=df['steps'][:k+1], mode='lines+markers'), go.Scatter(x=df['date'][:k+1], y=df['active_min'][:k+1], mode='lines+markers')]) for k in range(len(df))]
    fig2.frames = frames2
    st.plotly_chart(fig2, width='stretch')

    # anomaly detection
    st.subheader("🔍 Anomaly Detection (Recent)")
    numeric_cols = ['steps','active_min','sleep_h','calories_in','calories_burned','weight_kg']
    recent = df.sort_values("date").tail(60)
    anom_flags = anomaly_model.predict(recent[numeric_cols])
    recent = recent.assign(anomaly = (anom_flags==-1))
    st.dataframe(recent[['date','weight_kg','steps','active_min','calories_in','calories_burned','anomaly']].tail(20))

with tab4:
    # calories prediction (interactive)
    st.subheader("🔥 Calories Burned Predictor")
    age = st.number_input("Age", value=int(latest['age']), min_value=10, max_value=100)
    sex = st.selectbox("Sex", ["M","F"], index=0 if latest['sex']=="M" else 1)
    height = st.number_input("Height (cm)", value=float(latest['height_cm']))
    weight = st.number_input("Weight (kg)", value=float(latest['weight_kg']))
    steps = st.number_input("Planned steps", value=int(latest['steps']))
    active = st.number_input("Active minutes", value=int(latest['active_min']))
    sleep = st.number_input("Expected sleep hours", value=float(latest['sleep_h']))

    if st.button("🔮 Predict Calories Burned"):
        sex_m = 1 if sex=="M" else 0
        X = pd.DataFrame([{
            "age": age, "sex_m": sex_m, "height_cm": height, "weight_kg": weight,
            "steps": steps, "active_min": active, "sleep_h": sleep
        }])
        pred = calories_model.predict(X)[0]
        st.success(f"🎯 Estimated calories burned: {int(pred)} kcal")

    # weekly forecasting of weight (simple linear regression on weekly averages)
    st.subheader("📅 Weekly Forecast (Next 4 Weeks) — Weight")
    # aggregate weekly
    df_week = df.set_index("date").select_dtypes(include=[np.number]).resample("W").mean().reset_index().dropna(subset=['weight_kg'])
    if len(df_week) < 4:
        st.info("📊 Need at least 4 weeks of data for forecasting")
    else:
        X_weeks = np.arange(len(df_week)).reshape(-1,1)
        y_weight = df_week['weight_kg'].values
        lr = LinearRegression()
        lr.fit(X_weeks, y_weight)
        future_idx = np.arange(len(df_week), len(df_week)+4).reshape(-1,1)
        preds = lr.predict(future_idx)
        future_dates = pd.date_range(df_week['date'].iloc[-1] + pd.Timedelta(days=7), periods=4, freq='W')
        out = pd.DataFrame({"date": future_dates, "pred_weight_kg": np.round(preds,2)})
        st.table(out)
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(x=df_week['date'], y=df_week['weight_kg'], mode='lines+markers', name='Historical', line=dict(color='blue')))
        fig_w.add_trace(go.Scatter(x=out['date'], y=out['pred_weight_kg'], mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash')))
        fig_w.update_layout(title="Weight Forecast", xaxis_title="Date", yaxis_title="Weight (kg)")
        st.plotly_chart(fig_w, width='stretch')

    # predict BMI change after 14 days
    st.subheader("⚖️ BMI Change Prediction (14 Days)")
    if st.button("🔮 Predict BMI Change"):
        # use latest rolling 7d averages for this user
        last = df.sort_values("date").iloc[-1]
        steps_7 = df['steps'].tail(7).mean()
        active_7 = df['active_min'].tail(7).mean()
        cal_in_7 = df['calories_in'].tail(7).mean()
        sex_m = 1 if last['sex']=="M" else 0
        X = pd.DataFrame([{
            "age": last['age'], "sex_m": sex_m, "height_cm": last['height_cm'],
            "weight_kg": last['weight_kg'],
            "steps_7d_avg": steps_7, "active_7d_avg": active_7, "cal_in_7d_avg": cal_in_7,
            "sleep_h": last['sleep_h'], "drank_alcohol": last['drank_alcohol'], "smoked": last['smoked']
        }])
        pred_change = bmi_model.predict(X)[0]
        st.write(f"📈 Predicted BMI change in 14 days: {pred_change:.3f}")
        st.write(f"🎯 Estimated future BMI: {last['bmi'] + pred_change:.3f}")

st.info("All models are local and saved under models/ — retrain with `python train_models.py` to update.")
