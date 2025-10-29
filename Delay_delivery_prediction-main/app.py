import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.metrics import confusion_matrix, precision_recall_curve

st.set_page_config(page_title="Predictive Delivery Optimizer", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(os.path.join('models','model.pkl'))

@st.cache_data
def load_data():
    path = os.path.join('data','final_dataset.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

model = load_model() if os.path.exists(os.path.join('models','model.pkl')) else None
data = load_data()

st.title("Predictive Delivery Optimizer")

pages = ["Dashboard Overview", "Order Explorer", "Single Order Predictor", "Model Insights"]
page = st.sidebar.radio("Navigate", pages)

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0 if c not in ['carrier','priority','weather_impact','product_category','destination'] else 'Unknown'
    return df

numeric_features = [
    'distance_km','order_value_inr','traffic_delay_minutes','promised_margin_min',
    'priority_score','carrier_reliability_score','route_risk_score',
    'warehouse_utilization','expected_travel_time_min','fuel_consumption_l','toll_charges_inr'
]
categorical_features = ['carrier','priority','weather_impact','product_category','destination']

if page == "Dashboard Overview":
    st.subheader("KPIs")
    if data.empty:
        st.info("Run the EDA notebook to generate data/final_dataset.csv and models/model.pkl")
    else:
        total = len(data)
        on_time = (data.get('is_delayed', pd.Series([0]*total)) == 0).mean()*100
        avg_delay = data.get('delivery_delay_minutes', pd.Series([0]*total)).mean()
        at_risk = (data.get('delay_probability', pd.Series([0]*total)) > 0.5).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("On-Time Delivery %", f"{on_time:.1f}%")
        c2.metric("Average Delay (mins)", f"{avg_delay:.1f}")
        c3.metric("Orders at Risk", int(at_risk))

        st.subheader("Charts")
        if 'delivery_delay_minutes' in data.columns:
            st.bar_chart(data['delivery_delay_minutes'].clip(upper=240))
        if 'carrier' in data.columns and 'is_delayed' in data.columns:
            chart1 = data.groupby('carrier')['is_delayed'].mean().reset_index()
            st.bar_chart(chart1, x='carrier', y='is_delayed')
        if 'priority' in data.columns and 'is_delayed' in data.columns:
            chart2 = data.groupby('priority')['is_delayed'].mean().reset_index()
            st.bar_chart(chart2, x='priority', y='is_delayed')

elif page == "Order Explorer":
    st.subheader("Order Explorer")
    if data.empty:
        st.info("No data available.")
    else:
        df = data.copy()
        min_date = pd.to_datetime(df.get('order_date')).min() if 'order_date' in df.columns else None
        max_date = pd.to_datetime(df.get('order_date')).max() if 'order_date' in df.columns else None
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.slider("Date range", min_value=min_date.to_pydatetime(), max_value=max_date.to_pydatetime(), value=(min_date.to_pydatetime(), max_date.to_pydatetime()))
            mask = (pd.to_datetime(df['order_date']) >= date_range[0]) & (pd.to_datetime(df['order_date']) <= date_range[1])
            df = df[mask]
        carrier = st.multiselect("Carrier", sorted(df.get('carrier', pd.Series()).dropna().unique().tolist()))
        priority = st.multiselect("Priority", sorted(df.get('priority', pd.Series()).dropna().unique().tolist()))
        if carrier:
            df = df[df['carrier'].isin(carrier)]
        if priority:
            df = df[df['priority'].isin(priority)]
        cols = ['order_id','carrier','priority','delay_probability','recommended_action']
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        st.dataframe(df[cols].fillna(''))

elif page == "Single Order Predictor":
    st.subheader("Single Order Predictor")
    if model is None:
        st.info("Model not found. Train it via the notebook.")
    else:
        # Inputs
        distance_km = st.number_input("Distance (km)", 0.0, 5000.0, 100.0)
        order_value_inr = st.number_input("Order value (INR)", 0.0, 100000.0, 1000.0)
        traffic_delay_minutes = st.number_input("Traffic delay (min)", 0.0, 600.0, 30.0)
        promised_margin_min = st.number_input("Promised margin (min)", 0.0, 100000.0, 1440.0)
        priority = st.selectbox("Priority", ["Express","Standard","Economy"]) 
        priority_score = {"Express":3, "Standard":2, "Economy":1}[priority]
        carrier = st.text_input("Carrier", value="SpeedyLogistics")
        weather_impact = st.selectbox("Weather", ["None","Light_Rain","Heavy_Rain","Fog"]) 
        route_risk_score = {"None":0.0, "Light_Rain":0.3, "Heavy_Rain":0.8, "Fog":0.4}[weather_impact]
        warehouse_utilization = st.number_input("Warehouse utilization", 0.0, 5.0, 1.0)
        expected_travel_time_min = st.number_input("Expected travel time (min)", 0.0, 100000.0, 120.0)
        fuel_consumption_l = st.number_input("Fuel consumption (liters)", 0.0, 2000.0, 50.0)
        toll_charges_inr = st.number_input("Toll charges (INR)", 0.0, 5000.0, 200.0)

        row = pd.DataFrame([{k:v for k,v in locals().items() if k in (
            'distance_km','order_value_inr','traffic_delay_minutes','promised_margin_min',
            'priority_score','carrier_reliability_score','route_risk_score',
            'warehouse_utilization','expected_travel_time_min','fuel_consumption_l','toll_charges_inr',
            'carrier','priority','weather_impact','product_category','destination'
        )}])
        row['carrier_reliability_score'] = 0.7
        row['product_category'] = 'General'
        row['destination'] = 'Unknown'
        row = ensure_cols(row, numeric_features + categorical_features)

        proba = float(model.predict_proba(row)[0][1])
        st.metric("Probability of Delay", f"{proba:.2f}")
        if proba > 0.8:
            st.warning("Recommended: Reroute / Reassign / Expedite")
        elif proba > 0.5:
            st.info("Recommended: Monitor closely")
        else:
            st.success("No action required")

elif page == "Model Insights":
    st.subheader("Model Insights")
    if data.empty or model is None:
        st.info("Train the model first.")
    else:
        if 'is_delayed' in data.columns and 'delay_probability' in data.columns:
            y_true = data['is_delayed'].fillna(0).astype(int)
            y_score = data['delay_probability'].fillna(0).astype(float)
            cm = confusion_matrix(y_true, (y_score>0.5).astype(int))
            st.write("Confusion Matrix:")
            st.write(cm)
            pr, rc, th = precision_recall_curve(y_true, y_score)
            st.line_chart(pd.DataFrame({'precision':pr[:-1],'recall':rc[:-1]}))


