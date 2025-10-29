import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib


def find_csv(filename: str) -> Path:
    here = Path.cwd()
    candidates = [
        here / 'data' / filename,
        here / filename,
        here.parent / 'data' / filename,
        here.parent / filename,
    ]
    for parent in here.parents:
        candidates.append(parent / 'data' / filename)
        candidates.append(parent / filename)
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not locate {filename}. Tried: {[str(c) for c in candidates]}")


def main():
    # Load CSVs
    orders = pd.read_csv(find_csv('orders.csv'))
    perf = pd.read_csv(find_csv('delivery_performance.csv'))
    routes = pd.read_csv(find_csv('routes_distance.csv'))
    warehouse = pd.read_csv(find_csv('warehouse_inventory.csv'))
    feedback = pd.read_csv(find_csv('customer_feedback.csv'))
    costs = pd.read_csv(find_csv('cost_breakdown.csv'))

    # Standardize column names to lowercase for consistency
    orders.columns = orders.columns.str.lower()
    perf.columns = perf.columns.str.lower()
    routes.columns = routes.columns.str.lower()
    warehouse.columns = warehouse.columns.str.lower()
    feedback.columns = feedback.columns.str.lower()
    costs.columns = costs.columns.str.lower()

    # Handle dates - convert order_date to datetime
    if 'order_date' in orders.columns:
        orders['order_date'] = pd.to_datetime(orders['order_date'], errors='coerce')

    # Merge datasets
    for df in (orders, perf, routes, costs, feedback):
        if 'order_id' in df.columns:
            df['order_id'] = df['order_id'].astype(str)
    
    # Merge all datasets
    base = orders.merge(perf, on='order_id', how='left') 
    base = base.merge(routes, on='order_id', how='left') 
    base = base.merge(costs, on='order_id', how='left') 
    
    # Merge feedback if available
    if set(['order_id','rating']).issubset(feedback.columns):
        base = base.merge(feedback[['order_id','rating']], on='order_id', how='left')
    
    # Merge warehouse data based on origin city
    if 'origin' in base.columns and 'location' in warehouse.columns:
        # Map origin cities to warehouse locations
        warehouse_mapping = warehouse.groupby('location').first().reset_index()
        base = base.merge(warehouse_mapping, left_on='origin', right_on='location', how='left', suffixes=('', '_wh'))

    # Targets - calculate delay based on days difference
    if 'promised_delivery_days' in base.columns and 'actual_delivery_days' in base.columns:
        base['delivery_delay_days'] = base['actual_delivery_days'] - base['promised_delivery_days']
        base['delivery_delay_minutes'] = base['delivery_delay_days'] * 24 * 60  # Convert days to minutes
    else:
        base['delivery_delay_minutes'] = 0
    
    base['delivery_delay_minutes'] = base['delivery_delay_minutes'].fillna(0)
    base['is_delayed'] = (base['delivery_delay_minutes'] > 0).astype(int)

    # Features
    # Calculate promised margin in minutes (from order date to promised delivery)
    if 'order_date' in base.columns and 'promised_delivery_days' in base.columns:
        # Convert promised delivery days to actual datetime
        base['promised_delivery_date'] = base['order_date'] + pd.to_timedelta(base['promised_delivery_days'], unit='D')
        base['promised_margin_min'] = (
            (base['promised_delivery_date'] - base['order_date']).dt.total_seconds() / 60
        )
    else:
        base['promised_margin_min'] = 1440  # Default to 1 day in minutes
    
    base['promised_margin_min'] = base['promised_margin_min'].fillna(base['promised_margin_min'].median())

    # Priority mapping
    priority_map = {'Express':3, 'Standard':2, 'Economy':1}
    base['priority_score'] = base.get('priority', pd.Series(index=base.index)).map(priority_map).fillna(1)

    # Traffic delay handling
    if 'traffic_delay_minutes' in base.columns:
        td = base['traffic_delay_minutes'].fillna(0)
        base['traffic_intensity_score'] = (td - td.min()) / (td.max() - td.min() + 1e-9)
    else:
        base['traffic_intensity_score'] = 0

    # Carrier reliability score
    if 'carrier' in base.columns:
        carrier_on_time = base.groupby('carrier')['is_delayed'].apply(lambda s: 1 - s.mean())
        base['carrier_reliability_score'] = base['carrier'].map(carrier_on_time).fillna(carrier_on_time.mean() if len(carrier_on_time)>0 else 0.5)
    else:
        base['carrier_reliability_score'] = 0.5

    # Weather impact mapping
    weather_map = {'None':0.0, 'Light_Rain':0.3, 'Heavy_Rain':0.8, 'Fog':0.4}
    w = base.get('weather_impact', pd.Series(index=base.index)).map(weather_map).fillna(0.0)
    base['route_risk_score'] = base['traffic_intensity_score'] * 0.6 + w * 0.4

    # Warehouse utilization
    if 'current_stock_units' in base.columns and 'reorder_level' in base.columns:
        base['warehouse_utilization'] = (base['current_stock_units'] / base['reorder_level']).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    else:
        base['warehouse_utilization'] = 1.0

    # Expected travel time calculation
    if 'distance_km' in base.columns:
        avg_speed = 50.0
        base['expected_travel_time_min'] = (base['distance_km'] / (avg_speed + 1e-9)) * 60
    else:
        base['expected_travel_time_min'] = 0

    # Feature column mapping for new dataset structure
    numeric_features = [
        'distance_km','order_value_inr','traffic_delay_minutes','promised_margin_min',
        'priority_score','carrier_reliability_score','route_risk_score',
        'warehouse_utilization','expected_travel_time_min','fuel_consumption_l','toll_charges_inr'
    ]
    categorical_features = ['carrier','priority','weather_impact','product_category','destination']

    for col in list(numeric_features):
        if col not in base.columns:
            base[col] = 0
    for col in list(categorical_features):
        if col not in base.columns:
            base[col] = 'Unknown'

    base = base.sort_values('order_date')
    X = base[numeric_features + categorical_features]
    y = base['is_delayed']

    preprocess = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ]
    )
    clf = Pipeline(steps=[('preprocess', preprocess), ('model', RandomForestClassifier(n_estimators=300, random_state=42))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:,1]
    preds = (probs > 0.5).astype(int)
    metrics = {
        'accuracy': float(accuracy_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'f1': float(f1_score(y_test, preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, probs)) if len(np.unique(y_test))>1 else None
    }
    print('Metrics:', metrics)

    # Attach predictions for test rows (for metrics traceability)
    risk_test = pd.Series(probs, index=X_test.index)
    base.loc[X_test.index, 'delay_probability'] = risk_test
    base.loc[X_test.index, 'recommended_action'] = risk_test.apply(
        lambda p: 'Reroute/Reassign/Expedite' if p>0.8 else ('Monitor closely' if p>0.5 else 'No action required')
    )

    # Refit on ALL data and generate predictions for ALL orders so the app has complete coverage
    clf.fit(X, y)
    probs_all = clf.predict_proba(X)[:,1]
    risk_all = pd.Series(probs_all, index=X.index)
    base['delay_probability'] = risk_all
    base['recommended_action'] = risk_all.apply(
        lambda p: 'Reroute/Reassign/Expedite' if p>0.8 else ('Monitor closely' if p>0.5 else 'No action required')
    )

    # Save
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    joblib.dump(clf, os.path.join('models','model.pkl'))
    base.to_csv(os.path.join('data','final_dataset.csv'), index=False)
    print('Saved model to models/model.pkl and dataset to data/final_dataset.csv')


if __name__ == '__main__':
    main()


