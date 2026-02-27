import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="CLV Prediction", layout="wide")

st.title("Customer Lifetime Value (CLV) Prediction")

st.write("""
This app predicts **future customer revenue** based on past transaction behavior.
Users can either use the **sample dataset** or upload their own dataset 
with the same structure.
""")

# -------------------------
# Load sample dataset
# -------------------------

if st.checkbox("Use sample dataset"):
    df = pd.read_csv("sample_data.csv")
else:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

# -------------------------
# Data preparation
# -------------------------
df = df.rename(columns={
    'Customer ID': 'customer_id',
    'InvoiceDate': 'transaction_date'
})

df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df = df.dropna(subset=['customer_id'])

df['transaction_amount'] = df['Quantity'] * df['Price']
df = df[df['transaction_amount'] > 0]

df = df[['customer_id', 'transaction_date', 'transaction_amount']]

# -------------------------
# Time split
# -------------------------
prediction_window_days = 180
max_date = df['transaction_date'].max()
cutoff_date = max_date - timedelta(days=prediction_window_days)

past_data = df[df['transaction_date'] <= cutoff_date]
future_data = df[df['transaction_date'] > cutoff_date]

# -------------------------
# Feature engineering
# -------------------------
reference_date = cutoff_date

customer_features = past_data.groupby('customer_id').agg(
    recency=('transaction_date', lambda x: (reference_date - x.max()).days),
    frequency=('transaction_amount', 'count'),
    avg_spend=('transaction_amount', 'mean'),
    tenure=('transaction_date', lambda x: (x.max() - x.min()).days)
).reset_index()

future_revenue = (
    future_data
    .groupby('customer_id')['transaction_amount']
    .sum()
    .reset_index(name='future_revenue')
)

training_data = customer_features.merge(
    future_revenue,
    on='customer_id',
    how='inner'
)

# -------------------------
# Train model
# -------------------------
X = training_data[['recency', 'frequency', 'avg_spend', 'tenure']]
y = training_data['future_revenue']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

st.subheader("Model Performance")
st.write(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
st.write(f"R² Score: {r2_score(y_test, preds):.2f}")

# -------------------------
# Predict CLV for all customers
# -------------------------
final_reference_date = df['transaction_date'].max() + timedelta(days=1)

all_customers = df.groupby('customer_id').agg(
    recency=('transaction_date', lambda x: (final_reference_date - x.max()).days),
    frequency=('transaction_amount', 'count'),
    avg_spend=('transaction_amount', 'mean'),
    tenure=('transaction_date', lambda x: (x.max() - x.min()).days)
).reset_index()

all_customers['predicted_clv'] = model.predict(
    all_customers[['recency', 'frequency', 'avg_spend', 'tenure']]
)

final_output = all_customers.sort_values(
    by='predicted_clv',
    ascending=False
)

st.subheader("Predicted Customer Lifetime Value")
st.dataframe(final_output.head(20))

# -------------------------
# Plot Top 10 customers
# -------------------------
top_10 = final_output.head(10)

plt.figure(figsize=(10, 5))
plt.bar(top_10['customer_id'].astype(str), top_10['predicted_clv'])
plt.xlabel("Customer ID")
plt.ylabel("Predicted CLV")
plt.title("Top 10 Customers")
plt.xticks(rotation=45)
st.pyplot(plt)
