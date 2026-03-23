import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load models
# -----------------------------
reg_model = joblib.load("retail_regression_model.joblib")
clf_model = joblib.load("retail_classification_model.joblib")

# -----------------------------
# Get model feature names
# -----------------------------
# StandardScaler preserves feature names in newer scikit-learn versions
try:
    reg_features = reg_model.named_steps['scaler'].feature_names_in_
    clf_features = clf_model.named_steps['scaler'].feature_names_in_
except:
    # fallback: define manually (from your training DataFrames)
    reg_features = ['Quantity','UnitPrice','Month','Day','Hour'] + [
        'Country_Austria','Country_Bahrain','Country_Belgium','Country_Brazil',
        'Country_Canada','Country_France','Country_Germany','Country_Ireland',
        'Country_Italy','Country_Netherlands','Country_Spain','Country_Sweden',
        'Country_Switzerland','Country_United Kingdom','Country_Unspecified'
    ]
    clf_features = reg_features.copy()

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Online Retail Prediction App")
st.sidebar.header("Input Features")

# Numeric features
Quantity = st.sidebar.number_input("Quantity", min_value=1, value=10)
UnitPrice = st.sidebar.number_input("Unit Price", min_value=0.01, value=5.0)
Month = st.sidebar.slider("Month", 1, 12, 6)
Day = st.sidebar.slider("Day", 1, 31, 15)
Hour = st.sidebar.slider("Hour", 0, 23, 12)

# Country selection dynamically
country_cols = [c for c in reg_features if 'Country_' in c]
selected_country = st.sidebar.selectbox("Country", [c.replace("Country_", "") for c in country_cols])

# Build input DataFrame
input_data = {
    "Quantity": Quantity,
    "UnitPrice": UnitPrice,
    "Month": Month,
    "Day": Day,
    "Hour": Hour
}

# Initialize all country columns to 0, set selected country to 1
for c in country_cols:
    input_data[c] = 1 if c == f"Country_{selected_country}" else 0

input_df = pd.DataFrame([input_data])

st.subheader("User Input")
st.write(input_df)

# -----------------------------
# Align input to model columns
# -----------------------------
def align_features(input_df, model_features):
    # Remove any columns not in model
    input_df = input_df[[c for c in input_df.columns if c in model_features]]
    # Add missing columns from model
    for c in model_features:
        if c not in input_df.columns:
            input_df[c] = 0
    # Reorder columns exactly as model expects
    return input_df[model_features]

input_df_reg = align_features(input_df.copy(), reg_features)
input_df_clf = align_features(input_df.copy(), clf_features)

# -----------------------------
# Predictions
# -----------------------------
reg_pred = reg_model.predict(input_df_reg)[0]
clf_pred = clf_model.predict(input_df_clf)[0]

st.subheader("Predictions")
st.write(f"Predicted Total Price (Regression): ${reg_pred:.2f}")
st.write(f"Predicted Target (High Price=1, Low Price=0): {clf_pred}")
# -----------------------------
# Predict Button
# -----------------------------
