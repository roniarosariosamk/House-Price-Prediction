import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="House Price Prediction", page_icon="🏡", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("modified_data.csv")

@st.cache_resource
def train_model():
    df = load_data()
    df = df[df["price"] > 0].copy()

    y = df["price"]
    X = df.drop(columns=["price", "price_per_sqft", "date", "street"])

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    model = Pipeline([
        ("prep", preprocess),
        ("model", LinearRegression())
    ])
    model.fit(X, y)
    return model, df, num_cols, cat_cols

try:
    model, df, num_cols, cat_cols = train_model()
except Exception as exc:
    st.error(f"Could not load the dataset or train the model: {exc}")
    st.info("Make sure modified_data.csv is in the same folder as app.py.")
    st.stop()

st.title("🏡 House Price Prediction")
st.write("Estimate a property's sale price using a Linear Regression model trained on the supplied real-estate dataset.")
st.caption(f"Training records: {len(df):,} | Model: Linear Regression")

with st.form("prediction_form"):
    st.subheader("Enter property details")
    col1, col2 = st.columns(2)
    values = {}

    with col1:
        values["bedrooms"] = st.number_input("Bedrooms", min_value=0.0, max_value=20.0, value=3.0, step=1.0)
        values["bathrooms"] = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
        values["sqft_living"] = st.number_input("Living area (sqft)", min_value=100, max_value=20000, value=1500, step=50)
        values["sqft_lot"] = st.number_input("Lot area (sqft)", min_value=100, max_value=1000000, value=5000, step=100)
        values["floors"] = st.number_input("Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5)
        values["waterfront"] = st.selectbox("Waterfront", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        values["view"] = st.selectbox("View rating", sorted(df["view"].dropna().unique().tolist()))

    with col2:
        values["condition"] = st.selectbox("Condition", sorted(df["condition"].dropna().unique().tolist()))
        values["sqft_above"] = st.number_input("Above-ground area (sqft)", min_value=0, max_value=20000, value=1500, step=50)
        values["sqft_basement"] = st.number_input("Basement area (sqft)", min_value=0, max_value=10000, value=0, step=50)
        values["yr_built"] = st.number_input("Year built", min_value=1800, max_value=2026, value=1990, step=1)
        values["yr_renovated"] = st.number_input("Year renovated (0 if never)", min_value=0, max_value=2026, value=0, step=1)
        values["city"] = st.selectbox("City", sorted(df["city"].dropna().unique().tolist()))
        values["statezip"] = st.selectbox("State ZIP", sorted(df["statezip"].dropna().unique().tolist()))

    submitted = st.form_submit_button("Predict house price", type="primary")

if submitted:
    input_df = pd.DataFrame([values])
    input_df = input_df[num_cols + cat_cols]
    prediction = float(model.predict(input_df)[0])
    st.success("Prediction complete")
    st.metric("Estimated house price", f"${prediction:,.0f}")
    st.caption("This is a model estimate, not a property appraisal or guaranteed market value.")

with st.expander("About this model"):
    st.write("The app uses the Linear Regression model and preprocessing approach from the supplied project script: median imputation and standardization for numeric columns, plus most-frequent imputation and one-hot encoding for categorical columns.")
    st.write("The model is fitted on all valid-price rows in the supplied CSV when the app starts.")
