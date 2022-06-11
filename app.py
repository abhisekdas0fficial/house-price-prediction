import pandas as pd
import streamlit as st
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import pickle

LL = pd.read_csv("LL.csv")

st.set_page_config(page_title="House Price Prediction App", page_icon="🏠")
st.title("House Price Prediction App")

with open ("model.pkl", "rb") as f:
    model = pickle.load(f)


location = st.sidebar.selectbox("Select Location", LL.location.unique())
bhk = st.number_input(
        "Total number of bedrooms in the house: ",
        min_value=1,
        max_value=8,
        value=2
    )
bath = st.number_input(
        "Total number of bathrooms in the house: " ,
        min_value=1,
        max_value=8,
        value=2
    )

total_sqft = st.slider("Total Square Feet", 800, 25000, value=1500, step=100)

if st.button("Predict"):
    hdf = pd.DataFrame({
        "location"   : location,
        "total_sqft" : total_sqft,
        "bath"       : bath,
        "bhk"        : bhk
    }, index=[1])
    y_pred = model.predict(hdf)
    st.write(f"# ₹{y_pred[0]:.2f} lakhs")
