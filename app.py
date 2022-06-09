import pandas as pd
import streamlit as st
import pickle

st.set_page_config(page_title="House Price Prediction App", page_icon="🏠")
st.title("House Price Prediction App")

with open ("model.pkl", "rb") as f:
    model = pickle.load(f)

def prediction(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

x = st.number_input("Enter the total number of rooms in the house: ")

if st.button("Predict"):
    X_test = pd.DataFrame({"x": [x]})
    y_pred = prediction(model, X_test)
    st.success(f"The price of the house is {y_pred[0]:.2f}")
