import streamlit as st
import pandas as pd

def show_predictions(model, df):
    st.subheader(":bar_chart: Prediction Results")

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    result_df = df.copy()
    result_df["No_Show_Prediction"] = predictions
    result_df["No_Show_Probability"] = probabilities

    st.dataframe(result_df.head())

    st.success("Predictions generated successfully!")