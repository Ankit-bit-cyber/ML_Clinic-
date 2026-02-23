import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ML Clinic", layout="wide")

st.title("🏥 Clinical Appointment No-Show Prediction System")
st.markdown("Milestone 1 Demo | Stable Deployment Version")

st.subheader("📂 Upload Patient Data (CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

def compute_risk_score(df):
    """
    Simple rule-based scoring system to simulate ML model.
    Stable and deployment-safe.
    """

    risk = np.zeros(len(df))

    # Lead time effect
    risk += np.clip(df["lead_time"] / 30, 0, 1) * 0.25

    # SMS not received increases risk
    risk += (df["SMS_received"] == 0) * 0.20

    # Alcoholism
    risk += df["Alcoholism"] * 0.15

    # Diabetes
    risk += df["Diabetes"] * 0.10

    # Hypertension
    risk += df["Hipertension"] * 0.10

    # Age factor
    risk += np.where(df["Age"] < 20, 0.10, 0)
    risk += np.where(df["Age"] > 65, 0.10, 0)

    return np.clip(risk, 0, 1)


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    try:
        # -----------------------
        # Feature Engineering
        # -----------------------

        df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
        df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

        df["lead_time"] = (
            df["AppointmentDay"] - df["ScheduledDay"]
        ).dt.days

        df = df[df["lead_time"] >= 0]
        df = df[df["Age"] >= 0]

        # -----------------------
        # Risk Prediction
        # -----------------------

        df["No_Show_Probability"] = compute_risk_score(df)
        df["Predicted_No_Show"] = (df["No_Show_Probability"] > 0.5).astype(int)

        st.subheader("📊 Prediction Results")
        st.dataframe(df.head())

        st.metric(
            label="Average No-Show Risk",
            value=f"{df['No_Show_Probability'].mean():.2%}"
        )

        st.success("Predictions generated successfully!")

    except Exception as e:
        st.error(f"Error during prediction: {e}")