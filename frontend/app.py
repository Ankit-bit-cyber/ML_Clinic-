import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.pkl"

model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="ML Clinic", layout="wide")

st.title(":hospital: Clinical Appointment No-Show Prediction System")
st.markdown("Built with XGBoost | AUC ~ 0.73")

st.subheader(":open_file_folder: Upload Patient Data (CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    try:
        # ---------------------------
        # TEMP FEATURE ENGINEERING
        # ---------------------------
        df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"]).dt.normalize()
        df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"]).dt.normalize()

        df["lead_time"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days
        df = df[df["lead_time"] >= 0]

        df["appointment_day_of_week"] = df["AppointmentDay"].dt.day_name()

        df = df[df["Age"] >= 0]

        df = df.drop(columns=["PatientId", "AppointmentID"], errors="ignore")

        # ---------------------------
        # Prediction
        # ---------------------------
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        df["Predicted_No_Show"] = predictions
        df["No_Show_Probability"] = probabilities

        st.subheader(":bar_chart: Prediction Results")
        st.dataframe(df.head())

        st.success("Predictions generated successfully!")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
    st.metric(
    label="Average No-Show Risk",
    value=f"{df['No_Show_Probability'].mean():.2%}"
)