import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show_feature_importance(model):
    st.subheader(":chart_with_upwards_trend: Feature Importance")

    try:
        classifier = model.named_steps["classifier"]
        importance = classifier.feature_importances_

        preprocessor = model.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()

        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(15)

        fig, ax = plt.subplots()
        ax.barh(feat_df["Feature"], feat_df["Importance"])
        ax.invert_yaxis()
        st.pyplot(fig)

    except:
        st.warning("Feature importance not available.")