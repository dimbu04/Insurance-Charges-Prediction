
import streamlit as st
import pandas as pd
import altair as alt
from snowflake.snowpark.context import get_active_session

# Page setup
st.set_page_config(
    page_title="Insurance Charges Prediction",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Insurance Charges Prediction Dashboard")
st.caption("Predict and analyze insurance charges using the Snowflake ML model.")

# Snowflake connection
session = get_active_session()

# Load prediction history
st.subheader("📊 Prediction Overview")

try:
    history = session.table(
        "INSURANCE_ML.ML_PIPE.INSURANCE_OUTPUT"
    ).to_pandas()
except Exception as e:
    history = pd.DataFrame()
    st.warning(f"Could not load prediction history: {e}")

# Dashboard metrics
if not history.empty and "PREDICTED_CHARGES" in history.columns:
    total = len(history)
    average = history["PREDICTED_CHARGES"].mean()
    highest = history["PREDICTED_CHARGES"].max()
else:
    total = 0
    average = 0
    highest = 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Predictions", total)
col2.metric("Average Predicted Charges", f"${average:,.2f}")
col3.metric("Highest Predicted Charges", f"${highest:,.2f}")

# Chart and prediction results
if not history.empty and "PREDICTED_CHARGES" in history.columns:
    st.subheader("📊 Predicted Insurance Charges")

    if "AGE" in history.columns:
        chart_data = history[["AGE", "PREDICTED_CHARGES"]].copy()

        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("AGE:Q", title="Age"),
            y=alt.Y("PREDICTED_CHARGES:Q", title="Predicted Charges"),
            tooltip=["AGE", "PREDICTED_CHARGES"]
        ).properties(height=350)

        st.altair_chart(chart, use_container_width=True)

    st.subheader("📋 Prediction Results")
    st.dataframe(history, use_container_width=True)
else:
    st.info("No prediction history is available yet.")

# Manual prediction
st.subheader("🔮 Predict Insurance Charges")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["female", "male"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=27.5)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox(
        "Region",
        ["southwest", "southeast", "northwest", "northeast"]
    )

    submitted = st.form_submit_button("🚀 Predict Charges")

if submitted:
    try:
        # Insert the new record into the input table
        session.sql(
            """
            INSERT INTO INSURANCE_ML.ML_PIPE.INSURANCE_INPUT
                (AGE, SEX, BMI, CHILDREN, SMOKER, REGION)
            SELECT ?, ?, ?, ?, ?, ?
            """,
            params=[age, sex, bmi, children, smoker, region]
        ).collect()

        # Run the stored procedure to process the input
        result = session.sql(
            "CALL INSURANCE_ML.ML_PIPE.PROCESS_INSURANCE_DATA()"
        ).collect()

        st.success("Prediction processing completed.")

        # Refresh the latest prediction history
        updated_history = session.table(
            "INSURANCE_ML.ML_PIPE.INSURANCE_OUTPUT"
        ).to_pandas()

        if not updated_history.empty:
            latest = updated_history.iloc[-1]
            predicted_value = float(latest["PREDICTED_CHARGES"])
            st.success(
                f"Predicted Insurance Charges: ${predicted_value:,.2f}"
            )
            st.dataframe(updated_history.tail(5), use_container_width=True)

        st.rerun()

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))

# Model information
st.subheader("ℹ️ Model Information")

info1, info2, info3 = st.columns(3)
info1.write("**Model:** INSURANCE_CHARGES_MODEL")
info2.write("**Version:** PRETTY_WORM_2")
info3.write("**Model Type:** XGBoost Regression")
