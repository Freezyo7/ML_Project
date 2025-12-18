import streamlit as st
import pandas as pd
import pickle

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
model = pickle.load(open("churn_model.pkl", "rb"))

# -------------------------------------------------
# Header Section
# -------------------------------------------------
st.markdown("""
# 📉 Customer Churn Prediction Web App
**Predict customer churn using Machine Learning**

This application helps businesses identify customers
who are likely to discontinue their services.
""")

st.divider()

# -------------------------------------------------
# Input Section
# -------------------------------------------------
st.markdown("## 🧾 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    tenure = st.number_input("Tenure (months)", 0, 100, 12)
    contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

with col3:
    total_spend = st.number_input("Total Spend", 0.0, 100000.0, 5000.0)
    subscription = st.selectbox(
        "Subscription Type", ["Basic", "Standard", "Premium"]
    )

st.markdown("## 📊 Usage & Support Metrics")

col4, col5, col6, col7 = st.columns(4)

with col4:
    usage = st.number_input("Usage Frequency", 0, 100, 20)

with col5:
    support = st.number_input("Support Calls", 0, 50, 2)

with col6:
    delay = st.number_input("Payment Delay (days)", 0, 60, 5)

with col7:
    last_interaction = st.number_input("Last Interaction (days ago)", 0, 365, 30)

# -------------------------------------------------
# Prepare Input Data
# -------------------------------------------------
input_df = pd.DataFrame([{
    "Age": age,
    "Tenure": tenure,
    "Usage Frequency": usage,
    "Support Calls": support,
    "Payment Delay": delay,
    "Total Spend": total_spend,
    "Last Interaction": last_interaction,
    "Gender": gender,
    "Subscription Type": subscription,
    "Contract Length": contract
}])

# -------------------------------------------------
# Prediction
# -------------------------------------------------
st.divider()

if st.button("🔍 Predict Churn", use_container_width=True):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("## 📌 Prediction Result")

    colA, colB = st.columns(2)

    with colA:
        if prediction == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is NOT likely to churn")

    with colB:
        st.metric("Churn Probability", f"{probability:.2%}")

    # -------------------------------------------------
    # Business Interpretation
    # -------------------------------------------------
    if prediction == 1:
        st.markdown("""
        ### 🔔 Recommended Business Actions
        - Provide personalized retention offers  
        - Improve customer support experience  
        - Engage with proactive communication  
        """)
    else:
        st.markdown("""
        ### 🎯 Customer Status
        - Customer shows strong engagement  
        - Continue current service strategy  
        """)

# -------------------------------------------------
# Model Info Section
# -------------------------------------------------
with st.expander("📈 Model & Technical Details"):
    st.markdown("""
    **Algorithm:** Logistic Regression  
    **Preprocessing:**
    - StandardScaler (Numerical Features)
    - OneHotEncoder (Categorical Features)

    **Why Logistic Regression?**
    - Interpretable
    - Probability-based predictions
    - Suitable for binary classification

    **Evaluation Metrics Used:**
    - Accuracy
    - ROC-AUC
    """)

st.divider()

st.caption("Built using Python, Scikit-learn & Streamlit")
