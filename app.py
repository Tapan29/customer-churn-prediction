import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("churn_model.pkl")

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦"
)

st.title("🏦 Bank Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to churn."
)

# -------------------------
# Customer Details
# -------------------------

age = st.number_input("Age", 18, 100, 30)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

num_dependents = st.number_input(
    "Number of Dependents",
    0,
    10,
    0
)

education = st.selectbox(
    "Education Level",
    [
        "High School",
        "Bachelor",
        "Master",
        "PhD"
    ]
)

employment = st.selectbox(
    "Employment Status",
    [
        "Employed",
        "Self-Employed",
        "Unemployed",
        "Retired"
    ]
)

annual_income = st.number_input(
    "Annual Income",
    0,
    1000000,
    50000
)

credit_score = st.slider(
    "Credit Score",
    300,
    900,
    650
)

account_balance = st.number_input(
    "Account Balance",
    0.0,
    1000000.0,
    10000.0
)

credit_limit = st.number_input(
    "Credit Limit",
    0.0,
    1000000.0,
    50000.0
)

days_without_activity = st.slider(
    "Days Without Activity",
    0,
    365,
    30
)

complaints = st.slider(
    "Complaints Filed Last Year",
    0,
    20,
    0
)

# -------------------------
# Predict Button
# -------------------------

if st.button("Predict Churn"):

    input_data = pd.DataFrame({
        "CustomerID":[0],
        "Age":[age],
        "Gender":[gender],
        "MaritalStatus":[marital_status],
        "NumDependents":[num_dependents],
        "EducationLevel":[education],
        "EmploymentStatus":[employment],
        "AnnualIncome":[annual_income],
        "CreditScore":[credit_score],
        "AccountAgeMonths":[24],
        "NumProducts":[2],
        "HasCreditCard":[1],
        "HasCheckingAccount":[1],
        "HasSavingsAccount":[1],
        "HasLoan":[0],
        "HasMortgage":[0],
        "AccountBalance":[account_balance],
        "CreditLimit":[credit_limit],
        "MonthlyTransactions":[20],
        "AvgTransactionAmount":[500],
        "MonthlyAccountSpending":[5000],
        "ATMUsageFrequency":[5],
        "OnlineBankingUsage":[10],
        "MobileAppUsage":[10],
        "ComplaintsFiledLastYear":[complaints],
        "ComplaintsResolved":[complaints],
        "ServiceCallsLastMonth":[1],
        "CreditUtilizationRatio":[0.3],
        "DaysWithoutActivity":[days_without_activity]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )

    if prediction == 1:
        st.error(
            "Customer likely to churn"
        )
    else:
        st.success(
            "Customer likely to stay"
        )

    if probability > 0.8:
        st.warning(
            "High Risk Customer. Consider retention offers."
        )