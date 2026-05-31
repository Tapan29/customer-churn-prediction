import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTabs [data-baseweb="tab-list"] button { font-size: 16px; padding: 10px 20px; }
    </style>
    """, unsafe_allow_html=True)

# ==================== SIDEBAR INFO ====================
st.sidebar.title("🔧 System Status")

# ==================== CHECK FILES ====================
@st.cache_data
def check_files():
    """Check which files exist"""
    files_status = {}
    
    if os.path.exists('banking_customers.csv'):
        files_status['data'] = True
    else:
        files_status['data'] = False
    
    rf_exists = os.path.exists('random_forest_model.pkl')
    lr_exists = os.path.exists('logistic_regression_model.pkl')
    churn_exists = os.path.exists('churn_model.pkl')
    
    files_status['rf'] = rf_exists
    files_status['lr'] = lr_exists
    files_status['churn'] = churn_exists
    
    return files_status

files = check_files()

# Display file status in sidebar
if files['data']:
    st.sidebar.success("✅ Data file found")
else:
    st.sidebar.error("❌ banking_customers.csv missing")

if files['rf']:
    st.sidebar.success("✅ RF model found")
elif files['churn']:
    st.sidebar.success("✅ Churn model found")
else:
    st.sidebar.error("❌ RF model missing")

if files['lr']:
    st.sidebar.success("✅ LR model found")
else:
    st.sidebar.warning("⚠️ LR model missing")

st.sidebar.divider()

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    """Load models with comprehensive error handling"""
    models = {}
    errors = []
    
    try:
        if os.path.exists('random_forest_model.pkl'):
            with open('random_forest_model.pkl', 'rb') as f:
                models['Random Forest'] = pickle.load(f)
                st.sidebar.success("✅ Random Forest loaded")
        elif os.path.exists('churn_model.pkl'):
            with open('churn_model.pkl', 'rb') as f:
                models['Random Forest'] = pickle.load(f)
                st.sidebar.info("📌 Using churn_model.pkl")
        else:
            errors.append("❌ Random Forest model not found")
    except pickle.UnpicklingError as e:
        errors.append(f"❌ RF corrupt: {str(e)[:50]}")
    except Exception as e:
        errors.append(f"❌ RF error: {str(e)[:50]}")
    
    try:
        if os.path.exists('logistic_regression_model.pkl'):
            with open('logistic_regression_model.pkl', 'rb') as f:
                models['Logistic Regression'] = pickle.load(f)
                st.sidebar.success("✅ Logistic Regression loaded")
        else:
            st.sidebar.warning("⚠️ LR model not found")
    except Exception as e:
        errors.append(f"❌ LR error: {str(e)[:50]}")
    
    return models, errors

models_dict, loading_errors = load_models()

if loading_errors:
    st.sidebar.divider()
    for error in loading_errors:
        st.sidebar.warning(error)

if not models_dict:
    st.error("""
    ❌ **CRITICAL ERROR: No models found!**
    
    **Required files missing:**
    - `random_forest_model.pkl` OR `churn_model.pkl`
    
    **How to fix:**
    1. Train your models using: `python churn_prediction_complete.py`
    2. Save the generated `.pkl` files
    3. Upload them to your GitHub repository
    4. Push changes: `git add . && git commit -m "Add models" && git push`
    5. Streamlit will redeploy automatically
    """)
    st.stop()

# ==================== LOAD DATA ====================
@st.cache_resource
def load_data():
    """Load training data"""
    try:
        df = pd.read_csv('banking_customers.csv')
        st.sidebar.success("✅ Data loaded")
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.sidebar.error(f"Data error: {str(e)[:30]}")
        return None

df = load_data()

# ==================== TITLE ====================
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("Predict customer churn and identify at-risk customers for targeted retention")

# ==================== MODEL SELECTION ====================
st.sidebar.divider()
st.sidebar.header("⚙️ Configuration")

available_models = list(models_dict.keys())
if len(available_models) > 0:
    model_choice = st.sidebar.radio(
        "Select Model:",
        available_models,
        help="Choose which model to use for predictions"
    )
    selected_model = models_dict.get(model_choice)
    
    if selected_model is None:
        st.error(f"❌ Error: {model_choice} model is None")
        st.stop()
else:
    st.error("No models available!")
    st.stop()

# ==================== TABS ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🔮 Make Prediction",
    "📈 Model Performance",
    "📁 Batch Prediction"
])

# ==================== TAB 1: DASHBOARD ====================
with tab1:
    st.header("Dashboard")
    
    if df is not None and len(df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", len(df))
        with col2:
            churn_count = df['Churn'].sum() if 'Churn' in df.columns else 0
            st.metric("Churned", f"{churn_count}/{len(df)}")
        with col3:
            retained = len(df) - (df['Churn'].sum() if 'Churn' in df.columns else 0)
            st.metric("Retained", retained)
        with col4:
            avg_age = df['Age'].mean() if 'Age' in df.columns else 0
            st.metric("Avg Age", f"{avg_age:.0f}")
        
        st.divider()
        
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Churn' in df.columns:
                    st.subheader("Churn Distribution")
                    churn_data = df['Churn'].value_counts()
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.pie(churn_data.values, labels=['Retained', 'Churned'], autopct='%1.1f%%',
                           colors=['#2ecc71', '#e74c3c'], startangle=90)
                    st.pyplot(fig)
                    plt.close()
            
            with col2:
                if 'Age' in df.columns and 'Churn' in df.columns:
                    st.subheader("Age by Churn Status")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    df[df['Churn'] == 0]['Age'].hist(bins=20, alpha=0.6, label='Retained', ax=ax)
                    df[df['Churn'] == 1]['Age'].hist(bins=20, alpha=0.6, label='Churned', ax=ax)
                    ax.legend()
                    st.pyplot(fig)
                    plt.close()
        
        except Exception as e:
            st.warning(f"Could not generate visualizations: {str(e)[:50]}")
    else:
        st.info("📊 Upload banking_customers.csv to see dashboard")

# ==================== TAB 2: MAKE PREDICTION ====================
with tab2:
    st.header("🔮 Individual Customer Prediction")
    st.write("Enter customer details below")
    
    if selected_model is None:
        st.error("❌ Model not loaded properly")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            age = st.slider("Age", 18, 80, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
            dependents = st.slider("Dependents", 0, 4, 1)
            education = st.selectbox("Education", ["High School", "Bachelor", "Master", "Doctorate"])
            employment = st.selectbox("Employment", ["Employed", "Self-Employed", "Retired", "Unemployed"])
        
        with col2:
            st.subheader("Financial Info")
            income = st.slider("Annual Income ($)", 20000, 200000, 100000, 5000)
            credit_score = st.slider("Credit Score", 300, 850, 700)
            balance = st.slider("Account Balance ($)", 0, 500000, 100000, 10000)
            cc_limit = st.slider("Credit Limit ($)", 1000, 50000, 10000, 1000)
            cc_util = st.slider("Credit Utilization", 0.0, 1.0, 0.5, 0.1)
        
        with col3:
            st.subheader("Account & Behavior")
            account_age = st.slider("Account Age (mo)", 1, 120, 60)
            num_products = st.slider("# Products", 1, 5, 3)
            has_cc = int(st.checkbox("Credit Card", True))
            has_checking = int(st.checkbox("Checking", True))
            has_savings = int(st.checkbox("Savings", True))
            has_loan = int(st.checkbox("Loan", False))
            has_mortgage = int(st.checkbox("Mortgage", False))
            monthly_trans = st.slider("Monthly Trans", 1, 100, 30)
            avg_trans = st.slider("Avg Trans ($)", 50, 5000, 1000)
            monthly_spend = st.slider("Monthly Spend ($)", 500, 20000, 5000)
            atm = st.slider("ATM Usage", 0, 20, 5)
            online = st.slider("Online Banking (d)", 0, 30, 15)
            mobile = st.slider("Mobile App (d)", 0, 20, 10)
            complaints = st.slider("Complaints", 0, 5, 0)
            resolved = st.slider("Resolved", 0, 5, 0)
            calls = st.slider("Service Calls", 0, 3, 1)
            inactivity = st.slider("Days No Activity", 0, 180, 30)
        
        if st.button("🎯 Predict Churn Risk", use_container_width=True, key="pred_btn"):
            try:
                input_data = pd.DataFrame({
                    'CustomerID': [9999],
                    'Age': [age],
                    'Gender': [gender],
                    'MaritalStatus': [marital],
                    'NumDependents': [dependents],
                    'EducationLevel': [education],
                    'EmploymentStatus': [employment],
                    'AnnualIncome': [income],
                    'CreditScore': [credit_score],
                    'AccountAgeMonths': [account_age],
                    'NumProducts': [num_products],
                    'HasCreditCard': [has_cc],
                    'HasCheckingAccount': [has_checking],
                    'HasSavingsAccount': [has_savings],
                    'HasLoan': [has_loan],
                    'HasMortgage': [has_mortgage],
                    'AccountBalance': [balance],
                    'CreditLimit': [cc_limit],
                    'MonthlyTransactions': [monthly_trans],
                    'AvgTransactionAmount': [avg_trans],
                    'MonthlyAccountSpending': [monthly_spend],
                    'ATMUsageFrequency': [atm],
                    'OnlineBankingUsage': [online],
                    'MobileAppUsage': [mobile],
                    'ComplaintsFiledLastYear': [complaints],
                    'ComplaintsResolved': [resolved],
                    'ServiceCallsLastMonth': [calls],
                    'CreditUtilizationRatio': [cc_util],
                    'DaysWithoutActivity': [inactivity],
                })
                
                if selected_model is None or not hasattr(selected_model, 'predict'):
                    st.error("❌ Model is not valid or not loaded")
                else:
                    try:
                        pred = selected_model.predict(input_data)[0]
                        prob = selected_model.predict_proba(input_data)[0][1]
                        
                        st.divider()
                        st.subheader("📊 Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if prob > 0.7:
                                st.error(f"⚠️ **HIGH RISK: {prob*100:.1f}%**")
                                st.write("→ Immediate action needed")
                            elif prob > 0.5:
                                st.warning(f"⚡ **MEDIUM RISK: {prob*100:.1f}%**")
                                st.write("→ Monitor closely")
                            else:
                                st.success(f"✅ **LOW RISK: {prob*100:.1f}%**")
                                st.write("→ Maintain engagement")
                        
                        with col2:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            risk_levels = ['Low\n(0-33%)', 'Medium\n(33-67%)', 'High\n(67-100%)']
                            risk_vals = [0.33, 0.34, 0.33]
                            colors = ['#2ecc71', '#f39c12', '#e74c3c']
                            ax.barh(risk_levels, risk_vals, color=colors)
                            ax.axvline(prob, color='black', linestyle='--', linewidth=2)
                            ax.set_xlim(0, 1)
                            ax.set_title('Risk Level', fontweight='bold')
                            st.pyplot(fig)
                            plt.close()
                    
                    except Exception as pred_error:
                        st.error(f"❌ Prediction failed: {str(pred_error)[:100]}")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)[:100]}")

# ==================== TAB 3: MODEL PERFORMANCE ====================
with tab3:
    st.header("📈 Model Performance")
    
    if df is not None and 'Churn' in df.columns:
        try:
            X = df.drop("Churn", axis=1)
            y = df["Churn"]
            
            y_pred = selected_model.predict(X)
            y_pred_proba = selected_model.predict_proba(X)[:, 1]
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred)
            rec = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            auc = roc_auc_score(y, y_pred_proba)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{acc*100:.1f}%")
            col2.metric("Precision", f"{prec*100:.1f}%")
            col3.metric("Recall", f"{rec*100:.1f}%")
            col4.metric("F1-Score", f"{f1*100:.1f}%")
            col5.metric("ROC-AUC", f"{auc*100:.1f}%")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                cm = confusion_matrix(y, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                fpr, tpr, _ = roc_curve(y, y_pred_proba)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f'AUC={auc:.3f}')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.legend()
                st.pyplot(fig)
                plt.close()
        
        except Exception as e:
            st.warning(f"Performance metrics unavailable: {str(e)[:50]}")
    else:
        st.info("⚠️ Data needed for performance metrics")

# ==================== TAB 4: BATCH PREDICTION ====================
with tab4:
    st.header("📁 Batch Prediction")
    
    uploaded = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            st.write(f"📊 {len(batch_df)} records loaded")
            
            if st.button("🎯 Predict All", use_container_width=True):
                preds = selected_model.predict(batch_df)
                probs = selected_model.predict_proba(batch_df)[:, 1]
                
                results = batch_df.copy()
                results['Churn_Probability'] = probs
                results['Risk_Level'] = ['High' if p > 0.7 else ('Medium' if p > 0.5 else 'Low') for p in probs]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total", len(results))
                col2.metric("High Risk", (results['Risk_Level'] == 'High').sum())
                col3.metric("Medium Risk", (results['Risk_Level'] == 'Medium').sum())
                
                st.dataframe(results[['Churn_Probability', 'Risk_Level']].head(20))
                
                csv = results.to_csv(index=False)
                st.download_button("📥 Download", csv, "predictions.csv", "text/csv", use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)[:100]}")

st.divider()
st.markdown("<div style='text-align:center; color:gray; font-size:11px;'>Bank Churn Prediction v2.0</div>", unsafe_allow_html=True)