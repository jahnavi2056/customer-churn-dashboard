# =============================================================================
# Step 1: Import Libraries
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# Step 2: Define All Functions
# =============================================================================

@st.cache_data
def load_and_train_model():
    """
    This function loads the dataset, cleans it, trains the model,
    and returns a DataFrame with results.
    """
    # Using a relative path is better practice than a hardcoded path
    df = pd.read_excel("Use case 2 - customer churn data.xlsx")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df['Churn_numeric'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    X = df.drop(['customerID', 'Churn', 'Churn_numeric'], axis=1)
    y = df['Churn_numeric']

    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', RandomForestClassifier(random_state=42))])
    model_pipeline.fit(X, y)

    churn_risk_scores = model_pipeline.predict_proba(X)[:, 1]
    results_df = df.copy()
    results_df['ChurnRiskScore'] = churn_risk_scores
    return results_df

def get_recommendation(row):
    """
    This function returns a specific recommendation based on a prioritized
    set of business rules, using more customer details.
    """
    
    # Priority 1: New customers are the most critical to onboard correctly.
    if row['tenure'] <= 6:
        return "Enroll in 'First 90 Days' Onboarding Program"
    
    # Priority 2: High-value customers who are suddenly at risk.
    elif row['tenure'] > 24 and row['MonthlyCharges'] > 70:
        return "Offer Loyalty Discount & Plan Review"
        
    # Priority 3: Customers with premium services but lacking support.
    elif row['InternetService'] == 'Fiber optic' and row['TechSupport'] == 'No':
        return "Schedule Tech Health Check & Offer Support Trial"
        
    # Priority 4: Customers with many services but no security add-ons.
    elif row['MonthlyCharges'] > 75 and row['OnlineSecurity'] == 'No':
        return "Offer 'Value & Security' Package Bundle"
        
    # Priority 5: Customers using a high-friction payment method.
    elif row['PaymentMethod'] == 'Electronic check':
        return "Incentivize Switch to AutoPay (e.g., small credit)"
        
    # Priority 6: The most common, general reason for churn.
    elif row['Contract'] == 'Month-to-month':
        return "Offer 1-Year Contract Upgrade"
        
    # Default action if no other specific conditions are met.
    else:
        return "Standard Retention Check-in Call"

# =============================================================================
# Step 3: Prepare Data and Recommendations
# =============================================================================
df_results = load_and_train_model()
# Apply the function to create the new 'RecommendedAction' column
df_results['RecommendedAction'] = df_results.apply(get_recommendation, axis=1)


# =============================================================================
# Step 4: Streamlit App User Interface
# =============================================================================
st.set_page_config(layout="wide")
st.title("🚀 Customer Churn Prediction Dashboard")
st.write("This dashboard predicts which customers are at high risk of churning.")

# --- Sidebar ---
st.sidebar.header("Project Information")
st.sidebar.info(
    "**Objective:** Proactively identify customers at risk of churning to enable targeted retention strategies."
)

# --- Main Dashboard ---
# Key Metrics
col1, col2, col3 = st.columns(3)
total_customers = len(df_results)
churned_customers = df_results['Churn_numeric'].sum()
churn_rate = churned_customers / total_customers
col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Overall Churn Rate", f"{churn_rate:.2%}")

st.markdown("---")

# Original Charts
col1, col2 = st.columns(2)
with col1:
    st.subheader("Churn by Contract Type")
    fig1 = px.histogram(df_results, x='Contract', color='Churn', barmode='group')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Churn by Internet Service")
    fig2 = px.histogram(df_results, x='InternetService', color='Churn', barmode='group')
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# High-Risk Customer Watchlist (Single, Corrected Version)
st.subheader("High-Risk Customer Watchlist")
st.write("This table shows customers ranked by their predicted churn risk, with a specific action recommended for each.")

num_customers_to_show = st.slider("Select number of top-risk customers to view:", 5, 100, 10)

high_risk_customers = df_results.sort_values(by='ChurnRiskScore', ascending=False)

# Define the columns to display, including the new 'RecommendedAction'
display_cols = ['customerID', 'tenure', 'Contract', 'ChurnRiskScore', 'RecommendedAction']

st.dataframe(high_risk_customers[display_cols].head(num_customers_to_show))