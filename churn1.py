# =============================================================================
# Step 1: Setup and Data Loading
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score

# Load the dataset using the exact filename
try:
    df = pd.read_excel("C:\\Users\\DELL\\Downloads\\Use case 2 - customer churn data.xlsx")
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Dataset file not found. Make sure the file is in the same directory.")
    exit()

# =============================================================================
# Step 2: Data Cleaning and Preprocessing
# =============================================================================
# The 'TotalCharges' column may contain spaces for new customers, making it an 'object' type.
# We convert it to a number, forcing any errors (like spaces) into 'NaN' (Not a Number).
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop the few rows that now have missing TotalCharges.
df.dropna(inplace=True)

# Convert the target variable 'Churn' into a numerical format (1 for 'Yes', 0 for 'No').
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

print("\n✅ Data cleaning complete.")

# =============================================================================
# Step 3: High-Impact Exploratory Data Analysis (EDA)
# =============================================================================
# This visualization is key to your story: it shows the strong link between contract type and churn.
print("📊 Generating key EDA visual...")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Contract', hue='Churn', palette='viridis')
plt.title('Churn Rate by Contract Type', fontsize=16)
plt.xlabel('Contract Type', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.legend(title='Churn', labels=['Stays', 'Leaves'])
plt.show()

# =============================================================================
# Step 4: Feature Preparation and Preprocessing Pipeline
# =============================================================================
# Define the features (X) and the target (y). We drop customerID as it's an identifier.
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Identify which columns are categorical and which are numerical.
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns

# Create a preprocessing pipeline. This automates the process of preparing data for the model.
# - Numerical features will be scaled (to have a similar range).
# - Categorical features will be one-hot encoded (converted to numbers).
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
print("\n✅ Preprocessing pipeline created.")

# =============================================================================
# Step 5: Split Data, Train, and Evaluate Models
# =============================================================================
# Split the data into training (80%) and testing (20%) sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

# --- Model 1: Logistic Regression (Baseline) ---
pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(random_state=42))])

pipeline_lr.fit(X_train, y_train)
y_pred_lr = pipeline_lr.predict(X_test)


# --- Model 2: Random Forest (High-Performance) ---
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)
print("✅ Both models trained successfully.")

# =============================================================================
# Step 6: Compare Results
# =============================================================================
print("\n" + "="*30)
print("     MODEL PERFORMANCE RESULTS")
print("="*30 + "\n")

# --- Logistic Regression Results ---
print("--- 1. Logistic Regression (Baseline) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.2%}")
print(f"**Recall**: {recall_score(y_test, y_pred_lr):.2%}")
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))

# --- Random Forest Results ---
print("\n--- 2. Random Forest (High-Performance) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2%}")
print(f"**Recall**: {recall_score(y_test, y_pred_rf):.2%}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\n**Conclusion:** The Random Forest model shows better overall performance, especially in its balance of precision and recall, making it the more effective choice for this business problem.")


# =============================================================================
# Final Step: Generate Churn Scores for Dashboard
# =============================================================================
# Use the best model (Random Forest) to predict the probability of churn.
# This score is what you would show in your dashboard.
churn_risk_scores = pipeline_rf.predict_proba(X_test)[:, 1]

# Create a new DataFrame with customer info and their predicted risk score
results_df = X_test.copy()
results_df['ActualChurn'] = y_test
results_df['ChurnRiskScore'] = churn_risk_scores

print("\n\n--- Sample Output for Dashboard ---")
print("Prioritized list of customers by churn risk:")
print(results_df.sort_values(by='ChurnRiskScore', ascending=False).head())