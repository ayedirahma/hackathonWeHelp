import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ========================================
# CONFIGURATION
# ========================================
N_SAMPLES = 50000  # Number of loan applications
DEFAULT_RATE = 0.08  # 8% default rate (realistic for credit risk)

print(f"Generating {N_SAMPLES} loan applications with alternative financial indicators...")

# ========================================
# HELPER FUNCTIONS
# ========================================

def generate_correlated_feature(base_feature, correlation, noise_scale=0.3):
    """Generate a feature correlated with base feature"""
    noise = np.random.normal(0, noise_scale, len(base_feature))
    return correlation * base_feature + (1 - correlation) * noise

def clip_to_range(arr, min_val, max_val):
    """Clip array to specified range"""
    return np.clip(arr, min_val, max_val)

def assign_risk_based_target(risk_score, base_default_rate=0.08):
    """Assign default target based on risk score with some randomness"""
    # Higher risk score = higher default probability
    default_prob = base_default_rate * (1 + 2 * risk_score)
    default_prob = np.clip(default_prob, 0.01, 0.5)
    return (np.random.random(len(risk_score)) < default_prob).astype(int)

# ========================================
# 1. GENERATE BASE DEMOGRAPHICS & TRADITIONAL FEATURES
# ========================================

data = pd.DataFrame()

# Primary Key
data['SK_ID_CURR'] = range(100000, 100000 + N_SAMPLES)

# Demographics
data['CODE_GENDER'] = np.random.choice(['M', 'F'], N_SAMPLES, p=[0.48, 0.52])
data['AGE'] = np.random.normal(40, 12, N_SAMPLES).clip(21, 70).astype(int)
data['DAYS_BIRTH'] = -data['AGE'] * 365

# Income (log-normal distribution)
income_base = np.random.lognormal(11.5, 0.6, N_SAMPLES)
data['AMT_INCOME_TOTAL'] = (income_base * 1000).clip(25000, 1000000).round(0)

# Employment
data['DAYS_EMPLOYED'] = -np.random.exponential(2000, N_SAMPLES).clip(0, 15000).astype(int)
data['OCCUPATION_TYPE'] = np.random.choice(
    ['Laborers', 'Sales staff', 'Core staff', 'Managers', 'Drivers', 
     'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff',
     'Cooking staff', 'Cleaning staff', 'Private service staff', 'Low-skill Laborers',
     'Waiters/barmen staff', 'Secretaries', 'Realty agents', 'HR staff', 'IT staff'],
    N_SAMPLES,
    p=[0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 
       0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.03]
)

# Loan Amount & Credit
data['AMT_CREDIT'] = np.random.uniform(0.5, 3.0, N_SAMPLES) * data['AMT_INCOME_TOTAL']
data['AMT_CREDIT'] = data['AMT_CREDIT'].clip(20000, 2000000).round(0)
data['AMT_ANNUITY'] = (data['AMT_CREDIT'] / np.random.uniform(12, 60, N_SAMPLES)).round(0)

# Family & Housing
data['CNT_CHILDREN'] = np.random.choice([0, 1, 2, 3, 4, 5], N_SAMPLES, p=[0.45, 0.25, 0.18, 0.08, 0.03, 0.01])
data['CNT_FAM_MEMBERS'] = data['CNT_CHILDREN'] + np.random.choice([1, 2], N_SAMPLES, p=[0.3, 0.7])
data['NAME_HOUSING_TYPE'] = np.random.choice(
    ['House / apartment', 'Rented apartment', 'With parents', 'Municipal apartment', 'Office apartment', 'Co-op apartment'],
    N_SAMPLES,
    p=[0.50, 0.25, 0.15, 0.05, 0.03, 0.02]
)

# Education
data['NAME_EDUCATION_TYPE'] = np.random.choice(
    ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'],
    N_SAMPLES,
    p=[0.50, 0.30, 0.12, 0.06, 0.02]
)

# ========================================
# 2. BEHAVIORAL & TRANSACTIONAL INDICATORS
# ========================================

print("Generating behavioral & transactional indicators...")

# Create customer segments for realistic correlations
# Segment 0: Stable salaried (40%)
# Segment 1: Gig workers (25%)
# Segment 2: Freelancers (20%)
# Segment 3: Unstable/risky (15%)

customer_segment = np.random.choice([0, 1, 2, 3], N_SAMPLES, p=[0.40, 0.25, 0.20, 0.15])

# Transaction Frequency & Stability
# Stable workers: fewer, regular transactions
# Gig workers: many small transactions
# Freelancers: moderate, variable transactions
# Unstable: erratic patterns

avg_transactions_by_segment = {0: 15, 1: 35, 2: 25, 3: 20}
data['avg_weekly_transactions'] = [
    np.random.normal(avg_transactions_by_segment[seg], 5, 1)[0] 
    for seg in customer_segment
]
data['avg_weekly_transactions'] = data['avg_weekly_transactions'].clip(2, 70).round(1)

# Transaction stability (lower std = more stable)
std_by_segment = {0: 3, 1: 8, 2: 6, 3: 12}
data['transaction_frequency_std_dev'] = [
    np.random.exponential(std_by_segment[seg], 1)[0] 
    for seg in customer_segment
]
data['transaction_frequency_std_dev'] = data['transaction_frequency_std_dev'].clip(0.5, 25).round(2)

# Merchant category entropy (diversity of spending)
# Higher entropy = more diverse spending categories
entropy_by_segment = {0: 0.6, 1: 0.8, 2: 0.7, 3: 0.5}
data['entropy_merchant_categories'] = [
    np.random.beta(entropy_by_segment[seg] * 5, (1 - entropy_by_segment[seg]) * 5, 1)[0] 
    for seg in customer_segment
]
data['entropy_merchant_categories'] = data['entropy_merchant_categories'].round(3)

# Income Regularity Score (0-1, higher = more regular)
regularity_by_segment = {0: 0.85, 1: 0.45, 2: 0.55, 3: 0.30}
data['income_regularity_score'] = [
    np.random.beta(regularity_by_segment[seg] * 10, (1 - regularity_by_segment[seg]) * 10, 1)[0] 
    for seg in customer_segment
]
data['income_regularity_score'] = data['income_regularity_score'].clip(0.05, 0.98).round(3)

# Primary Income Frequency
frequency_mapping = {
    0: ['monthly', 'bi-weekly'],
    1: ['irregular', 'weekly'],
    2: ['monthly', 'irregular', 'bi-weekly'],
    3: ['irregular', 'weekly']
}
data['primary_income_frequency'] = [
    np.random.choice(frequency_mapping[seg]) 
    for seg in customer_segment
]

# Months with Income (last 6 months)
months_income_by_segment = {0: 6, 1: 4, 2: 5, 3: 3}
data['months_with_income_last_6'] = [
    min(6, max(0, int(np.random.normal(months_income_by_segment[seg], 1)))) 
    for seg in customer_segment
]

# Payment Delays & Cash Flow Timing
# Days until paycheck when bill is due (negative = no money, positive = buffer)
days_buffer_by_segment = {0: 10, 1: -3, 2: 5, 3: -8}
data['days_until_paycheck_at_bill_due'] = [
    np.random.normal(days_buffer_by_segment[seg], 7, 1)[0] 
    for seg in customer_segment
]
data['days_until_paycheck_at_bill_due'] = data['days_until_paycheck_at_bill_due'].clip(-30, 30).round(1)

# Average time to clear checks (days)
data['avg_time_to_clear_checks'] = np.random.gamma(2, 1.5, N_SAMPLES).clip(0.5, 10).round(1)

# Recurring payment on-time rate
ontime_by_segment = {0: 0.92, 1: 0.75, 2: 0.83, 3: 0.55}
data['recurring_payment_ontime_rate'] = [
    np.random.beta(ontime_by_segment[seg] * 20, (1 - ontime_by_segment[seg]) * 20, 1)[0] 
    for seg in customer_segment
]
data['recurring_payment_ontime_rate'] = data['recurring_payment_ontime_rate'].clip(0.1, 1.0).round(3)

# ========================================
# 3. DIGITAL & LIFESTYLE INDICATORS
# ========================================

print("Generating digital & lifestyle indicators...")

# Mobile Wallet Active Months
data['mobile_wallet_active_months'] = np.random.choice(
    [0, 3, 6, 12, 18, 24, 36, 48], 
    N_SAMPLES, 
    p=[0.15, 0.10, 0.15, 0.20, 0.15, 0.12, 0.08, 0.05]
)

# Average mobile wallet balance
wallet_balance_base = data['AMT_INCOME_TOTAL'] / 12 * np.random.uniform(0.05, 0.30, N_SAMPLES)
data['avg_mobile_wallet_balance'] = wallet_balance_base.clip(0, 50000).round(0)
data.loc[data['mobile_wallet_active_months'] == 0, 'avg_mobile_wallet_balance'] = 0

# P2P Transaction Ratio
p2p_by_segment = {0: 0.15, 1: 0.35, 2: 0.40, 3: 0.25}
data['P2P_transaction_ratio'] = [
    np.random.beta(p2p_by_segment[seg] * 5, (1 - p2p_by_segment[seg]) * 5, 1)[0] 
    for seg in customer_segment
]
data['P2P_transaction_ratio'] = data['P2P_transaction_ratio'].clip(0, 0.8).round(3)

# Utility Bill On-Time Rate
utility_ontime_by_segment = {0: 0.95, 1: 0.80, 2: 0.88, 3: 0.60}
data['utility_bill_ontime_rate'] = [
    np.random.beta(utility_ontime_by_segment[seg] * 20, (1 - utility_ontime_by_segment[seg]) * 20, 1)[0] 
    for seg in customer_segment
]
data['utility_bill_ontime_rate'] = data['utility_bill_ontime_rate'].clip(0.2, 1.0).round(3)

# Subscription Cancellation Rate (lower = more stable)
data['subscription_cancellation_rate'] = np.random.beta(2, 8, N_SAMPLES).clip(0, 0.6).round(3)

# Consistent Internet Payment (Boolean)
internet_payment_prob = {0: 0.92, 1: 0.70, 2: 0.85, 3: 0.50}
data['has_consistent_internet_payment'] = [
    np.random.random() < internet_payment_prob[seg] 
    for seg in customer_segment
]
data['has_consistent_internet_payment'] = data['has_consistent_internet_payment'].astype(int)

# ========================================
# 4. ENHANCED FINANCIAL BEHAVIOR PATTERNS
# ========================================

print("Generating enhanced financial behavior patterns...")

# Savings Rate (% of income)
savings_by_segment = {0: 0.15, 1: 0.05, 2: 0.12, 3: 0.02}
data['savings_rate'] = [
    np.random.beta(savings_by_segment[seg] * 10, (1 - savings_by_segment[seg]) * 10, 1)[0] 
    for seg in customer_segment
]
data['savings_rate'] = data['savings_rate'].clip(0, 0.50).round(3)

# Has Automatic Savings Transfer
auto_savings_prob = {0: 0.60, 1: 0.20, 2: 0.40, 3: 0.10}
data['has_automatic_savings_transfer'] = [
    np.random.random() < auto_savings_prob[seg] 
    for seg in customer_segment
]
data['has_automatic_savings_transfer'] = data['has_automatic_savings_transfer'].astype(int)

# Savings Balance Growth Trend (monthly % change)
growth_by_segment = {0: 0.02, 1: -0.01, 2: 0.01, 3: -0.03}
data['savings_balance_growth_trend'] = [
    np.random.normal(growth_by_segment[seg], 0.03, 1)[0] 
    for seg in customer_segment
]
data['savings_balance_growth_trend'] = data['savings_balance_growth_trend'].clip(-0.15, 0.15).round(4)

# Cash Flow Volatility Index (higher = more volatile)
volatility_by_segment = {0: 0.20, 1: 0.65, 2: 0.45, 3: 0.85}
data['cash_flow_volatility_index'] = [
    np.random.gamma(volatility_by_segment[seg] * 5, 0.1, 1)[0] 
    for seg in customer_segment
]
data['cash_flow_volatility_index'] = data['cash_flow_volatility_index'].clip(0.05, 2.0).round(3)

# Average End-of-Month Balance (as % of monthly income)
eom_balance_by_segment = {0: 0.25, 1: 0.08, 2: 0.15, 3: 0.03}
data['avg_end_of_month_balance'] = [
    np.random.beta(eom_balance_by_segment[seg] * 10, (1 - eom_balance_by_segment[seg]) * 10, 1)[0] 
    for seg in customer_segment
]
data['avg_end_of_month_balance'] = data['avg_end_of_month_balance'].clip(0, 0.60).round(3)

# Dynamic DTI Last 6 Months (slope: negative = improving, positive = worsening)
dti_trend_by_segment = {0: -0.01, 1: 0.03, 2: 0.01, 3: 0.08}
data['dynamic_dti_last_6m'] = [
    np.random.normal(dti_trend_by_segment[seg], 0.03, 1)[0] 
    for seg in customer_segment
]
data['dynamic_dti_last_6m'] = data['dynamic_dti_last_6m'].clip(-0.15, 0.25).round(4)

# Income vs Expenses Trend (positive = improving surplus)
income_expense_trend_by_segment = {0: 0.02, 1: -0.02, 2: 0.00, 3: -0.05}
data['income_vs_expenses_trend'] = [
    np.random.normal(income_expense_trend_by_segment[seg], 0.03, 1)[0] 
    for seg in customer_segment
]
data['income_vs_expenses_trend'] = data['income_vs_expenses_trend'].clip(-0.20, 0.20).round(4)

# ========================================
# 5. CALCULATE COMPOSITE RISK SCORE & TARGET
# ========================================

print("Calculating risk scores and target variable...")

# Normalize features to 0-1 for risk scoring (negative indicators)
risk_components = pd.DataFrame()

risk_components['income_irregularity'] = 1 - data['income_regularity_score']
risk_components['transaction_instability'] = (data['transaction_frequency_std_dev'] / 25).clip(0, 1)
risk_components['payment_delays'] = 1 - data['recurring_payment_ontime_rate']
risk_components['utility_delays'] = 1 - data['utility_bill_ontime_rate']
risk_components['low_savings'] = 1 - data['savings_rate']
risk_components['high_volatility'] = (data['cash_flow_volatility_index'] / 2).clip(0, 1)
risk_components['low_eom_balance'] = 1 - data['avg_end_of_month_balance']
risk_components['worsening_dti'] = ((data['dynamic_dti_last_6m'] + 0.15) / 0.40).clip(0, 1)
risk_components['negative_trend'] = ((0.20 - data['income_vs_expenses_trend']) / 0.40).clip(0, 1)
risk_components['few_income_months'] = (6 - data['months_with_income_last_6']) / 6
risk_components['low_wallet_usage'] = 1 - (data['mobile_wallet_active_months'] / 48).clip(0, 1)

# Weighted composite risk score
weights = {
    'income_irregularity': 0.15,
    'transaction_instability': 0.08,
    'payment_delays': 0.12,
    'utility_delays': 0.10,
    'low_savings': 0.08,
    'high_volatility': 0.12,
    'low_eom_balance': 0.10,
    'worsening_dti': 0.15,
    'negative_trend': 0.05,
    'few_income_months': 0.03,
    'low_wallet_usage': 0.02
}

data['RISK_SCORE'] = sum(risk_components[col] * weight for col, weight in weights.items())
data['RISK_SCORE'] = data['RISK_SCORE'].clip(0, 1).round(3)

# Generate TARGET based on risk score with realistic default rate
data['TARGET'] = assign_risk_based_target(data['RISK_SCORE'], DEFAULT_RATE)

print(f"\nDefault Rate: {data['TARGET'].mean():.2%}")
print(f"Average Risk Score: {data['RISK_SCORE'].mean():.3f}")

# ========================================
# 6. ADD SOME REALISTIC NOISE & EDGE CASES
# ========================================

# Some customers with missing/NA values (5% missing rate for optional features)
missing_mask = np.random.random(N_SAMPLES) < 0.05
data.loc[missing_mask, 'avg_mobile_wallet_balance'] = np.nan
data.loc[missing_mask, 'mobile_wallet_active_months'] = np.nan

missing_mask2 = np.random.random(N_SAMPLES) < 0.03
data.loc[missing_mask2, 'subscription_cancellation_rate'] = np.nan

# ========================================
# 7. FINAL DATASET STRUCTURE & EXPORT
# ========================================

# Reorder columns logically
column_order = [
    # Identifiers & Target
    'SK_ID_CURR', 'TARGET', 'RISK_SCORE',
    
    # Demographics
    'CODE_GENDER', 'AGE', 'DAYS_BIRTH', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
    'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
    
    # Traditional Financial
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_EMPLOYED',
    
    # Behavioral & Transactional
    'avg_weekly_transactions', 'transaction_frequency_std_dev', 
    'entropy_merchant_categories', 'income_regularity_score',
    'primary_income_frequency', 'months_with_income_last_6',
    'days_until_paycheck_at_bill_due', 'avg_time_to_clear_checks',
    'recurring_payment_ontime_rate',
    
    # Digital & Lifestyle
    'mobile_wallet_active_months', 'avg_mobile_wallet_balance',
    'P2P_transaction_ratio', 'utility_bill_ontime_rate',
    'subscription_cancellation_rate', 'has_consistent_internet_payment',
    
    # Enhanced Financial Behavior
    'savings_rate', 'has_automatic_savings_transfer',
    'savings_balance_growth_trend', 'cash_flow_volatility_index',
    'avg_end_of_month_balance', 'dynamic_dti_last_6m',
    'income_vs_expenses_trend'
]

final_dataset = data[column_order].copy()

# ========================================
# 8. GENERATE SUMMARY STATISTICS
# ========================================

print("\n" + "="*80)
print("DATASET GENERATION COMPLETE")
print("="*80)

print(f"\nDataset Shape: {final_dataset.shape}")
print(f"Total Samples: {len(final_dataset):,}")
print(f"Total Features: {len(final_dataset.columns)}")

print("\n--- Target Distribution ---")
print(final_dataset['TARGET'].value_counts())
print(f"Default Rate: {final_dataset['TARGET'].mean():.2%}")

print("\n--- Customer Segment Distribution ---")
segment_names = {0: 'Stable Salaried', 1: 'Gig Workers', 2: 'Freelancers', 3: 'Unstable/Risky'}
segment_dist = pd.Series(customer_segment).map(segment_names).value_counts()
print(segment_dist)

print("\n--- Risk Score Distribution ---")
print(final_dataset['RISK_SCORE'].describe())

print("\n--- Sample Alternative Indicators Statistics ---")
alt_features = [
    'income_regularity_score', 'avg_weekly_transactions',
    'utility_bill_ontime_rate', 'cash_flow_volatility_index',
    'dynamic_dti_last_6m', 'savings_rate'
]
print(final_dataset[alt_features].describe().round(3))

print("\n--- Missing Values ---")
missing_summary = final_dataset.isnull().sum()
missing_summary = missing_summary[missing_summary > 0]
if len(missing_summary) > 0:
    print(missing_summary)
else:
    print("No missing values in core features")

print("\n--- Income Frequency Distribution ---")
print(final_dataset['primary_income_frequency'].value_counts())

# ========================================
# 9. SAVE TO CSV
# ========================================

output_filename = 'home_credit_alternative_indicators_dataset.csv'
final_dataset.to_csv(output_filename, index=False)
print(f"\n✅ Dataset saved to: {output_filename}")

# ========================================
# 10. CREATE DATA DICTIONARY
# ========================================

data_dictionary = pd.DataFrame({
    'Column_Name': final_dataset.columns,
    'Data_Type': final_dataset.dtypes.values,
    'Description': [
        'Unique client identifier',
        'Target variable (1 = default, 0 = repaid)',
        'Composite risk score (0-1, higher = riskier)',
        'Gender (M/F)',
        'Age in years',
        'Days from birth (negative value)',
        'Number of children',
        'Number of family members',
        'Education level',
        'Housing type',
        'Occupation category',
        'Annual income (currency units)',
        'Credit amount requested',
        'Loan annuity amount',
        'Days employed (negative value)',
        'Average number of transactions per week',
        'Standard deviation of weekly transactions',
        'Entropy of spending across merchant categories (0-1)',
        'Income regularity score (0-1, higher = more regular)',
        'Detected pay cycle (monthly, bi-weekly, weekly, irregular)',
        'Number of months with income in last 6 months (0-6)',
        'Days until paycheck when bill is due (buffer)',
        'Average days to clear checks',
        'Recurring payment on-time rate (0-1)',
        'Months since first mobile wallet usage',
        'Average stored value in mobile wallet',
        'Ratio of P2P transactions to total (0-1)',
        'Utility bill on-time payment rate (0-1)',
        'Subscription cancellation rate (0-1)',
        'Has consistent internet payment (0/1)',
        'Savings rate as % of income (0-1)',
        'Has automatic savings transfer (0/1)',
        'Monthly % change in savings balance',
        'Cash flow volatility index (higher = more volatile)',
        'Average end-of-month balance as % of income (0-1)',
        'DTI trend slope over last 6 months (positive = worsening)',
        'Income vs expenses trend (positive = improving)'
    ],
    'Range': [
        '100000-149999',
        '0 or 1',
        '0.0-1.0',
        'M, F',
        '21-70',
        '-25550 to -7665',
        '0-5',
        '1-7',
        'Categorical',
        'Categorical',
        'Categorical',
        '25000-1000000',
        '20000-2000000',
        'Varies',
        '-15000 to 0',
        '2-70',
        '0.5-25',
        '0.0-1.0',
        '0.05-0.98',
        'Categorical',
        '0-6',
        '-30 to 30',
        '0.5-10',
        '0.1-1.0',
        '0-48',
        '0-50000',
        '0.0-0.8',
        '0.2-1.0',
        '0.0-0.6',
        '0 or 1',
        '0.0-0.5',
        '0 or 1',
        '-0.15 to 0.15',
        '0.05-2.0',
        '0.0-0.6',
        '-0.15 to 0.25',
        '-0.20 to 0.20'
    ]
})

dict_filename = 'data_dictionary.csv'
data_dictionary.to_csv(dict_filename, index=False)
print(f"✅ Data dictionary saved to: {dict_filename}")

# ========================================
# 11. CREATE TRAIN/TEST SPLIT FILES
# ========================================

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(final_dataset, test_size=0.2, random_state=42, stratify=final_dataset['TARGET'])

train_df.to_csv('train_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

print(f"\n✅ Training set saved: train_dataset.csv ({len(train_df):,} samples)")
print(f"✅ Test set saved: test_dataset.csv ({len(test_df):,} samples)")

# ========================================
# 12. FEATURE IMPORTANCE PREVIEW
# ========================================

print("\n--- Correlation with Target (Top 15 Features) ---")
correlations = final_dataset.corr(numeric_only=True)['TARGET'].abs().sort_values(ascending=False)
print(correlations.head(16)[1:])  # Exclude TARGET itself

print("\n" + "="*80)
print("DATASET GENERATION SUMMARY")
print("="*80)
print(f"""
✅ Main Dataset: {output_filename} ({len(final_dataset):,} rows × {len(final_dataset.columns)} columns)
✅ Training Set: train_dataset.csv ({len(train_df):,} rows, {train_df['TARGET'].mean():.2%} default rate)
✅ Test Set: test_dataset.csv ({len(test_df):,} rows, {test_df['TARGET'].mean():.2%} default rate)
✅ Data Dictionary: {dict_filename}

🎯 Key Alternative Indicators Included:
   • Behavioral: avg_weekly_transactions, income_regularity_score, transaction patterns
   • Digital: mobile_wallet metrics, P2P transactions, digital payment consistency
   • Financial: savings_rate, cash_flow_volatility, dynamic_dti_last_6m

📊 Dataset is ready for model training with realistic correlations and edge cases
""")

# ========================================
# 13. QUICK MODEL VALIDATION (OPTIONAL)
# ========================================

print("\n--- Quick Model Validation ---")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Select features for quick validation
feature_cols = [col for col in final_dataset.columns 
                if col not in ['SK_ID_CURR', 'TARGET', 'RISK_SCORE', 
                              'CODE_GENDER', 'NAME_EDUCATION_TYPE', 
                              'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
                              'primary_income_frequency']]

X_train = train_df[feature_cols].fillna(0)
y_train = train_df['TARGET']
X_test = test_df[feature_cols].fillna(0)
y_test = test_df['TARGET']

# Train quick RF model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
y_pred = rf_model.predict(X_test)

auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\n✅ Random Forest AUC Score: {auc_score:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Repaid', 'Default']))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n--- Top 15 Most Important Features ---")
print(feature_importance.head(15).to_string(index=False))

print("\n" + "="*80)
print("✅ ALL DONE! Your dataset is ready for AI model training.")
print("="*80)
