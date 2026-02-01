import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\hp\Desktop\hackathon\train_dataset.csv")  
type(df)
df.head()
df.info()
df.describe()
df.isnull().sum()

df = df.drop_duplicates()
df = df.dropna() 
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["TARGET", "SK_ID_CURR", "RISK_SCORE"])
y = df["TARGET"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y  # preserve default rate
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)
print("Training default rate:", y_train.mean())
print("Test default rate:", y_test.mean())

# missing_summary = X_train.isnull().mean().sort_values(ascending=False)
missing_summary = X_train.isnull().mean().sort_values(ascending=False)
print(missing_summary[missing_summary > 0])


# Simple strategy: fill missing values with median (robust for finance data)
X_train_filled = X_train.fillna(X_train.median())
X_test_filled = X_test.fillna(X_train.median())  # use train statistics only

print("Remaining missing values (train):", X_train_filled.isnull().sum().sum())
print("Remaining missing values (test):", X_test_filled.isnull().sum().sum())

log_reg = LogisticRegression(
    max_iter=10000,
    random_state = 42,
    n_jobs =-1,
)

log_reg.fit(X_train_filled, y_train)
proba_test_lr = log_reg.predict_proba(X_test_filled)[:, 1]

scaler = StandardScaler()

# Fit scaler on training data only
X_train_scaled = scaler.fit_transform(X_train_filled)
X_test_scaled = scaler.transform(X_test_filled)

# Re-train Logistic Regression on scaled data
log_reg_scaled = LogisticRegression(
    max_iter=20000,
    random_state=42,
    n_jobs=-1
)

log_reg_scaled.fit(X_train_scaled, y_train)

# Predict probabilities again
proba_test_lr = log_reg_scaled.predict_proba(X_test_scaled)[:, 1]

print("First 10 predicted default probabilities (scaled model):")
print(np.round(proba_test_lr[:10], 3))


plt.figure(figsize=(6, 4))
plt.hist(proba_test_lr, bins=50)
plt.xlabel("Predicted probability of default")
plt.ylabel("Number of borrowers")
plt.title("Distribution of Predicted Default Risk")
plt.show()


# Basic summary statistics
print("Probability summary:")
print(pd.Series(proba_test_lr).describe())

coef_df = pd.DataFrame({
    "feature": X.columns,
    "weight": log_reg_scaled.coef_[0]
}).sort_values(by="weight")

print(coef_df)


plt.figure(figsize=(6, 5))
plt.barh(coef_df["feature"], coef_df["weight"])
plt.title("Logistic Regression Weights (Direction of Risk)")
plt.axvline(0)
plt.show()

z_test = log_reg_scaled.decision_function(X_test_scaled)

# Corresponding probabilities (already computed, but shown for clarity)
p_test = proba_test_lr

print("First 10 scores (z):", np.round(z_test[:10], 2))
print("First 10 probabilities:", np.round(p_test[:10], 3))


# Plot fitted sigmoid with real data points

# Create smooth range of scores for the curve
z_curve = np.linspace(z_test.min(), z_test.max(), 500)
p_curve = 1 / (1 + np.exp(-z_curve))

plt.figure(figsize=(7, 5))

# Plot fitted sigmoid
plt.plot(z_curve, p_curve, label="Fitted sigmoid", color="black")

# Overlay a subset of real borrowers
idx = np.random.choice(len(z_test), size=500, replace=False)
plt.scatter(z_test[idx], p_test[idx], 
            c=y_test.iloc[idx], cmap="coolwarm", 
            alpha=0.5, s=20, label="Borrowers")
plt.xlabel("Model score (w·x + b)")
plt.ylabel("Probability of default")
plt.title("Our Logistic Model: From Score to Probability")
plt.legend()
plt.show()

t_approve = 0.08
t_reject = 0.25

def policy(p):
    if p<t_approve:
        return "Approve"
    elif p<t_reject:
        return "Review"
    else:
        return "Reject"
    
decisions = pd.Series(proba_test_lr).apply(policy)

print("Decision distribution:")
pd.Series(decisions).value_counts(normalize=True)


print("Decision counts:")
print(decisions.value_counts())
print("\nDecision proportions:")
print(decisions.value_counts(normalize=True))

def logit(p):
    return np.log(p / (1 - p))

z_approve = logit(t_approve)
z_reject  = logit(t_reject)

plt.figure(figsize=(7, 5))

# Fitted sigmoid curve
z_curve = np.linspace(z_test.min(), z_test.max(), 500)
p_curve = 1 / (1 + np.exp(-z_curve))
plt.plot(z_curve, p_curve, color="black", label="Fitted sigmoid")

# Threshold lines (horizontal)
plt.axhline(t_approve, linestyle="--", alpha=0.8, label=f"Approve threshold p={t_approve}")
plt.axhline(t_reject, linestyle="--", alpha=0.8, label=f"Reject threshold p={t_reject}")

# Optional: vertical lines at corresponding scores
plt.axvline(z_approve, linestyle=":", alpha=0.8)
plt.axvline(z_reject, linestyle=":", alpha=0.8)
plt.xlabel("Model score (w·x + b)")
plt.ylabel("Probability of default")
plt.title("Decision Policy Boundaries on Our Logistic Model")
plt.ylim(-0.02, 1.02)
plt.legend()
plt.show()

print("Score threshold for approve (z):", round(z_approve, 3))
print("Score threshold for reject  (z):", round(z_reject, 3))

COST_BAD_APPROVE = -10000   # approve & default
COST_BAD_REJECT  = -1000    # reject & no default
COST_REVIEW      = -200     # manual review (regardless of outcome)
COST_GOOD_APPROVE = 500     # approve & no default (profit)

# Evaluate policy outcomes on the test set


results = pd.DataFrame({
    "prob_default": proba_test_lr,
    "decision": decisions,
    "actual_default": y_test.values
})

def compute_cost(row):
    if row["decision"] == "Approve":
        if row["actual_default"] == 1:
            return COST_BAD_APPROVE
        else:
            return COST_GOOD_APPROVE
    elif row["decision"] == "Reject":
        if row["actual_default"] == 1:
            return 0  
        else:
            return COST_BAD_REJECT
    else:  
        return COST_REVIEW

results["cost"] = results.apply(compute_cost, axis=1)


# Summary metrics


total_cost = results["cost"].sum()
avg_cost = results["cost"].mean()

summary = results.groupby("decision")["cost"].agg(["count", "mean", "sum"])

print("Total business outcome:", total_cost)
print("Average outcome per borrower:", round(avg_cost, 2))
print(summary)