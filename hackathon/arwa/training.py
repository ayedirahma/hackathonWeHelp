import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\hp\Desktop\hackathon\train_dataset.csv")  
type(df)
df.head()
df.info()
df.describe()
df.isnull().sum()

df = df.drop_duplicates()
df = df.dropna() 
df = pd.get_dummies(df, drop_first=True)



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

# Check missing values
missing_summary = X_train.isnull().mean().sort_values(ascending=False)
display(missing_summary[missing_summary > 0])

# Simple strategy: fill missing values with median (robust for finance data)
X_train_filled = X_train.fillna(X_train.median())
X_test_filled = X_test.fillna(X_train.median())  # use train statistics only

print("Remaining missing values (train):", X_train_filled.isnull().sum().sum())
print("Remaining missing values (test):", X_test_filled.isnull().sum().sum())

log_reg = LogisticRegression(
    max_iter=1000,
    random_state = RANDOM_STATE,
    n_jobs =-1,
)

log_reg.fit(X_train_filled, y_train)