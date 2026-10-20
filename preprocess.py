
import pandas as pd
import os

from sklearn.impute import SimpleImputer

print("=" * 55)
print("FRAUD DETECTION - CHRONOLOGICAL PREPROCESSING")
print("=" * 55)

# 1. Load the original dataset
file_path = "data/Base.csv"

print("\n1. Loading dataset...")
df = pd.read_csv(file_path, low_memory=False)

print("Dataset shape:", df.shape)

# 2. Check required columns
if "fraud_bool" not in df.columns:
    raise ValueError("Target column 'fraud_bool' was not found.")

if "month" not in df.columns:
    raise ValueError("Month column was not found.")

# 3. Check missing values and duplicates
print("\n2. Checking data quality")

print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().copy()

# 4. Check the target
print("\n3. Fraud distribution")
print(df["fraud_bool"].value_counts())
print(
    (df["fraud_bool"].value_counts(normalize=True) * 100).round(2)
)

# 5. Split rows chronologically BEFORE preprocessing
print("\n4. Splitting by month")

train_df = df[df["month"].between(0, 5)].copy()
val_df = df[df["month"] == 6].copy()
test_df = df[df["month"] == 7].copy()

if train_df.empty or val_df.empty or test_df.empty:
    raise ValueError(
        "One or more splits are empty. Check the values in the month column."
    )

print("Training months:", sorted(train_df["month"].unique()))
print("Validation months:", sorted(val_df["month"].unique()))
print("Testing months:", sorted(test_df["month"].unique()))

# 6. Separate features and target
target = "fraud_bool"

X_train_raw = train_df.drop(columns=[target])
y_train = train_df[target]

X_val_raw = val_df.drop(columns=[target])
y_val = val_df[target]

X_test_raw = test_df.drop(columns=[target])
y_test = test_df[target]

# 7. Encode categorical features consistently
print("\n5. Encoding categorical features")

X_train = pd.get_dummies(
    X_train_raw, drop_first=True, dtype=int
)
X_val = pd.get_dummies(
    X_val_raw, drop_first=True, dtype=int
)
X_test = pd.get_dummies(
    X_test_raw, drop_first=True, dtype=int
)

# Align validation and test columns with training columns
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("Number of features:", X_train.shape[1])

# 8. Handle missing values
# Fit the imputer ONLY on training features
print("\n6. Handling missing values")

imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns
)

X_val = pd.DataFrame(
    imputer.transform(X_val),
    columns=X_train.columns
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_train.columns
)

print("Missing values handled using training data statistics.")

# 9. Save all datasets
print("\n7. Saving processed datasets")

os.makedirs("processed", exist_ok=True)

X_train.to_csv("processed/X_train.csv", index=False)
y_train.to_frame().to_csv("processed/y_train.csv", index=False)

X_val.to_csv("processed/X_val.csv", index=False)
y_val.to_frame().to_csv("processed/y_val.csv", index=False)

X_test.to_csv("processed/X_test.csv", index=False)
y_test.to_frame().to_csv("processed/y_test.csv", index=False)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))

print("Training fraud cases:", int(y_train.sum()))
print("Validation fraud cases:", int(y_val.sum()))
print("Testing fraud cases:", int(y_test.sum()))

print("\nSaved files:")
print("processed/X_train.csv")
print("processed/y_train.csv")
print("processed/X_val.csv")
print("processed/y_val.csv")
print("processed/X_test.csv")
print("processed/y_test.csv")

print("\nPREPROCESSING COMPLETED!")
print("=" * 55)
