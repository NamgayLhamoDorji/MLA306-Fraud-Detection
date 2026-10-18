
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

print("=" * 50)
print("FRAUD DETECTION - DATA PREPROCESSING")
print("=" * 50)

# 1. Load the dataset
file_path = "data/Base.csv"

print("\n1. Loading dataset...")

df = pd.read_csv(file_path, low_memory=False)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Display basic information
print("\n2. Dataset information")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# 3. Check missing values
print("\n3. Checking missing values")

missing_values = df.isnull().sum()
print("Total missing values:", missing_values.sum())

if missing_values.sum() > 0:
    print(missing_values[missing_values > 0])
else:
    print("No missing values found.")

# 4. Check duplicate rows
print("\n4. Checking duplicate rows")

duplicates = df.duplicated().sum()
print("Duplicate rows:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")
else:
    print("No duplicate rows found.")

# 5. Check the target variable
print("\n5. Checking fraud distribution")

if "fraud_bool" not in df.columns:
    raise ValueError("Target column 'fraud_bool' was not found.")

print(df["fraud_bool"].value_counts())

print("\nFraud percentages:")
print((df["fraud_bool"].value_counts(normalize=True) * 100).round(2))

# 6. Separate features and target
print("\n6. Separating features and target")

X = df.drop(columns=["fraud_bool"])
y = df["fraud_bool"]

# Convert categorical features into numerical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

print("Number of features after encoding:", X.shape[1])

# 7. Handle missing values
print("\n7. Handling missing values")

imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

print("Missing values handled.")

# 8. Split the dataset
print("\n8. Splitting the dataset")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 9. Save the processed data
print("\n9. Saving processed datasets")

os.makedirs("processed", exist_ok=True)

X_train.to_csv("processed/X_train.csv", index=False)
X_test.to_csv("processed/X_test.csv", index=False)
y_train.to_csv("processed/y_train.csv", index=False)
y_test.to_csv("processed/y_test.csv", index=False)

print("Saved files:")
print("- processed/X_train.csv")
print("- processed/X_test.csv")
print("- processed/y_train.csv")
print("- processed/y_test.csv")

print("\n" + "=" * 50)
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 50)
