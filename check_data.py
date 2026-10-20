
import pandas as pd

# Check feature columns without loading the entire large files
for filename in ["X_train.csv", "X_test.csv"]:
    df = pd.read_csv("processed/" + filename, nrows=5)
    print(f"\n{filename}")
    print("Number of feature columns:", len(df.columns))
    print("Feature names:", df.columns.tolist())
    print(df.head(2))

# Check the target labels
for filename in ["y_train.csv", "y_test.csv"]:
    df = pd.read_csv("processed/" + filename)
    print(f"\n{filename}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Label counts:")
    print(df.iloc[:, 0].value_counts(dropna=False))
