"""Stratified (balanced) split of the BAF Base dataset.

Why: the temporal split (months 0-5 / 6 / 7) leaves only ~1,400 fraud cases in
the test month and the fraud counts differ from month to month. Here every
split gets the SAME fraud rate (about 1.1%) by splitting each class
separately, so the fraud cases are shared out equally between train,
validation and test.

Outputs go to processed_balanced/ with the SAME file names and columns as
preprocess.py (the "month" column is kept so the training scripts, which
drop it, keep working). Point the training scripts at this folder, or copy
the files into processed/ after backing up the temporal split.

Run from the project root:
    python preprocess_balanced.py
    python preprocess_balanced.py --train 0.70 --val 0.15 --test 0.15 --seed 42
"""
import argparse
import os

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=float, default=0.70)
parser.add_argument("--val", type=float, default=0.15)
parser.add_argument("--test", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="processed_balanced")
args = parser.parse_args()

if abs(args.train + args.val + args.test - 1.0) > 1e-9:
    raise ValueError("train + val + test fractions must add up to 1.")

TARGET = "fraud_bool"

print("=" * 55)
print("FRAUD DETECTION - STRATIFIED (BALANCED) PREPROCESSING")
print("=" * 55)

df = pd.read_csv("data/Base.csv", low_memory=False)
print("Dataset shape:", df.shape)
df = df.drop_duplicates().copy()

# Two-step stratified split: first carve out test, then split the rest into
# train and validation. stratify= keeps the fraud rate identical everywhere.
rest_df, test_df = train_test_split(
    df, test_size=args.test, stratify=df[TARGET], random_state=args.seed
)
val_share = args.val / (args.train + args.val)
train_df, val_df = train_test_split(
    rest_df, test_size=val_share, stratify=rest_df[TARGET], random_state=args.seed
)

for name, part in [("train", train_df), ("validation", val_df), ("test", test_df)]:
    print(f"{name:<11} rows={len(part):>8,}  fraud={int(part[TARGET].sum()):>6,}  "
          f"fraud rate={part[TARGET].mean() * 100:.3f}%")

# Encode and align columns to the training columns (same as preprocess.py).
def split_xy(part):
    return part.drop(columns=[TARGET]), part[TARGET]

X_train_raw, y_train = split_xy(train_df)
X_val_raw, y_val = split_xy(val_df)
X_test_raw, y_test = split_xy(test_df)

X_train = pd.get_dummies(X_train_raw, drop_first=True, dtype=int)
X_val = pd.get_dummies(X_val_raw, drop_first=True, dtype=int).reindex(
    columns=X_train.columns, fill_value=0)
X_test = pd.get_dummies(X_test_raw, drop_first=True, dtype=int).reindex(
    columns=X_train.columns, fill_value=0)

# Impute with training medians only.
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_val = pd.DataFrame(imputer.transform(X_val), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns)

os.makedirs(args.out, exist_ok=True)
for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val),
                   ("test", X_test, y_test)]:
    X.to_csv(f"{args.out}/X_{name}.csv", index=False)
    y.reset_index(drop=True).to_frame().to_csv(f"{args.out}/y_{name}.csv", index=False)

print(f"\nSaved to {args.out}/  (X_*.csv and y_*.csv)")
print("NOTE: the split is random, not chronological, so results are NOT")
print("comparable to the month-7 results and cannot show drift over time.")
print("=" * 55)