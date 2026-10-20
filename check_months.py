
import pandas as pd

for split in ["train", "test"]:
    X = pd.read_csv(
        f"processed/X_{split}.csv",
        usecols=["month"]
    )
    y = pd.read_csv(f"processed/y_{split}.csv")

    data = X.copy()
    data["fraud_bool"] = y["fraud_bool"]

    print(f"\n{'=' * 40}")
    print(f"{split.upper()} MONTH DISTRIBUTION")
    print("=" * 40)

    print(
        data.groupby("month")["fraud_bool"]
        .agg(["count", "sum", "mean"])
        .rename(columns={
            "count": "applications",
            "sum": "fraud_cases",
            "mean": "fraud_rate"
        })
    )
