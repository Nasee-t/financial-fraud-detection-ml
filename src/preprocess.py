import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(path):
    df = pd.read_csv(path, nrows=300_000)

    # Balance inconsistency (very common in fraud)
    df["balance_inconsistency"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]

    df["dest_balance_inconsistency"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]

    # Transaction size relative to balance
    df["amount_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    df = pd.get_dummies(df, columns=["type"], drop_first=True)

    features = ["amount", "oldbalanceOrg", "newbalanceOrig", "balance_inconsistency", "dest_balance_inconsistency", "amount_ratio"
                ] + [col for col in df.columns if col.startswith("type_")]
    
    X = df[features]
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    numeric_cols = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "balance_inconsistency",
        "amount_ratio"
    ]

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test
