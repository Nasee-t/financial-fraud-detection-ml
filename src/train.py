from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=300,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_depth=10
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    neg, pos = y_train.value_counts()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg/pos,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
        )

    model.fit(X_train, y_train)
    return model
