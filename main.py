from src.preprocess import load_and_preprocess_data
from src.train import train_logistic_regression, train_random_forest, train_xgboost
from src.evaluate import evaluate_model

DATA_PATH = "data/onlinefraud.csv"

X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)

print("\n=== Logistic Regression ===")
lr_model = train_logistic_regression(X_train, y_train)
evaluate_model(lr_model, X_test, y_test, threshold=0.7)

print("\n=== Random Forest ===")
rf_model = train_random_forest(X_train, y_train)
evaluate_model(rf_model, X_test, y_test, threshold=0.5)

print("\n=== XGBoost ===")
xgb_model = train_xgboost(X_train, y_train)
evaluate_model(xgb_model, X_test, y_test, threshold=0.2)
