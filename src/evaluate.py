from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print("Threshold:", threshold)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred)) 

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
