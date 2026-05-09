from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def train_model(X_train, y_train):
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    baseline = (y_test.shift(1) == y_test).mean()
    return acc, baseline, y_pred

def predict_latest(model, X):
    latest = X.iloc[-1:]
    pred = model.predict(latest)[0]
    proba = model.predict_proba(latest)[0]
    confidence = proba[1] if pred == 1 else proba[0]
    return pred, proba, confidence

def feature_importance(model, features):
    importance = model.feature_importances_
    return pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
