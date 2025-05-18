import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import joblib

def get_stacking_model() -> StackingClassifier:
    """Stacking 모델 반환"""
    base_models = [
        ("lr", LogisticRegression(C=1.0, max_iter=1000)),
        ("rf", RandomForestClassifier(
            n_estimators=600,
            max_depth=20,
            random_state=42)),
        ("lgb", LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.02,
            num_leaves=50,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42))
    ]
    return StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=5,
        n_jobs=-1
    )

if __name__ == "__main__":
    train = pd.read_csv("../train_final_with_selection.csv")
    test = pd.read_csv("../test_final_with_selection.csv")

    X_train = train.drop(columns=["id", "y"], errors="ignore")
    y_train = train["y"]
    X_test = test.drop(columns=["id"], errors="ignore")
    test_id = test.get("id", pd.Series(np.arange(len(test)), name="id"))

    model = get_stacking_model()
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    pd.DataFrame({
        "id": test_id,
        "y_prob": y_prob,
        "y_predict": y_pred
    }).to_csv("../prediction_stacking.csv", index=False)

    joblib.dump(model, "../stacking_model.pkl")
    print("prediction_stacking.csv, stacking_model.pkl 저장 완료")