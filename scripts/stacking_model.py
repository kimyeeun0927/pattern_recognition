import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import joblib  # 모델 저장용

# 데이터 불러오기
train = pd.read_csv("../data/final_train.csv")
test = pd.read_csv("../data/final_test.csv")

# 안전하게 id, y 컬럼 제거
drop_cols_train = [col for col in ["id", "y"] if col in train.columns]
drop_cols_test = [col for col in ["id"] if col in test.columns]

X_train = train.drop(columns=drop_cols_train)
y_train = train["y"]
X_test = test.drop(columns=drop_cols_test)

# id 컬럼이 없으면 0~n-1로 생성하고 이름 지정
if "id" in test.columns:
    test_id = test["id"]
else:
    test_id = pd.Series(np.arange(len(test)), name="id")

# Base 모델 정의
base_models = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('lgb', LGBMClassifier(n_estimators=100, random_state=42))
]

# Stacking 모델 구성
stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=StratifiedKFold(n_splits=5),
    n_jobs=-1
)

# 학습
stack_model.fit(X_train, y_train)

# 예측
y_prob = stack_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# 결과 저장
submission = pd.DataFrame({
    "id": test_id,
    "y_prob": y_prob,
    "y_predict": y_pred
})
submission.to_csv("../prediction_stack.csv", index=False)

# 모델 저장
joblib.dump(stack_model, "../stacking_model.pkl")

print("prediction_stack.csv 저장 완료")
print("stacking_model.pkl 저장 완료")