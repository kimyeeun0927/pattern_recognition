import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import joblib  # 모델 저장용

# 학습/테스트 데이터 불러오기
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

# LightGBM 모델 학습
model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# 예측
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# 결과 저장
submission = pd.DataFrame({
    "id": test_id,
    "y_prob": y_prob,
    "y_predict": y_pred
})
submission.to_csv("../prediction_lgb.csv", index=False)

# 모델 저장
joblib.dump(model, "../lightgbm_model.pkl")

print("prediction_lgb.csv 저장 완료")
print("lightgbm_model.pkl 저장 완료")