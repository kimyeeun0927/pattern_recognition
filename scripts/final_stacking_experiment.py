import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib  # 모델 저장용

# 튜닝된 stacking model 가져오기
from final_stacking_model import get_stacking_model

# 1. 데이터 불러오기
train = pd.read_csv("../train_final_with_selection.csv")
X = train.drop(columns=["y"])
y = train["y"]

# 2. 모델 정의
models = {
    "Stacking": get_stacking_model()
}

# 3. 학습/검증 분할
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 결과 저장용 딕셔너리
val_results = {}
cv_results = {}

# 5. 모델별 성능 측정
for name, model in models.items():
    # Validation 평가
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    mean_val = (acc + f1 + auc) / 3
    val_results[name] = [acc, f1, auc, mean_val]

    # Cross-validation 평가
    scores = cross_validate(model, X, y, cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])
    acc_cv = scores["test_accuracy"].mean()
    f1_cv = scores["test_f1"].mean()
    auc_cv = scores["test_roc_auc"].mean()
    mean_cv = (acc_cv + f1_cv + auc_cv) / 3
    cv_results[name] = [acc_cv, f1_cv, auc_cv, mean_cv]

# 6. 결과 출력
print("Model Performance on Validation Set:")
print(f"{'Model':<18} {'Accuracy':>10} {'F1-score':>10} {'AUC':>10} {'MeanScore':>12}")
for name, (a, f, u, m) in val_results.items():
    print(f"{name:<18} {a:10.6f} {f:10.6f} {u:10.6f} {m:12.6f}")

print("\nCross-Validation Results (5-fold):")
print(f"{'Model':<18} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'MeanScore':>12}")
for name, (a, f, u, m) in cv_results.items():
    print(f"{name:<18} {a:10.6f} {f:10.6f} {u:10.6f} {m:12.6f}")

# 7. 테스트셋 예측 및 결과 저장
test_final = pd.read_csv("../test_final_with_selection.csv")  # 모델 입력용
test_raw = pd.read_csv("../test.csv")  # 원본 id 추출용

X_test = test_final[X.columns]  # 학습한 컬럼 순서와 맞춰줌
test_id = test_raw["id"]        # 원본 id 사용

final_model = list(models.values())[0].fit(X, y)  # 전체 train으로 학습
y_pred = final_model.predict(X_test)

submission = pd.DataFrame({
    "id": test_id,
    "y_predict": y_pred
})
submission.to_csv("../prediction_stacking.csv", index=False)

# 8. 모델 저장
joblib.dump(final_model, "../stacking_model.pkl")

print("\nprediction_stacking.csv, stacking_model.pkl 저장 완료")