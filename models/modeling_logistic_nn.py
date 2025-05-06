import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os

# 데이터 불러오기
train = pd.read_csv('../data/final_train.csv')
test_features = pd.read_csv('../data/test_selected.csv')
test_ids = pd.read_csv('../data/test.csv')['id']

# 학습/검증 분리
X = train.drop(columns=['y'])
y = train['y']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_features)

# 모델 정의
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42)
}

# 성능 저장
val_results = {}
cv_results = {}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = ['accuracy', 'f1', 'roc_auc']

# 평가 시작
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    val_pred = model.predict(X_val_scaled)
    val_prob = model.predict_proba(X_val_scaled)[:, 1]

    acc = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred)
    auc = roc_auc_score(y_val, val_prob)
    val_results[name] = {'Accuracy': acc, 'F1': f1, 'AUC': auc}

    cv_score = cross_validate(model, X_scaled, y, cv=kfold, scoring=metrics)
    cv_results[name] = {
        'CV_Accuracy': cv_score['test_accuracy'].mean(),
        'CV_F1': cv_score['test_f1'].mean(),
        'CV_AUC': cv_score['test_roc_auc'].mean()
    }

# 결과 출력
val_df = pd.DataFrame(val_results).T
val_df['Mean'] = val_df.mean(axis=1)
print('\nValidation Performance:')
print(val_df)

cv_df = pd.DataFrame(cv_results).T
cv_df['Mean'] = cv_df.mean(axis=1)
print('\nCross-Validation Performance (5-fold):')
print(cv_df)

# Best 모델 고르기 (CV 기준)
best_model_name = cv_df['Mean'].idxmax()
best_model = models[best_model_name]
best_model.fit(X_scaled, y)

# 예측 및 저장
y_final = best_model.predict(X_test_scaled)
y_final_prob = best_model.predict_proba(X_test_scaled)[:, 1]
submission = pd.DataFrame({'id': test_ids, 'y_predict': y_final, 'y_prob': y_final_prob})

os.makedirs('../outputs', exist_ok=True)
save_path = f'../outputs/{best_model_name.lower()}_predict.csv'
submission.to_csv(save_path, index=False)
print(f'\n📁 Saved to: {save_path}')
