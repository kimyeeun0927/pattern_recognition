import pandas as pd

# 1. 데이터 불러오기
df = pd.read_csv('../test_clean_reg.csv')

# 2. 이상치 처리할 feature 선택
features = [col for col in df.columns if col not in ['id', 'shares', 'y', 'data_channel', 'weekday']]

# 3. 각 feature별로 IQR 계산하고, 이상치 클리핑
for feature in features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)

# 4. 결과 저장
df.to_csv('../test_outlier_handled.csv', index=False)

print("이상치 처리 완료! 저장됨: test_outlier_handled.csv")