import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

def drop_features(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
    base_drop = {'id', 'shares'}
    y_col = 'y'

    # 0. Drop noisy rows (train only)
    if is_train:
        drop_path = os.path.join('data', 'rows_to_drop.csv')
        if os.path.exists(drop_path):
            rows_to_drop = pd.read_csv(drop_path, header=None).squeeze().tolist()
            df = df.drop(index=rows_to_drop)
            print(f"✅ Dropped {len(rows_to_drop)} rows listed in rows_to_drop.csv")
        else:
            print("⚠️ rows_to_drop.csv not found. No rows dropped.")

    # 1. 분리: 더미 변수 vs 수치 변수
    dummy_cols = [col for col in df.columns if col.startswith('data_channel_') or col.startswith('weekday_')]
    cols_to_exclude = base_drop | ({y_col} if is_train else set())
    numeric_cols = [col for col in df.columns if col not in dummy_cols and col not in cols_to_exclude]

    df_dummy = df[dummy_cols]
    df_numeric = df[numeric_cols]

    # 2. VarianceThreshold
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(df_numeric)
    low_var = df_numeric.columns[~vt.get_support()]

    # 3. Correlation
    corr = df_numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [c for c in upper.columns if any(upper[c] > 0.9)]

    # 4. VIF
    vif_vals = [variance_inflation_factor(df_numeric.values, i) for i in range(df_numeric.shape[1])]
    vif_drop = df_numeric.columns[np.array(vif_vals) > 10]

    # 5. Custom drop
    custom_drop = {'kw_avg_min'}
    drop_cols = set(low_var) | set(high_corr) | set(vif_drop) | custom_drop
    df_numeric_filtered = df_numeric.drop(columns=list(drop_cols))

    # 6. 최종 병합
    result = pd.concat([df_numeric_filtered, df_dummy], axis=1)
    if is_train:
        result = pd.concat([df[[y_col]], result], axis=1)

    return result

if __name__ == "__main__":
    train_path = os.path.join('data', 'train_encoded.csv')
    test_path = os.path.join('data', 'test_encoded.csv')

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_sel = drop_features(train, is_train=True)
    test_sel = drop_features(test, is_train=False)

    train_sel.to_csv(os.path.join('data', 'train_selected.csv'), index=False)
    test_sel.to_csv(os.path.join('data', 'test_selected.csv'), index=False)

    print("✅ Feature selection complete : train_selected.csv, test_selected.csv saved to /data/")
