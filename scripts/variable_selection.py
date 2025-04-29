import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

def drop_features(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
    base_drop = {'id', 'shares'}
    y_col = 'y'

    # 0. Drop rows listed in rows_to_drop.csv (only for train)
    if is_train:
        drop_path = os.path.join('data', 'rows_to_drop.csv')
        if os.path.exists(drop_path):
            rows_to_drop = pd.read_csv(drop_path, header=None).squeeze().tolist()
            df = df.drop(index=rows_to_drop)
            print(f"✅ Dropped {len(rows_to_drop)} rows listed in rows_to_drop.csv")
        else:
            print("⚠️ rows_to_drop.csv not found. No rows dropped.")

    if is_train:
        X = df.drop(columns=list(base_drop | {y_col}))
    else:
        X = df.drop(columns=list(base_drop))

    # 1. VarianceThreshold
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X)
    low_var = X.columns[~vt.get_support()]

    # 2. Correlation
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [c for c in upper.columns if any(upper[c] > 0.9)]

    # 3. VIF
    vif_vals = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_drop = X.columns[np.array(vif_vals) > 10]

    # 4. Custom drop
    custom_drop = {'kw_avg_min', 'data_channel_World', 'weekday_Sunday'}

    drop_cols = set(low_var) | set(high_corr) | set(vif_drop) | custom_drop | base_drop
    if is_train:
        return df.drop(columns=list(drop_cols))
    else:
        return df.drop(columns=list(drop_cols - {y_col}))  # y는 test에 없음

if __name__ == "__main__":
    train_path = os.path.join('data', 'train_encoded.csv')
    test_path = os.path.join('data', 'test_encoded.csv')

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_sel = drop_features(train, is_train=True)
    test_sel = drop_features(test, is_train=False)

    train_sel.to_csv(os.path.join('data', 'train_selected.csv'), index=False)
    test_sel.to_csv(os.path.join('data', 'test_selected.csv'), index=False)

    print("✅ Feature selection complete: train_selected.csv, test_selected.csv saved to /data/")
