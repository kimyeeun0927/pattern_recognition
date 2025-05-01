import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

def select_features_from_train(df: pd.DataFrame) -> (pd.DataFrame, list):
    drop_path = os.path.join('data', 'rows_to_drop.csv')
    if os.path.exists(drop_path):
        rows_to_drop = pd.read_csv(drop_path, header=None).squeeze().tolist()
        df = df.drop(index=rows_to_drop)
        print(f"✅ Dropped {len(rows_to_drop)} rows from train")
    else:
        print("⚠️ rows_to_drop.csv not found.")

    dummy_cols = [col for col in df.columns if col.startswith('data_channel_') or col.startswith('weekday_')]
    numeric_cols = [col for col in df.columns if col not in dummy_cols and col not in {'id', 'shares', 'y'}]

    df_numeric = df[numeric_cols]

    vt = VarianceThreshold(threshold=0.0)
    vt.fit(df_numeric)
    low_var = df_numeric.columns[~vt.get_support()]

    corr = df_numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [c for c in upper.columns if any(upper[c] > 0.9)]

    vif_vals = [variance_inflation_factor(df_numeric.values, i) for i in range(df_numeric.shape[1])]
    vif_drop = df_numeric.columns[np.array(vif_vals) > 10]

    custom_drop = {'kw_avg_min'}
    drop_cols = set(low_var) | set(high_corr) | set(vif_drop) | custom_drop

    final_numeric = [c for c in numeric_cols if c not in drop_cols]
    final_cols = final_numeric + dummy_cols

    df_result = df[final_cols + ['y']]
    return df_result, final_cols  # 컬럼 리스트도 같이 반환

def apply_feature_selection(df: pd.DataFrame, selected_cols: list) -> pd.DataFrame:
    # test에는 y 없으니까 제외
    return df[selected_cols]

if __name__ == "__main__":
    train = pd.read_csv('data/train_encoded.csv')
    test = pd.read_csv('data/test_encoded.csv')

    train_sel, selected_cols = select_features_from_train(train)
    test_sel = apply_feature_selection(test, selected_cols)

    train_sel.to_csv('data/train_selected.csv', index=False)
    test_sel.to_csv('data/test_selected.csv', index=False)

    print("✅ Feature selection complete — test와 train 컬럼 일치!")
