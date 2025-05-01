import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from matplotlib.backends.backend_pdf import PdfPages

# Mac에서 한글 폰트 깨짐 방지
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 0. 데이터 불러오기
# -----------------------------
df = pd.read_csv('train.csv')

# -----------------------------
# 1. 변수 구분 및 결측치 분석
# -----------------------------
missing_counts = df.isnull().sum().sort_values(ascending=False)
missing_ratio = (df.isnull().sum() / len(df)).sort_values(ascending=False)

numeric_cols = df.select_dtypes(include='number').columns.tolist()
categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
cols_to_analyze = [col for col in numeric_cols if col not in ['id', 'shares', 'y']]

# 타겟 y에 따른 평균 차이
y_grouped_mean = df.groupby('y')[cols_to_analyze].mean().T
y_grouped_mean['diff'] = abs(y_grouped_mean[1] - y_grouped_mean[0])
top_diff = y_grouped_mean.sort_values(by='diff', ascending=False).head(10)
top_diff_vars = top_diff.index.tolist()[:3]

# -----------------------------
# 2. 시각화 결과 PDF로 저장
# -----------------------------
with PdfPages("../eda_visualizations_clean.pdf") as pdf:

    # [1] 결측치 매트릭스
    msno.matrix(df)
    plt.title("Missing Value Matrix\n→ 전체 데이터에서 결측치가 어떤 열에 퍼져 있는지 시각적으로 확인")
    pdf.savefig(); plt.close()

    # [2] 결측치 바 차트
    msno.bar(df)
    plt.title("Missing Value Bar Chart\n→ 결측치가 많은 상위 변수들을 막대그래프로 확인")
    pdf.savefig(); plt.close()

    # [3] 결측치 많은 변수 TOP 10 수평 막대 그래프
    plt.figure(figsize=(10, 5))
    missing_counts.head(10).plot(kind='barh')
    plt.title("결측치 많은 변수 TOP 10\n→ 결측치를 우선적으로 처리해야 할 주요 변수 파악")
    plt.xlabel("결측치 개수")
    plt.ylabel("변수 이름")
    plt.tight_layout()
    pdf.savefig(); plt.close()

    # [4] 대표 변수 히스토그램 (y별)
    for col in top_diff_vars:
        plt.figure(figsize=(8, 4))
        sns.histplot(data=df, x=col, hue='y', kde=True, bins=30)

        # 설명 추가
        if col == "self_reference_max_shares":
            title = f"{col} 분포 (y=0 vs y=1)\n→ 인기 기사일수록 자기 참조 수치가 높게 나타남"
        elif col == "kw_max_max":
            title = f"{col} 분포 (y=0 vs y=1)\n→ y=0일 때 키워드 최대값이 더 큼"
        elif col == "kw_avg_max":
            title = f"{col} 분포 (y=0 vs y=1)\n→ 평균 최대 키워드 값은 y=1에서 살짝 높음"
        else:
            title = f"{col} 분포 (y=0 vs y=1)"

        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("빈도수")
        plt.tight_layout()
        pdf.savefig(); plt.close()

    # [5] 상관관계 히트맵
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[cols_to_analyze].corr(), cmap='coolwarm', center=0)
    plt.title("수치형 변수 간 상관관계 Heatmap\n→ 변수 간 강한 상관성을 시각적으로 파악")
    plt.tight_layout()
    pdf.savefig(); plt.close()
