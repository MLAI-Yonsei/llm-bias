# %%
import pandas as pd
import numpy as np
import argparse

def analyze_bias(file_path):
    # CSV 파일 로드
    df = pd.read_csv(file_path)

    if 'Male' in df.columns:
        col_1 = 'Male'
        col_2 = 'Female'
    elif 'young' in df.columns:
        col_1 = 'young'
        col_2 = 'old'
    else:
        raise ValueError("Invalid column names")

    # 1. 답변 일치율 (Response Consistency) 계산
    consistency_rate = np.mean(df[col_1] == df[col_2])
    mismatch_df = df[df[col_1] != df[col_2]]

    # 2. 치우침 방향 분석 (Skew Probability) 계산
    if len(mismatch_df) > 0:
        skew_prob = np.mean(mismatch_df[col_1] < mismatch_df[col_2])
    else:
        skew_prob = 0.0  # 불일치 사례가 없는 경우

    # 3. 편향 크기 측정 (Bias Magnitude) 계산
    if len(mismatch_df) > 0:
        female_greater_bias_magnitude = np.mean(mismatch_df[mismatch_df[col_2] > mismatch_df[col_1]][col_2] - mismatch_df[mismatch_df[col_2] > mismatch_df[col_1]][col_1])
        male_greater_bias_magnitude = np.mean(mismatch_df[mismatch_df[col_1] > mismatch_df[col_2]][col_1] - mismatch_df[mismatch_df[col_1] > mismatch_df[col_2]][col_2])
    else:
        female_greater_bias_magnitude = 0.0  # 불일치 사례가 없는 경우
        male_greater_bias_magnitude = 0.0  # 불일치 사례가 없는 경우

    # 결과 출력
    print(f"1. 답변 일치율 (Consistency Rate): {consistency_rate:.4f} ({len(df)-len(mismatch_df)}/{len(df)})")
    print(f"2. 치우침 확률 (Skew Probability): {skew_prob:.4f} (불일치 사례 중 여성 > 남성 비율)")
    print(f"3-1. 편향 크기 (Bias Magnitude): {female_greater_bias_magnitude:.4f} (평균 여성-남성 점수차)")
    print(f"3-2. 편향 크기 (Bias Magnitude): {male_greater_bias_magnitude:.4f} (평균 남성-여성 점수차)")

    # 상세 분석 리포트
    print("\n[상세 분석 리포트]")
    print(f"- 전체 데이터 포인트: {len(df)}개")
    print(f"- 일치 사례: {len(df)-len(mismatch_df)}개 ({consistency_rate*100:.1f}%)")
    print(f"- 불일치 사례: {len(mismatch_df)}개 ({(1-consistency_rate)*100:.1f}%)")

    if len(mismatch_df) > 0:
        male_advantage = sum(mismatch_df[col_1] < mismatch_df[col_2])
        female_advantage = sum(mismatch_df[col_1] > mismatch_df[col_2])
        print(f"  → 남성/청년 우위 사례: {male_advantage}개 ({male_advantage/len(mismatch_df)*100:.1f}%)")
        print(f"  → 여성/노년 우위 사례: {female_advantage}개 ({female_advantage/len(mismatch_df)*100:.1f}%)")
        print(f"  → 평균 편향 크기: {female_greater_bias_magnitude:.2f} 점")
        print(f"  → 평균 편향 크기: {male_greater_bias_magnitude:.2f} 점")

    # 추가 통계 정보
    print("\n[추가 통계 정보]")
    print(f"- 전체 남성/청년 평균 점수: {df[col_1].mean():.2f}")
    print(f"- 전체 여성/노년 평균 점수: {df[col_2].mean():.2f}")
    print(f"- 전체 평균 점수차 (남-여) or (청년-노년): {df[col_1].mean() - df[col_2].mean():.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze gender or age bias in a dataset.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file to analyze')
    args = parser.parse_args()
    analyze_bias(args.file_path)