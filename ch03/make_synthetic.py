"""03-5 실습 1: 합성 데이터셋 만들기.

실행:  python ch03/make_synthetic.py
결과:  data/synthetic/daily/*.parquet, data/synthetic/market.parquet
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab import synthetic  # noqa: E402

frames, market = synthetic.generate(start="2020-01-01", years=6, s0=300.0, seed=4)
root = synthetic.save(frames, market, "data/synthetic")

print(f"월물 {len(frames)}개 저장 → {root}/daily/")
print(f"기간 {market.index[0]:%Y-%m-%d} ~ {market.index[-1]:%Y-%m-%d} ({len(market)}거래일)")
print("국면별 일수:", market.regime.value_counts().to_dict())
print(f"스프레드 확대 국면 일수: {int(market.widen.sum())}")
print(f"지수 시작 {market['index'].iloc[0]:.2f} → 끝 {market['index'].iloc[-1]:.2f}, "
      f"최저 {market['index'].min():.2f}, 최고 {market['index'].max():.2f}")

code = "SR202312"
df = frames[code]
print(f"\n예시 월물 {code}: {len(df)}행")
print(df.tail(3).to_string(index=False))
