"""03-4 실습 2: 두 월물 정렬과 거래량 필터의 효과.

실행:  python ch03/align_example.py [데이터폴더]        기본 data/real
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab.contracts import pair_for  # noqa: E402
from hedgelab.data import align_pair, hedge_window, load_daily  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "data/real"
reg_code, mini_code = pair_for(2025, 12)          # 101W12 + 105W11
reg, mini = load_daily(root, reg_code), load_daily(root, mini_code)
print(f"{reg_code}: {len(reg)}일 {reg.index[0]:%Y-%m-%d}~{reg.index[-1]:%Y-%m-%d}")
print(f"{mini_code}: {len(mini)}일 {mini.index[0]:%Y-%m-%d}~{mini.index[-1]:%Y-%m-%d}")

raw = align_pair(reg, mini, min_volume=0)         # 필터 없음
flt = align_pair(reg, mini, min_volume=100)       # 양쪽 모두 하루 100계약 이상
print(f"\n공통 거래일 {len(raw)}일 → 거래량 필터 후 {len(flt)}일")
for name, a in (("필터 없음", raw), ("필터 적용", flt)):
    print(f"  {name}: 스프레드 평균 {a.spread.mean():6.2f}pt, 표준편차 {a.spread.std():5.2f}pt, "
          f"최소 {a.spread.min():6.2f}, 최대 {a.spread.max():6.2f}")

w = hedge_window(flt, mini.index.max())
print(f"\n헤지 구간(미니 만기 전 45일): {len(w)}일 {w.index[0]:%Y-%m-%d}~{w.index[-1]:%Y-%m-%d}")
print(w.tail(5).round(2).to_string())
