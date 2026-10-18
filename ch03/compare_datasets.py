"""03-5 실습 2: 실제 데이터와 합성 데이터의 통계 비교.

실행:  python ch03/compare_datasets.py [실제데이터_폴더]
       실제 데이터 폴더 기본값은 data/real (03-4의 collect_real.py 결과). 없으면 합성만 요약한다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab.contracts import pairs_between  # noqa: E402
from hedgelab.data import load_daily  # noqa: E402
from hedgelab.stats import pair_stats, summarize  # noqa: E402
from hedgelab.synthetic import syn_pair_for  # noqa: E402


def collect(root, pairs):
    rows = []
    for reg_code, mini_code in pairs:
        reg, mini = load_daily(root, reg_code), load_daily(root, mini_code)
        if reg is not None and mini is not None:
            rows.append(pair_stats(reg, mini))
    return rows


real_root = sys.argv[1] if len(sys.argv) > 1 else "data/real"
results = {}
if os.path.isdir(os.path.join(real_root, "daily")):
    pairs = [(r, m) for _, _, r, m in pairs_between(2015, 2025)]
    results["실제 2015~2025"] = summarize(collect(real_root, pairs))
syn_pairs = [syn_pair_for(y, m) for y in range(2020, 2026) for m in (3, 6, 9, 12)]
results["합성 seed=4"] = summarize(collect("data/synthetic", syn_pairs))

names = list(results)
keys = list(results[names[0]])
print(f"{'지표':<28}" + "".join(f"{n:>20}" for n in names))
for k in keys:
    print(f"{k:<28}" + "".join(f"{str(results[n][k]):>20}" for n in names))
