"""08-5 실습: 에피소드 단위 비교와 부트스트랩 — 차이가 우연일 가능성 가늠하기.

실행:  python ch08/episode_stats.py
에피소드(표준 쌍)마다 순손익과 에피소드 내 최대낙폭을 구하고,
R2(시드 3개 평균)와 기준선의 차이를 에피소드 재표본추출(부트스트랩)로 신뢰구간을 낸다.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab.backtest import run_many  # noqa: E402
from hedgelab.episode import build_all  # noqa: E402
from hedgelab.strategies import BASELINES  # noqa: E402

YEARS = range(2020, 2027)


def per_episode(rec):
    pnl = rec.pnl_reg + rec.pnl_mini - rec.cost
    g = pd.DataFrame({"pnl": pnl, "ep": rec["episode"]}).groupby("ep", sort=False).pnl
    total = g.sum()
    dd = g.apply(lambda s: (s.cumsum() - s.cumsum().cummax().clip(lower=0)).min())
    return pd.DataFrame({"pnl": total, "mdd": dd})


episodes = build_all("data/real", 2020, 2026)
table = {name: per_episode(run_many(episodes, BASELINES[name]))
         for name in ("B1 고정 헤지(롤오버)", "B1n 고정 헤지(롤오버 없음)", "B2 ATR 규칙형")}
for name in ("wf_r1", "wf_r2_risk"):
    seeds = []
    for s in (0, 1, 2):
        rec = pd.concat([pd.read_parquet(f"results/wf/{name}_{T}_seed{s}.parquet") for T in YEARS])
        seeds.append(per_episode(rec))
    table[name + " (시드 평균)"] = sum(seeds) / len(seeds)

summary = pd.DataFrame({k: {"에피소드 수": len(v), "평균 순손익": v.pnl.mean(), "이익 에피소드": int((v.pnl > 0).sum()),
                            "평균 에피소드 MDD": v.mdd.mean(), "최악 에피소드 MDD": v.mdd.min()}
                        for k, v in table.items()}).T
print("== 에피소드 단위 요약 (실제 2020~2026, 원) ==")
print(summary.round(0).to_string())

rng = np.random.default_rng(0)
r2 = table["wf_r2_risk (시드 평균)"]
print("\n== R2(시드 평균) − 기준선: 에피소드 부트스트랩 10,000회, 95% 구간 ==")
for base in ("B1 고정 헤지(롤오버)", "B2 ATR 규칙형"):
    for col, label in (("pnl", "평균 순손익 차이"), ("mdd", "평균 에피소드 MDD 차이")):
        diff = (r2[col] - table[base][col]).to_numpy()
        boots = [rng.choice(diff, len(diff)).mean() for _ in range(10_000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  vs {base:<16} {label:<14} {diff.mean():>12,.0f}  [{lo:>12,.0f} ~ {hi:>12,.0f}]  "
              f"(차이>0 비율 {np.mean(np.array(boots) > 0):.0%})")
