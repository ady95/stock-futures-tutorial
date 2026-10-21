"""08-1~08-5 실습: 워크포워드 결과 보고서.

실행:  python ch08/report.py
입력:  results/wf/*.parquet (walk_forward.py 결과), data/real
출력:  1) 전체 평가 구간(2020~2026) 성과표 — 기준선 vs PPO(시드별)
       2) 연도별 순손익
       3) 손익 분해와 노출
       4) results/ch08_equity.png 누적 손익 그래프
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab import plotting  # noqa: E402
from hedgelab.backtest import run_many  # noqa: E402
from hedgelab.episode import build_all  # noqa: E402
from hedgelab.metrics import summary  # noqa: E402
from hedgelab.strategies import BASELINES  # noqa: E402

EQUITY0 = 50_000_000
YEARS = list(range(2020, 2027))
KEYS = ["순손익(원)", "MDD(원)", "Sharpe", "CVaR5%(원)", "최악일(원)", "평균 노출(정규 계약)",
        "거래 계약 수", "방향 손익(원)", "스프레드 손익(원)", "비용(원)"]


def chain(frames):
    """연도별 기록을 이어 붙이고 순자산을 다시 계산한다."""
    out = pd.concat(frames)
    out["equity"] = EQUITY0 + (out.pnl_reg + out.pnl_mini - out.cost).cumsum()
    out.attrs["equity0"] = EQUITY0
    return out


def yearly_pnl(rec):
    """에피소드(정규 만기 연도) 기준 연도별 순손익. 3월물 에피소드는 전년 12월 말에 시작하므로 달력 연도로 묶지 않는다."""
    pnl = rec.pnl_reg + rec.pnl_mini - rec.cost
    return pnl.groupby(rec["episode"].str[:4].astype(int)).sum()


episodes = build_all("data/real", YEARS[0], YEARS[-1])
print(f"평가 구간 에피소드 {len(episodes)}개: {episodes[0].index[0]:%Y-%m-%d} ~ {episodes[-1].index[-1]:%Y-%m-%d}")

recs = {name: run_many(episodes, policy, EQUITY0) for name, policy in BASELINES.items()}
for name in ("wf_r1", "wf_r2_risk"):
    for seed in (0, 1, 2):
        files = [f"results/wf/{name}_{T}_seed{seed}.parquet" for T in YEARS]
        if all(os.path.exists(f) for f in files):
            recs[f"{name} s{seed}"] = chain([pd.read_parquet(f) for f in files])

table = pd.DataFrame({k: summary(v, EQUITY0) for k, v in recs.items()}).reindex(KEYS)
with pd.option_context("display.width", 300, "display.max_columns", 20):
    print("\n== 1) 전체 평가 구간 성과 (2020~2026, 시작 자본 5,000만 원) ==")
    print(table.to_string())

    print("\n== 2) 연도별 순손익 (원) ==")
    by_year = pd.DataFrame({k: yearly_pnl(v) for k, v in recs.items()}).round(0)
    print(by_year.to_string())

# 학습 기간 체크포인트 선택 결과
best_steps = [pd.read_parquet(f)["best_step"].iloc[0] for f in glob.glob("results/wf/*.parquet")]
if best_steps:
    print(f"\n검증으로 고른 체크포인트 스텝: 중앙값 {int(np.median(best_steps)):,}, "
          f"분포 {pd.Series(best_steps).value_counts().sort_index().to_dict()}")

plt = plotting.setup()
fig, axes = plt.subplots(2, 1, figsize=(9, 9.5))   # 세로 배치: 책(A4) 폭에서도 글자가 읽히게
ref = recs["B0 무거래"]
years = ref["episode"].str[:4].to_numpy()
ticks = [i for i in range(len(ref)) if i == 0 or years[i] != years[i - 1]]
for ax, group, title in zip(axes, ("wf_r2_risk", "wf_r1"), ("R2 + 위험엔진 vs 기준선", "R1 (페널티 없음) vs 기준선")):
    for name, rec in recs.items():
        if name.startswith("wf_") and not name.startswith(group):
            continue
        style = {"lw": 1.0, "ls": "--"} if name.startswith("wf_") else {"lw": 1.8}
        ax.plot(range(len(rec)), (rec.equity - EQUITY0) / 1e6, label=name.replace(" 고정 헤지", ""), **style)
    ax.set_xticks(ticks)
    ax.set_xticklabels(years[ticks])
    ax.set_title(title)
    ax.set_ylabel("누적 손익 (백만 원)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
fig.suptitle("워크포워드 평가 구간 누적 손익 — 실제 데이터\n(에피소드 기준 연도 2020~2026)")
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/ch08_equity.png", dpi=120)
print("그래프 저장: results/ch08_equity.png")
