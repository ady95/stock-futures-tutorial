"""10-4 실습: 같은 에피소드의 백테스트(8장 워크포워드)와 모의운영(9장 재생) 비교.

실행:  python ch10/compare_backtest.py [모의운영 기록 폴더] [에피소드]     기본 live_bt 2026-09
비교 대상
  백테스트: results/wf/wf_r2_risk_2026_seed0.parquet 중 해당 에피소드 (build_episode + EpisodeSim)
  모의운영: <폴더>/<에피소드>/decisions.jsonl 을 종가 장부로 다시 계산 (ledger_from_decisions)
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab.live import ledger_from_decisions  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "live_bt"
ep_id = sys.argv[2] if len(sys.argv) > 2 else "2026-09"
year, month = map(int, ep_id.split("-"))

bt = pd.read_parquet(f"results/wf/wf_r2_risk_{year}_seed0.parquet")
bt = bt[bt.episode == ep_id]

decisions = {}
with open(os.path.join(root, ep_id, "decisions.jsonl"), encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        decisions[d["date"]] = d
paper, marked = ledger_from_decisions("data/real", year, month, decisions)

days = bt.index.intersection(paper.index)
pos_bt = list(zip(bt.loc[days, "n_reg"], bt.loc[days, "h"]))
pos_pp = list(zip(paper.loc[days, "n_reg"], paper.loc[days, "h"]))
diff = [d for d, a, b in zip(days, pos_bt, pos_pp) if a != b]

pnl_bt = (bt.pnl_reg + bt.pnl_mini - bt.cost).loc[days].sum()
pnl_pp = (paper.pnl_reg + paper.pnl_mini - paper.cost).loc[days].sum()
print(f"에피소드 {ep_id}: 비교 거래일 {len(days)}일 ({days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d})")
print(f"  포지션이 다른 날: {len(diff)}일")
for d in diff[:10]:
    print(f"    {d:%Y-%m-%d}  백테스트 {tuple(map(int, bt.loc[d, ['n_reg', 'h']]))}  "
          f"모의운영 {tuple(map(int, paper.loc[d, ['n_reg', 'h']]))}")
print(f"  같은 기간 손익: 백테스트 {pnl_bt:,.0f}원 / 모의운영(종가 장부) {pnl_pp:,.0f}원 / 차이 {pnl_pp - pnl_bt:,.0f}원")
