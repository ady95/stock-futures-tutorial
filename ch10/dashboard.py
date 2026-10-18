"""10장 실습: 모의운영 모니터링 대시보드 (Streamlit).

실행:  streamlit run ch10/dashboard.py
       브라우저에서 표시되는 주소(기본 http://localhost:8501)를 연다.
필요:  9장 run_paper.py 의 기록(live/ 또는 --root 로 지정한 폴더), data/real 일봉 (ch03/collect_real.py)
"""
import glob
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab import MINI_MULT, REG_MULT  # noqa: E402
from hedgelab.backtest import MarginModel  # noqa: E402
from hedgelab.live import ledger_from_decisions  # noqa: E402

st.set_page_config(page_title="AI 선물 헤지 모의운영", layout="wide")
st.title("AI 선물 헤지 모의운영 대시보드")
st.caption("실제 주문 없음 · 가상 체결 · 교육·연구용")

root = st.sidebar.text_input("기록 폴더", "live")
episodes = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p))
if not episodes:
    st.warning(f"{root}/ 아래에 기록이 없습니다. 먼저 python ch09/run_paper.py 를 실행하세요.")
    st.stop()
ep_id = st.sidebar.selectbox("에피소드 (정규 만기 연-월)", episodes, index=len(episodes) - 1)
data_root = st.sidebar.text_input("일봉 폴더", "data/real")
warn_margin = st.sidebar.slider("증거금 사용률 경고 기준", 0.1, 0.6, 0.3, 0.05)
warn_dd = st.sidebar.slider("에피소드 낙폭 경고 기준 (시작 자본 대비)", 0.01, 0.10, 0.04, 0.01)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


decisions_list = read_jsonl(os.path.join(root, ep_id, "decisions.jsonl"))
orders = pd.DataFrame(read_jsonl(os.path.join(root, ep_id, "orders.jsonl")))
decisions = {d["date"]: d for d in decisions_list}
year, month = map(int, ep_id.split("-"))

ledger, marked = ledger_from_decisions(data_root, year, month, decisions)
last = decisions_list[-1]
n_reg, h = last["final"]
equity0 = 50_000_000
start_equity = equity0
peak = max(ledger.equity.max(), start_equity, marked)
drawdown = (peak - marked) / equity0
price = last.get("prices", {}).get("reg") or 0
margin_use = MarginModel().required(n_reg, h, price) / marked if price else 0.0

# ---------------------------------------------------------------- 1) 현재 상태
c = st.columns(3)
c[0].metric("최근 결정일", last["date"], last.get("source", ""))
c[1].metric("포지션 (정규, 미니 매도)", f"{n_reg}, {h}")
c[2].metric("순노출 (원/포인트)", f"{REG_MULT * n_reg - MINI_MULT * h:,}")
c = st.columns(3)
c[0].metric("평가 순자산", f"{marked:,.0f}원", f"{marked - start_equity:+,.0f}원")
c[1].metric("에피소드 낙폭", f"{drawdown:.1%}")
c[2].metric("증거금 사용률", f"{margin_use:.1%}")

if last.get("blocked"):
    st.error(f"주문 차단: {last['blocked']} — 기존 포지션 유지")
if margin_use > warn_margin:
    st.warning(f"증거금 사용률 {margin_use:.1%} > 기준 {warn_margin:.0%}")
if drawdown > warn_dd:
    st.warning(f"에피소드 낙폭 {drawdown:.1%} > 기준 {warn_dd:.0%}")

# ---------------------------------------------------------------- 2) 그래프
left, right = st.columns(2)
with left:
    st.subheader("누적 손익 (종가 기준 장부, 원)")
    st.line_chart(ledger.equity - equity0)
with right:
    st.subheader("미니 매도 계약 수와 정규 보유")
    st.line_chart(ledger[["h", "n_reg"]])

# ---------------------------------------------------------------- 3) 결정·위험관리 로그
st.subheader("결정 로그")
log = pd.DataFrame(decisions_list)
cols = [c for c in ["date", "time", "source", "current", "action", "target", "final", "masked",
                    "risk_adjusted", "blocked", "equity"] if c in log]
st.dataframe(log[cols].iloc[::-1], use_container_width=True, hide_index=True)
n_live = int((log.source != "backfill").sum())
n_block = int(log.get("blocked", pd.Series(dtype=object)).notna().sum())
n_adj = int(log.get("risk_adjusted", pd.Series(dtype=object)).fillna(False).astype(bool).sum())
st.caption(f"실시간·재생 결정 {n_live}회 · 백필 {len(log) - n_live}회 · 차단 {n_block}회 · 위험엔진 조정 {n_adj}회")

st.subheader("가상 주문")
if orders.empty:
    st.info("가상 주문이 없습니다 (포지션 변화 없음).")
else:
    st.dataframe(orders.iloc[::-1], use_container_width=True, hide_index=True)
