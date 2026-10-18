"""09-4 실습: 하루 한 번 모의운영 결정 (실제 주문 없음).

실행:
  python ch09/run_paper.py                              오늘, 장 마감 무렵(15:20~15:35 권장)에 실행
  python ch09/run_paper.py --replay 2026-10-16          과거 하루를 그날 종가로 재생 (장 시간과 무관하게 연습)
옵션:
  --config configs/wf_r2_risk.yaml  --model models/wf/wf_r2_risk_2026_seed0.zip  --root live
기록:
  <root>/<에피소드>/decisions.jsonl   결정 (백필·실시간·차단 여부)
  <root>/<에피소드>/orders.jsonl      가상 주문
주문 차단: 시세 이상, 또는 <root>/KILL 파일이 있으면 기존 포지션을 유지한다.
"""
import argparse
import os
import sys
from datetime import date, datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab.contracts import expiry  # noqa: E402
from hedgelab.kis import KISClient  # noqa: E402
from hedgelab.live import PaperBook, check_quote, current_pair, fetch_history, pair_codes, run_once  # noqa: E402
from hedgelab.rl import load_config, model_class, predict  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/wf_r2_risk.yaml")
ap.add_argument("--model", default="models/wf/wf_r2_risk_2026_seed0.zip")
ap.add_argument("--replay", default=None, help="YYYY-MM-DD: 그날 종가를 현재가로 보고 재생")
ap.add_argument("--root", default="live")
args = ap.parse_args()

today = datetime.strptime(args.replay, "%Y-%m-%d").date() if args.replay else date.today()
pair = current_pair(today)
if pair is None:
    print(f"{today}: 진행 중인 표준 쌍이 없습니다 (무포지션 기간)")
    sys.exit(0)
codes = pair_codes(*pair)
cfg = load_config(args.config)
model = model_class(cfg).load(args.model, device="cpu")
client = KISClient(mode="paper")


def exact_expiry(code, fallback):
    """현재가 응답의 최종거래일을 쓰고, 만기가 지나 조회되지 않으면 둘째 목요일 근사값을 쓴다."""
    q = client.price(code)
    d = q.get("futs_last_tr_date")
    return datetime.strptime(d, "%Y%m%d").date() if d else fallback


y, m = pair
reg_exp = exact_expiry(codes["reg"], expiry(y, m))
a_exp = exact_expiry(codes["a"], expiry(y, m - 1))
start = (a_exp - timedelta(days=70)).strftime("%Y%m%d")
root = os.path.join(args.root)
book = PaperBook(root, f"{y}-{m:02d}")
print(f"{today} 에피소드 {y}-{m:02d}: 정규 {codes['reg']}(만기 {reg_exp}), "
      f"미니 a {codes['a']}(만기 {a_exp}), 미니 b {codes['b']}")

full = fetch_history(client, codes, start, today + timedelta(days=1))
if pd.Timestamp(today) not in full["reg"].index:
    # 휴장일에는 현재가 API가 직전 거래일 값을 돌려준다. 그 값을 오늘 시세로 착각하지 않도록 결정을 건너뛴다
    print(f"{today}: 오늘 일봉이 없습니다 (휴장일이거나 장 시작 전) — 결정하지 않음")
    sys.exit(0)
hist = {k: v[v.index < pd.Timestamp(today)] for k, v in full.items()}
if args.replay:
    bar = full["reg"].loc[pd.Timestamp(today)]
    quotes = {"reg_bar": (bar.close, bar.high, bar.low),
              "a": full["a"].close.get(pd.Timestamp(today)), "b": full["b"].close.get(pd.Timestamp(today))}
else:
    q_reg = client.price(codes["reg"])
    q_b = client.price(codes["b"])
    q_a = client.price(codes["a"]) if today <= a_exp else {}
    problems = [check_quote(q_reg, hist["reg"].close.iloc[-1]), check_quote(q_b, hist["b"].close.iloc[-1])]
    if today <= a_exp:
        problems.append(check_quote(q_a, hist["a"].close.iloc[-1]))
    blocked = next((p for p in problems if p), None)
    quotes = {"reg_bar": (float(q_reg.get("futs_prpr") or 0), float(q_reg.get("futs_hgpr") or 0),
                          float(q_reg.get("futs_lwpr") or 0)),
              "a": float(q_a["futs_prpr"]) if q_a.get("futs_prpr") else None,
              "b": float(q_b.get("futs_prpr") or 0), "blocked": blocked}
quotes["codes"] = codes

rec = run_once(model, cfg, hist, today, quotes, a_exp, reg_exp, book, predict,
               kill_file=os.path.join(root, "KILL"), source="replay" if args.replay else "live")
print(f"결정: 현재 {rec.get('current')} → 최종 {rec['final']} "
      f"(행동 {rec.get('action')}, 차단 {rec.get('blocked')}, 위험조정 {rec.get('risk_adjusted')}), "
      f"평가 순자산 {rec['equity']:,.0f}원")
