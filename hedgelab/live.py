"""모의운영(페이퍼트레이딩) 9장.

하루 한 번, 장 마감 무렵(예: 15:30)에 실행한다.
  1. 오늘이 속한 표준 쌍(에피소드)을 찾는다
  2. 그 에피소드의 지난 일봉 + 오늘의 현재가 + 남은 날(자리표시자)로 에피소드 데이터를 만든다
  3. 학습 때와 같은 CalendarSpreadEnv 에 지난 결정을 다시 넣어 오늘의 관측·계좌 상태를 재현한다
  4. PPO가 오늘의 목표 포지션을 정하고, 위험관리 엔진이 검사한다
  5. 가상 주문을 현재가 ± 슬리피지로 체결한 것으로 기록한다 (실제 주문은 하지 않는다)
시세가 이상하거나 KILL 파일이 있으면 주문을 차단하고 기존 포지션을 유지한다.
"""
import json
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import MINI_TICK, REG_TICK
from .contracts import REG_MONTHS, contract_code, expiry
from .data import to_frame
from .env import CalendarSpreadEnv, RewardConfig
from .risk import RiskEngine

WINDOW_DAYS = 45


# ---------------------------------------------------------------- 오늘의 에피소드 찾기
def current_pair(today):
    """today 가 헤지 구간~정규 만기 사이에 있는 표준 쌍 (연, 월). 없으면 None."""
    for y in (today.year, today.year + 1):
        for m in REG_MONTHS:
            if (y, m) < (today.year, today.month):
                continue
            reg_exp = expiry(y, m)
            a_exp = expiry(y if m > 1 else y - 1, m - 1 if m > 1 else 12)
            if a_exp - timedelta(days=WINDOW_DAYS) <= today <= reg_exp:
                return y, m
    return None


def build_live_episode(hist, today, today_quotes, a_exp, reg_exp):
    """지난 일봉(hist) + 오늘 현재가 + 남은 영업일 자리표시자로 에피소드 DataFrame 을 만든다.

    hist: {"reg": df, "a": df, "b": df} — 날짜 인덱스 일봉 (오늘 이전)
    today_quotes: {"reg": (종가, 고가, 저가), "a": 가격 또는 None, "b": 가격}
    자리표시자 행은 만기까지 남은 날 수(days_to_*)를 맞추기 위한 것이며, 가격은 쓰이지 않는다.
    """
    start = pd.Timestamp(a_exp) - pd.Timedelta(days=WINDOW_DAYS)
    reg = hist["reg"][(hist["reg"].index >= start) & (hist["reg"].index < pd.Timestamp(today))]
    rows = pd.DataFrame({"reg": reg.close, "reg_high": reg.high, "reg_low": reg.low})
    rows = rows[reg.volume > 0]
    rows["mini_a"] = hist["a"].close.reindex(rows.index)
    rows["mini_b"] = hist["b"].close.reindex(rows.index)
    q = today_quotes
    rows.loc[pd.Timestamp(today)] = [q["reg"][0], q["reg"][1], q["reg"][2],
                                     np.nan if q["a"] is None else q["a"], np.nan if q["b"] is None else q["b"]]
    future = pd.bdate_range(pd.Timestamp(today) + pd.Timedelta(days=1), pd.Timestamp(reg_exp))
    for d in future:                                   # 자리표시자: 오늘 가격 복사
        rows.loc[d] = rows.loc[pd.Timestamp(today)]
    rows = rows.sort_index()
    rows["near"] = np.where(rows.index <= pd.Timestamp(a_exp), "a", "b")
    rows["a_last"] = rows.index == pd.Timestamp(a_exp)
    rows["mini_a"] = rows["mini_a"].ffill()
    rows["mini_b"] = rows["mini_b"].ffill().bfill()
    return rows


# ---------------------------------------------------------------- 시세 검사
def check_quote(q, prev_close, max_move=0.15):
    """주문을 막아야 할 시세 이상을 찾는다. 문제가 없으면 None."""
    if not q or float(q.get("futs_prpr") or 0) <= 0:
        return "현재가 없음"
    price = float(q["futs_prpr"])
    if int(q.get("acml_vol") or 0) == 0:
        return "오늘 거래량 0 (장 시작 전이거나 거래 정지)"
    if prev_close and abs(price / prev_close - 1) > max_move:
        return f"전일 대비 {price / prev_close - 1:+.1%} — 데이터 오류 의심"
    return None


# ---------------------------------------------------------------- 운영 상태
class PaperBook:
    """결정·주문 기록 (live/<에피소드>/ 아래 JSON Lines)."""

    def __init__(self, root, episode_id):
        self.dir = os.path.join(root, episode_id)
        os.makedirs(self.dir, exist_ok=True)
        self.decisions_path = os.path.join(self.dir, "decisions.jsonl")
        self.orders_path = os.path.join(self.dir, "orders.jsonl")

    def decisions(self):
        if not os.path.exists(self.decisions_path):
            return {}
        out = {}
        with open(self.decisions_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                out[d["date"]] = d
        return out

    def append(self, path, record):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def target_to_action(target, mode):
    n_reg, h = target
    return 6 if (mode == "R2" and n_reg == 0) else h


def run_once(model, cfg, hist, today, quotes, a_exp, reg_exp, book, predict, kill_file="live/KILL", source="live"):
    """오늘 하루의 결정을 내리고 기록한다. 반환: 결정 기록 dict."""
    ep = build_live_episode(hist, today, {"reg": quotes["reg_bar"], "a": quotes.get("a"), "b": quotes["b"]},
                            a_exp, reg_exp)
    ep.attrs.update(year=reg_exp.year, month=reg_exp.month)
    risk = RiskEngine(**cfg["risk"]) if cfg.get("risk") else None
    env = CalendarSpreadEnv([ep], mode=cfg.get("mode", "R1"), reward=RewardConfig(**cfg.get("reward", {})),
                            risk=risk, random_episode=False)
    obs, _ = env.reset(options={"episode_index": 0})
    logged = book.decisions()
    key_today = pd.Timestamp(today).strftime("%Y-%m-%d")
    if logged.get(key_today, {}).get("source") in ("live", "replay"):
        return {**logged[key_today], "note": "오늘 결정이 이미 있음 — 다시 주문하지 않음"}   # 같은 날 두 번 실행 방지
    today_i = ep.index.get_loc(pd.Timestamp(today))

    # 지난 날들: 기록된 결정을 다시 넣는다. 기록이 없는 날(운영 시작 전)은 모델의 결정으로 채운다(백필)
    for t in range(today_i):
        key = ep.index[t].strftime("%Y-%m-%d")
        backfill = key not in logged
        action = int(predict(model, env, obs)) if backfill else target_to_action(tuple(logged[key]["final"]), env.mode)
        obs, _, _, _, _ = env.step(action)
        if backfill:
            rec = env.sim.records[t]                       # 위험관리 엔진까지 거친 실제 포지션
            book.append(book.decisions_path, {"date": key, "source": "backfill", "action": action,
                                              "final": [int(rec["n_reg"]), int(rec["h"])]})

    # 오늘
    sim = env.sim
    if sim.done:                                           # 오늘이 정규 만기일: 모두 결제, 결정 없음
        record = {"date": pd.Timestamp(today).strftime("%Y-%m-%d"), "source": source, "final": [0, 0],
                  "blocked": None, "note": "정규 만기 결제", "equity": sim.equity}
        book.append(book.decisions_path, record)
        return record
    current = (sim.n_reg, sim.h)
    key = pd.Timestamp(today).strftime("%Y-%m-%d")
    record = {"date": key, "time": datetime.now().strftime("%H:%M:%S"), "source": source,
              "prices": {"reg": quotes["reg_bar"][0], "a": quotes.get("a"), "b": quotes["b"]},
              "current": list(current), "equity": sim.marked_equity}
    blocked = quotes.get("blocked") or ("KILL 파일" if os.path.exists(kill_file) else None)
    if blocked:
        final = current
        record.update(action=None, target=None, final=list(final), blocked=blocked)
    else:
        mask = env.action_masks()
        action = int(predict(model, env, obs))
        target = env.to_target(action)
        final = env.risk.adjust(*target, **env._risk_args()) if env.risk else target
        record.update(action=action, target=list(target), final=list(final),
                      masked=[i for i, ok in enumerate(mask) if not ok], blocked=None,
                      risk_adjusted=final != target)
    book.append(book.decisions_path, record)

    # 가상 체결: 바뀐 수량만큼 현재가 ± 1호가
    near = "b" if ep.a_last.iloc[today_i] else ep.near.iloc[today_i]
    d_reg, d_h = final[0] - current[0], final[1] - current[1]
    if d_reg:
        px = quotes["reg_bar"][0] + np.sign(d_reg) * REG_TICK
        book.append(book.orders_path, {"date": key, "code": quotes["codes"]["reg"], "side": "매수" if d_reg > 0 else "매도",
                                       "qty": abs(d_reg), "price": round(px, 2), "virtual": True})
    if d_h:
        mini_px = quotes["b"] if near == "b" else quotes["a"]
        px = mini_px - np.sign(d_h) * MINI_TICK            # 매도 늘리면 한 호가 낮게
        book.append(book.orders_path, {"date": key, "code": quotes["codes"][near], "side": "매도" if d_h > 0 else "환매수",
                                       "qty": abs(d_h), "price": round(px, 2), "virtual": True})
    return record


def fetch_history(client, codes, start, before):
    """KIS 일봉을 받아 오늘(before) 이전 행만 남긴다."""
    out = {}
    for k, code in codes.items():
        df = to_frame(client.daily(code, start), code)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        out[k] = df[df.index < pd.Timestamp(before)]
    return out


def pair_codes(year, month):
    return {"reg": contract_code("reg", year, month),
            "a": contract_code("mini", year if month > 1 else year - 1, month - 1 if month > 1 else 12),
            "b": contract_code("mini", year, month)}


def today_kst():
    return date.today()


def ledger_from_decisions(data_root, year, month, decisions, equity0=50_000_000):
    """결정 기록을 종가 기준으로 다시 계산한 장부 (10장 대시보드·10-4 비교용).

    가상 주문은 결정 시점 가격으로 기록되지만, 장부는 백테스트와 같은 '종가 체결' 규칙으로 계산한다.
    반환: (일별 기록 DataFrame, 마지막 결정일의 평가 순자산)
    """
    from .backtest import EpisodeSim
    from .data import load_daily

    codes = pair_codes(year, month)
    full = {k: load_daily(data_root, c) for k, c in codes.items()}
    if any(v is None for v in full.values()):
        raise FileNotFoundError(f"{data_root}/daily 에 {list(codes.values())} 일봉이 필요합니다 (ch03/collect_real.py)")
    dates = sorted(decisions)
    last = pd.Timestamp(dates[-1])
    a_exp = full["a"].index.max() if full["a"].index.max() < last else pd.Timestamp(expiry(year, month - 1))
    reg_exp = pd.Timestamp(expiry(year, month))
    hist = {k: v[v.index < last] for k, v in full.items()}
    bar = full["reg"].loc[last]
    quotes = {"reg": (bar.close, bar.high, bar.low), "a": full["a"].close.get(last), "b": full["b"].close.get(last)}
    ep = build_live_episode(hist, last.date(), quotes, a_exp.date(), reg_exp.date())
    ep.attrs.update(year=year, month=month)
    sim = EpisodeSim(ep, equity0)
    final = (0, 0)
    for t in range(ep.index.get_loc(last)):
        key = ep.index[t].strftime("%Y-%m-%d")
        if key in decisions:
            final = tuple(decisions[key]["final"])
        sim.act(*final)
    return sim.frame(), sim.marked_equity
