"""합성 선물 데이터 생성기 (03-5).

시장 데이터 없이도 모든 실습을 돌릴 수 있도록, 실제 데이터와 같은 형식의 정규·미니 월물 일봉을 만든다.
여기서 나온 성과를 실제 시장 성과로 해석하면 안 된다 — 회계 검증과 학습 실습용이다.

구성
- 지수: 국면 전환(상승·하락·횡보·급락) 로그수익률 모형
- 월물 가격: 지수 × exp(보유비용 × 잔존기간) + 월물별 AR(1) 괴리
- 스프레드 확대 국면: 괴리의 크기가 커지고 정규 쪽으로 치우친다
- 거래량: 만기가 멀수록 줄어 원월물은 거래가 거의 없다
"""
import os

import numpy as np
import pandas as pd

from . import MINI_TICK, REG_TICK
from .contracts import expiry

# 국면별 일간 로그수익률 평균·표준편차
REGIMES = {
    "상승": (0.0008, 0.010),
    "하락": (-0.0008, 0.012),
    "횡보": (0.0, 0.007),
    "급락": (-0.006, 0.035),
}
NAMES = list(REGIMES)
# 행: 오늘 국면, 열: 내일 국면 (상승, 하락, 횡보, 급락)
TRANSITION = np.array([
    [0.980, 0.008, 0.010, 0.002],
    [0.012, 0.968, 0.015, 0.005],
    [0.012, 0.010, 0.975, 0.003],
    [0.100, 0.030, 0.020, 0.850],   # 급락 뒤에는 반등(상승)으로 넘어가기 쉽다
])

CARRY_MEAN = 0.012       # 연 보유비용(이자 − 배당) 평균. 월물 간 스프레드의 평균을 만든다
CARRY_SD = 0.015         # 보유비용은 금리·배당 환경에 따라 천천히 변한다 (정상분포 표준편차)
CARRY_PHI = 0.995
NOISE_PHI = 0.7          # 괴리의 자기상관
NOISE_SD = 0.0004        # 괴리 표준편차 (지수 대비 비율)
WIDEN_START, WIDEN_END = 0.01, 0.10   # 스프레드 확대 국면 진입·이탈 확률 (일간)


def syn_code(kind, year, month):
    """합성 월물 코드. 실제 코드와 섞이지 않도록 SR/SM 접두사를 쓴다. 예) SR202512, SM202511"""
    return f"{'SR' if kind == 'reg' else 'SM'}{year}{month:02d}"


def syn_pair_for(year, month):
    return syn_code("reg", year, month), syn_code("mini", year, month - 1)


def _round_tick(x, tick):
    return np.round(np.round(x / tick) * tick, 2)


def generate(start="2020-01-01", years=6, s0=300.0, seed=0):
    """합성 데이터 생성. (월물별 일봉 dict, 날짜별 지수·국면 DataFrame)을 돌려준다."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=int(252 * years))
    n = len(days)

    # 1) 국면과 지수
    state = np.empty(n, dtype=int)
    state[0] = 2
    for t in range(1, n):
        state[t] = rng.choice(4, p=TRANSITION[state[t - 1]])
    mu = np.array([REGIMES[NAMES[s]][0] for s in state])
    sd = np.array([REGIMES[NAMES[s]][1] for s in state])
    ret = mu + sd * rng.standard_normal(n)
    ret[0] = 0.0
    index = s0 * np.exp(np.cumsum(ret))

    # 2) 스프레드 확대 국면 (지수 국면과 별개)
    widen = np.zeros(n, dtype=bool)
    for t in range(1, n):
        widen[t] = (rng.random() > WIDEN_END) if widen[t - 1] else (rng.random() < WIDEN_START)

    # 3) 보유비용: 평균 주위를 천천히 움직이는 AR(1)
    carry = np.empty(n)
    carry[0] = CARRY_MEAN
    innov = CARRY_SD * np.sqrt(1 - CARRY_PHI ** 2)
    for t in range(1, n):
        carry[t] = CARRY_MEAN + CARRY_PHI * (carry[t - 1] - CARRY_MEAN) + innov * rng.standard_normal()

    market = pd.DataFrame({"index": index, "regime": [NAMES[s] for s in state], "widen": widen,
                           "carry": carry}, index=days)

    # 4) 월물 목록: 기간을 덮는 정규 분기물과 미니 월물
    contracts = []
    for y in range(days[0].year, days[-1].year + 2):
        for m in range(1, 13):
            exp = pd.Timestamp(expiry(y, m))
            if exp < days[0] or exp > days[-1] + pd.Timedelta(days=200):
                continue
            contracts.append(("mini", y, m, exp))
            if m in (3, 6, 9, 12):
                contracts.append(("reg", y, m, exp))

    out = {}
    for kind, y, m, exp in contracts:
        listed = exp - pd.Timedelta(days=370 if kind == "reg" else 190)
        live = (days >= listed) & (days <= exp)
        if live.sum() < 5:
            continue
        d = days[live]
        tau = (exp - d).days.to_numpy() / 365.0
        s = index[live]
        w = widen[live]

        # AR(1) 괴리. 확대 국면에서는 크기 3배, 정규선물은 위쪽으로 치우침
        e = np.zeros(len(d))
        for t in range(1, len(d)):
            scale = NOISE_SD * s[t] * (3.0 if w[t] else 1.0)
            e[t] = NOISE_PHI * e[t - 1] + scale * rng.standard_normal()
        if kind == "reg":
            e += np.where(w, 0.002 * s, 0.0)
        close = s * np.exp(carry[live] * tau) + e

        tick = REG_TICK if kind == "reg" else MINI_TICK
        prev = np.concatenate([[close[0]], close[:-1]])
        day_sd = sd[live] * s
        opn = prev + 0.3 * day_sd * rng.standard_normal(len(d))
        high = np.maximum(opn, close) + np.abs(0.5 * day_sd * rng.standard_normal(len(d)))
        low = np.minimum(opn, close) - np.abs(0.5 * day_sd * rng.standard_normal(len(d)))

        # 거래량: 잔존일이 짧을수록 많고, 먼 월물은 0인 날이 많다
        dte = (exp - d).days.to_numpy()
        base = 150_000 if kind == "reg" else 400_000
        lam = base * np.exp(-dte / (45 if kind == "reg" else 25))
        volume = rng.poisson(np.maximum(lam, 0.3))

        code = syn_code(kind, y, m)
        out[code] = pd.DataFrame({
            "code": code, "date": d.strftime("%Y%m%d"),
            "open": _round_tick(opn, tick), "high": _round_tick(high, tick),
            "low": _round_tick(low, tick), "close": _round_tick(close, tick),
            "volume": volume, "value": 0,
        })
    return out, market


def save(frames, market, root="data/synthetic"):
    os.makedirs(os.path.join(root, "daily"), exist_ok=True)
    for code, df in frames.items():
        df.to_parquet(os.path.join(root, "daily", f"{code}.parquet"), index=False)
    market.to_parquet(os.path.join(root, "market.parquet"))
    return root
