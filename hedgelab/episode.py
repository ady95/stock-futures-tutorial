"""에피소드 구성 (4장~).

한 에피소드 = 표준 쌍 하나의 거래 기간.
- 시작: 미니 m−1월물(a) 만기 전 window_days일 (두 월물 모두 거래가 있는 첫날)
- 끝:   정규 m월물 만기일
- 미니 매도 다리는 항상 '현재 근월 미니'다. a가 만기되면 m월물(b)로 넘어간다.
  b는 정규와 같은 날 만기되므로, 공백 구간에도 b로 다시 헤지할 수 있다.

반환 DataFrame (날짜 인덱스)
  reg, reg_high, reg_low      정규 m월물 종가·고가·저가
  mini_a, mini_b              미니 m−1월물, m월물 종가 (없는 날은 NaN)
  near                        그날 장 마감 기준 근월 미니 ('a' 또는 'b')
  a_last                      a의 마지막 거래일(최종결제) 여부
"""
import numpy as np
import pandas as pd

from .contracts import contract_code, pair_for
from .data import load_daily
from .synthetic import syn_code, syn_pair_for


def episode_codes(year, month, synthetic=False):
    if synthetic:
        reg, a = syn_pair_for(year, month)
        return reg, a, syn_code("mini", year, month)
    reg, a = pair_for(year, month)
    return reg, a, contract_code("mini", year, month)


def build_episode(root, year, month, synthetic=False, window_days=45, min_volume=1):
    """표준 쌍 하나의 에피소드 데이터를 만든다. 데이터가 모자라면 None."""
    reg_code, a_code, b_code = episode_codes(year, month, synthetic)
    reg, a, b = (load_daily(root, c) for c in (reg_code, a_code, b_code))
    if reg is None or a is None or b is None:
        return None
    a_exp, reg_exp = a.index.max(), reg.index.max()
    if reg_exp <= a_exp:
        return None

    days = reg.index[(reg.index >= a_exp - pd.Timedelta(days=window_days)) & (reg.index <= reg_exp)]
    ep = pd.DataFrame(index=days)
    ep["reg"], ep["reg_high"], ep["reg_low"] = reg.close, reg.high, reg.low
    ep["mini_a"], ep["mini_b"] = a.close.reindex(days), b.close.reindex(days)
    ep["near"] = np.where(ep.index <= a_exp, "a", "b")
    ep["a_last"] = ep.index == a_exp

    # 거래가 없는 날(낡은 가격)은 버린다: 정규, 그리고 그날의 근월 미니
    vol_reg = reg.volume.reindex(days).fillna(0)
    vol_near = np.where(ep.near == "a", a.volume.reindex(days).fillna(0), b.volume.reindex(days).fillna(0))
    ok = (vol_reg >= min_volume) & (vol_near >= min_volume)
    ok &= ep.mini_b.notna() | (ep.near == "a")
    ok |= ep.a_last                       # a 최종결제일은 반드시 남긴다
    ok.iloc[-1] = True                    # 정규 만기일도
    ep = ep[ok]
    # a 만기일에는 b 가격이 있어야 롤오버할 수 있다
    if ep.loc[ep.a_last, "mini_b"].isna().any():
        return None
    ep.attrs.update(year=year, month=month, reg_code=reg_code, a_code=a_code, b_code=b_code)
    return ep


def build_all(root, start_year, end_year, synthetic=False, **kw):
    """기간 안의 모든 에피소드 목록."""
    out = []
    for y in range(start_year, end_year + 1):
        for m in (3, 6, 9, 12):
            ep = build_episode(root, y, m, synthetic, **kw)
            if ep is not None and len(ep) > 10:
                out.append(ep)
    return out
