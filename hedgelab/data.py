"""시세 저장·불러오기와 두 월물 정렬.

저장 형식 (Parquet):
  data/real/daily/<코드>.parquet       일봉   [code, date, open, high, low, close, volume, value]
  data/real/minute/<코드>/<날짜>.parquet 1분봉 [code, date, time, open, high, low, close, volume, value]
합성 데이터는 data/synthetic/ 아래에 같은 형식으로 둔다 (03-5).
"""
import glob
import os

import pandas as pd

RENAME = {"stck_bsop_date": "date", "stck_cntg_hour": "time",
          "futs_oprc": "open", "futs_hgpr": "high", "futs_lwpr": "low", "futs_prpr": "close",
          "acml_vol": "volume", "cntg_vol": "volume", "acml_tr_pbmn": "value"}
COLUMNS = ["code", "date", "time", "open", "high", "low", "close", "volume", "value"]


def to_frame(rows, code):
    """KIS 응답(output2 목록)을 표준 컬럼의 DataFrame 으로 바꾼다."""
    df = pd.DataFrame(rows).rename(columns=RENAME)
    df.insert(0, "code", code)
    for c in ("open", "high", "low", "close", "volume", "value"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[[c for c in COLUMNS if c in df]]


def save_daily(df, root, code):
    path = os.path.join(root, "daily", f"{code}.parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_daily(root, code):
    """일봉을 날짜 인덱스 DataFrame 으로. 파일이 없거나 비어 있으면 None."""
    path = os.path.join(root, "daily", f"{code}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"].astype(str))
    return df.set_index("date").sort_index()


def load_minute(root, code):
    """1분봉 전체를 타임스탬프 인덱스 DataFrame 으로. 없으면 None."""
    files = sorted(glob.glob(os.path.join(root, "minute", code, "*.parquet")))
    frames = [f for f in (pd.read_parquet(p) for p in files) if len(f)]
    if not frames:
        return None
    df = pd.concat(frames)
    df["ts"] = pd.to_datetime(df["date"].astype(str) + df["time"].astype(str), format="%Y%m%d%H%M%S")
    return df.set_index("ts").sort_index()


def align_pair(reg, mini, min_volume=1):
    """정규·미니 두 시계열을 같은 시점끼리 맞춘다.

    - 두 쪽 모두 있는 시점만 남긴다 (inner join)
    - 거래량이 min_volume 미만인 시점은 버린다 — 거래가 없으면 가격이 마지막 체결에 멈춰 있어
      스프레드가 실제보다 튀어 보이기 때문이다
    반환 컬럼: reg, mini, spread(= reg − mini), vol_reg, vol_mini
    """
    j = reg[["close", "volume"]].join(mini[["close", "volume"]], lsuffix="_reg", rsuffix="_mini", how="inner")
    j = j[(j.volume_reg >= min_volume) & (j.volume_mini >= min_volume)]
    out = pd.DataFrame({"reg": j.close_reg, "mini": j.close_mini,
                        "vol_reg": j.volume_reg, "vol_mini": j.volume_mini})
    out["spread"] = out.reg - out.mini
    return out


def hedge_window(aligned, mini_expiry, days=45):
    """미니 만기 전 days일(달력 기준) 구간만 자른다 — 이 책의 '헤지 구간'."""
    start = pd.Timestamp(mini_expiry) - pd.Timedelta(days=days)
    return aligned[(aligned.index >= start) & (aligned.index <= pd.Timestamp(mini_expiry))]
