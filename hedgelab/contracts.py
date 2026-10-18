"""월물 코드와 만기일 계산.

코드 체계 (2026-10 실측):
- 2025년 이전 만기: 구 코드  101/105 + 연도문자 + 월   예) 101W12 = 정규 2025-12
- 2026년 이후 만기: 신 코드  A + 01/05 + 연도 끝자리 + 월  예) A01612 = 정규 2026-12
"""
from datetime import date, timedelta

# 구 코드 연도문자. I·O·U는 쓰지 않는다 (2017~2025 실측 확인)
OLD_YEAR = {2015: "K", 2016: "L", 2017: "M", 2018: "N", 2019: "P", 2020: "Q",
            2021: "R", 2022: "S", 2023: "T", 2024: "V", 2025: "W"}

REG_MONTHS = (3, 6, 9, 12)


def contract_code(kind, year, month):
    """kind: 'reg'(정규) 또는 'mini'(미니)."""
    if kind not in ("reg", "mini"):
        raise ValueError("kind 는 'reg' 또는 'mini'")
    if kind == "reg" and month not in REG_MONTHS:
        raise ValueError("정규선물은 3·6·9·12월물만 있습니다")
    if year >= 2026:
        return f"A0{'1' if kind == 'reg' else '5'}{year % 10}{month:02d}"
    return f"10{'1' if kind == 'reg' else '5'}{OLD_YEAR[year]}{month:02d}"


def expiry(year, month):
    """최종거래일 근사값: 결제월 둘째 목요일.

    실제로는 공휴일이면 앞당겨지므로, 정확한 날짜는 시세 데이터의 마지막 거래일로 확인한다.
    """
    d = date(year, month, 1)
    d += timedelta(days=(3 - d.weekday()) % 7)   # 첫째 목요일
    return d + timedelta(days=7)


def pair_for(year, month):
    """이 책의 표준 쌍: 정규 m월물 + 미니 m−1월물. (정규 코드, 미니 코드)를 돌려준다."""
    if month not in REG_MONTHS:
        raise ValueError("정규선물 결제월(3·6·9·12)을 넣으세요")
    return contract_code("reg", year, month), contract_code("mini", year, month - 1)


def pairs_between(start_year, end_year):
    """start_year~end_year 사이 표준 쌍 목록 [(연, 월, 정규코드, 미니코드), ...]."""
    out = []
    for y in range(start_year, end_year + 1):
        for m in REG_MONTHS:
            if (y, m) < (2015, 9):   # 미니선물은 2015년 7월 상장
                continue
            out.append((y, m) + pair_for(y, m))
    return out
