"""03-3 실습: 종목 마스터, 일봉, 분봉 조회.

실행:  python ch03/query_examples.py
"""
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab.kis import KISClient, download_master  # noqa: E402

# 1) 상장 중인 월물 확인
master = download_master()
reg = [r for r in master if r[0] == "1"]     # 코스피200 정규선물
mini = [r for r in master if r[0] == "B"]    # 미니 코스피200 선물
print("상장 정규선물:", [f"{r[1]}({r[3]})" for r in reg])
print("상장 미니선물:", [f"{r[1]}({r[3]})" for r in mini])

client = KISClient(mode="paper")
near = reg[0][1]

# 2) 근월물 일봉 최근 5일
start = (date.today() - timedelta(days=14)).strftime("%Y%m%d")
rows = client.daily(near, start)
print(f"\n{near} 일봉 최근 {min(5, len(rows))}일")
for x in rows[-5:]:
    print(f"  {x['stck_bsop_date']}  시 {x['futs_oprc']:>8} 고 {x['futs_hgpr']:>8} "
          f"저 {x['futs_lwpr']:>8} 종 {x['futs_prpr']:>8}  거래량 {int(x['acml_vol']):>7,}")

# 3) 신·구 코드 체계: 같은 2025년 12월물을 두 방식으로 조회
for code in ("101W12", "A01512"):
    rows = client.daily(code, "20251201", "20251211")
    print(f"\n{code}: {len(rows)}건" + (f", 마지막 {rows[-1]['stck_bsop_date']} 종가 {rows[-1]['futs_prpr']}"
                                     if rows else "  ← 오류 없이 빈 결과"))

# 4) 1분봉: 가장 최근 거래일 하루치
day = date.today()
while True:
    day -= timedelta(days=1)
    if day.weekday() < 5:
        bars = client.minute_day(near, day.strftime("%Y%m%d"))
        if bars:
            break
print(f"\n{near} {day} 1분봉 {len(bars)}개: {bars[0]['stck_cntg_hour']} ~ {bars[-1]['stck_cntg_hour']}")

# 5) 분봉 보존 기간: 13개월 전 날짜는 비어 있다
old = (date.today() - timedelta(days=400)).strftime("%Y%m%d")
print(f"{old} 1분봉 (구 코드 101W12): {len(client.minute_day('101W12', old))}개")
print(f"\nAPI 호출 {client.calls}회")
