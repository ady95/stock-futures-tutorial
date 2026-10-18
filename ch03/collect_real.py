"""03-4 실습 1: 표준 쌍(정규 m월물 + 미니 m−1월물)의 일봉을 모두 받아 저장.

4장부터 쓰는 롤오버용 미니 m월물(정규와 같은 날 만기)도 함께 받는다.

실행:  python ch03/collect_real.py [시작연도] [끝연도]      기본 2015 2025
결과:  data/real/daily/<코드>.parquet  (이미 있는 만기 지난 월물은 건너뜀)

원본 시장 데이터는 재배포하지 마세요. 이 저장소에도 커밋되지 않습니다(.gitignore).
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab.contracts import contract_code, expiry, pairs_between  # noqa: E402
from hedgelab.data import save_daily, to_frame  # noqa: E402
from hedgelab.kis import KISClient  # noqa: E402

start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
end_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
root = "data/real"

client = KISClient(mode="paper")
for y, m, reg_code, mini_code in pairs_between(start_year, end_year):
    roll_code = contract_code("mini", y, m)
    for code, (cy, cm) in ((reg_code, (y, m)), (mini_code, (y, m - 1)), (roll_code, (y, m))):
        path = os.path.join(root, "daily", f"{code}.parquet")
        if os.path.exists(path) and expiry(cy, cm) < date.today():
            continue
        # 정규는 최장 3년 전 상장 → 만기 3년 전부터 조회
        rows = client.daily(code, f"{cy - 3}0101")
        if not rows:
            print(f"{code}: 0건 (코드·기간 확인)")
            continue
        save_daily(to_frame(rows, code), root, code)
        print(f"{code}: {len(rows)}건 {rows[0]['stck_bsop_date']}~{rows[-1]['stck_bsop_date']}")
print(f"완료 — API 호출 {client.calls}회")
