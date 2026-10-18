"""한국투자증권 Open API 클라이언트 (국내 지수선물 시세 조회 전용).

- 시세 조회는 모의투자 앱키로 충분하다 (2026-10 실측)
- 키는 .env 에서 읽는다. 코드·노트북에 직접 쓰지 않는다
- 접근 토큰은 파일에 캐시해 재사용한다
"""
import io
import json
import os
import time
import zipfile
from datetime import date, datetime, timedelta

import requests

DOMAINS = {
    "paper": "https://openapivts.koreainvestment.com:29443",   # 모의투자
    "real": "https://openapi.koreainvestment.com:9443",        # 실전투자
}
# 모의투자 도메인은 0.6초 간격에서 EGW00201(초당 거래건수 초과)이 났다. 1.1초면 안정적
MIN_INTERVAL = {"paper": 1.1, "real": 0.1}
MASTER_URL = "https://new.real.download.dws.co.kr/common/master/fo_idx_code_mts.mst.zip"

PRICE_PATH = "/uapi/domestic-futureoption/v1/quotations/inquire-price"
DAILY_PATH = "/uapi/domestic-futureoption/v1/quotations/inquire-daily-fuopchartprice"
MINUTE_PATH = "/uapi/domestic-futureoption/v1/quotations/inquire-time-fuopchartprice"


def load_env(path=".env"):
    """KEY=VALUE 형식의 .env 파일을 dict 로 읽는다."""
    env = {}
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


class KISError(RuntimeError):
    pass


class KISClient:
    def __init__(self, mode="paper", env_path=None, token_dir=".kis"):
        env_path = env_path or os.environ.get("KIS_ENV_FILE", ".env")
        env = load_env(env_path)
        prefix = "KIS_PAPER_" if mode == "paper" else "KIS_REAL_"
        self.app_key = env[prefix + "APP_KEY"]
        self.app_secret = env[prefix + "APP_SECRET"]
        self.mode = mode
        self.base = DOMAINS[mode]
        os.makedirs(token_dir, exist_ok=True)
        self.token_file = os.path.join(token_dir, f"token_{mode}.json")
        self._last_call = 0.0
        self.calls = 0

    # ---------------------------------------------------------------- 인증
    def token(self):
        """캐시된 토큰이 10분 이상 남아 있으면 재사용하고, 아니면 새로 발급한다."""
        if os.path.exists(self.token_file):
            with open(self.token_file, encoding="utf-8") as f:
                cached = json.load(f)
            if cached["expires_at"] > time.time() + 600:
                return cached["access_token"]
        r = requests.post(self.base + "/oauth2/tokenP", timeout=10, json={
            "grant_type": "client_credentials", "appkey": self.app_key, "appsecret": self.app_secret})
        body = r.json()
        if "access_token" not in body:
            raise KISError(f"토큰 발급 실패: HTTP {r.status_code} {body.get('error_code', '')} "
                           f"{body.get('error_description', '')}")
        with open(self.token_file, "w", encoding="utf-8") as f:
            json.dump({"access_token": body["access_token"],
                       "expires_at": time.time() + int(body.get("expires_in", 86400))}, f)
        return body["access_token"]

    # ---------------------------------------------------------------- 공통 호출
    def get(self, path, tr_id, params, retries=4):
        """GET 호출. 호출 간격을 지키고, 초당 건수 초과 같은 일시 오류는 재시도한다."""
        for attempt in range(retries + 1):
            wait = MIN_INTERVAL[self.mode] - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            headers = {
                "authorization": "Bearer " + self.token(),
                "appkey": self.app_key, "appsecret": self.app_secret,
                "tr_id": tr_id, "custtype": "P", "content-type": "application/json; charset=utf-8",
            }
            r = requests.get(self.base + path, headers=headers, params=params, timeout=20)
            self._last_call = time.time()
            self.calls += 1
            body = r.json()
            if body.get("rt_cd") == "0":
                return body
            if body.get("msg_cd") == "EGW00123" and os.path.exists(self.token_file):
                os.remove(self.token_file)          # 토큰 만료 → 다음 시도에서 재발급
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
        raise KISError(f"{tr_id} 실패: {body.get('msg_cd')} {body.get('msg1', '').strip()}")

    # ---------------------------------------------------------------- 시세
    def price(self, code):
        """현재가 (9장). 모의투자 도메인에서도 동작한다. output1 dict 를 그대로 돌려준다."""
        body = self.get(PRICE_PATH, "FHMIF10000000", {"FID_COND_MRKT_DIV_CODE": "F", "FID_INPUT_ISCD": code})
        return body.get("output1") or {}

    def daily(self, code, start, end=None):
        """일봉. start/end 는 'YYYYMMDD'. 100건 제한을 넘으면 종료일을 당기며 이어 받는다.

        주의: 종목코드가 틀려도 오류 없이 빈 리스트가 돌아온다 (rt_cd=0).
        """
        end = end or date.today().strftime("%Y%m%d")
        rows = {}
        while True:
            body = self.get(DAILY_PATH, "FHKIF03020100", {
                "FID_COND_MRKT_DIV_CODE": "F", "FID_INPUT_ISCD": code,
                "FID_INPUT_DATE_1": start, "FID_INPUT_DATE_2": end, "FID_PERIOD_DIV_CODE": "D"})
            page = [x for x in body.get("output2") or [] if x.get("stck_bsop_date")]
            for x in page:
                rows[x["stck_bsop_date"]] = x
            if len(page) < 100:
                break
            oldest = min(x["stck_bsop_date"] for x in page)
            if oldest <= start:
                break
            end = (datetime.strptime(oldest, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
        return [rows[k] for k in sorted(rows)]

    def minute_day(self, code, day, bar_seconds=60):
        """하루치 분봉 ('YYYYMMDD'). 한 번에 102건까지라 시각을 당기며 이어 받는다.

        KIS 분봉은 최근 약 12개월만 조회된다 (2026-10 실측). 그보다 오래된 날짜는 빈 리스트.
        """
        hour, out = "154500", {}
        for _ in range(12):
            body = self.get(MINUTE_PATH, "FHKIF03020200", {
                "FID_COND_MRKT_DIV_CODE": "F", "FID_INPUT_ISCD": code,
                "FID_HOUR_CLS_CODE": str(bar_seconds), "FID_PW_DATA_INCU_YN": "Y",
                "FID_FAKE_TICK_INCU_YN": "N", "FID_INPUT_DATE_1": day, "FID_INPUT_HOUR_1": hour})
            rows = [x for x in body.get("output2") or [] if x.get("stck_bsop_date")]
            today = [x for x in rows if x["stck_bsop_date"] == day]
            for x in today:
                out[x["stck_cntg_hour"]] = x
            if not today or len(today) < len(rows):
                break                                   # 전날 봉이 섞이면 하루의 시작에 도달
            first = min(x["stck_cntg_hour"] for x in today)
            if first <= "084500":
                break
            hour = (datetime.strptime(first, "%H%M%S") - timedelta(seconds=bar_seconds)).strftime("%H%M%S")
        return [out[k] for k in sorted(out)]


def download_master():
    """지수선물옵션 종목 마스터를 내려받아 [상품종류, 단축코드, 표준코드, 종목명] 목록으로 돌려준다.

    상품종류 '1' = 코스피200 정규선물, 'B' = 미니 코스피200 선물.
    """
    raw = requests.get(MASTER_URL, timeout=30).content
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0]).decode("cp949")
    return [[c.strip() for c in line.split("|")[:4]] for line in text.splitlines() if line.strip()]
