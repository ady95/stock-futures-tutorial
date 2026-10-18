"""03-2 실습: KIS Open API 접근 토큰 발급 확인.

실행:  python ch03/check_token.py
준비:  저장소 루트의 .env 에 KIS_PAPER_APP_KEY, KIS_PAPER_APP_SECRET
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab.kis import KISClient  # noqa: E402

client = KISClient(mode="paper")
token = client.token()
with open(client.token_file, encoding="utf-8") as f:
    cached = json.load(f)

print(f"도메인: {client.base}")
print(f"토큰 길이: {len(token)}자 (앞 8자만 표시: {token[:8]}...)")
print(f"남은 유효시간: {(cached['expires_at'] - time.time()) / 3600:.1f}시간")
print(f"캐시 파일: {client.token_file}")
