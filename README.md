# stock-futures-tutorial

위키독스 책 **「강화학습 기반 선물 자동매매 따라하기 — PPO로 구현하는 코스피200 선물 동적 헤지와 위험관리」**의 예제 코드 저장소입니다.

- 책: https://wikidocs.net/book/21732

코스피200 정규선물 1계약 매수 + 미니선물 0~5계약 매도 포지션에서, PPO 강화학습 에이전트가 헤지비율을 조절하도록 학습시키고 그 위험조정 성과를 검증합니다.

> **투자 위험 고지** — 이 저장소의 코드는 교육·연구 목적입니다. 어떠한 수익도 보장하지 않으며, 실제 주문 기능은 포함하지 않습니다. 실제 거래에 따른 판단과 책임은 사용자 본인에게 있습니다.

## 시작하기 (책 03-1)

```bash
git clone https://github.com/ady95/stock-futures-tutorial.git
cd stock-futures-tutorial
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python ch03/make_synthetic.py       # API 키 없이 쓸 합성 데이터 생성
```

버전은 저자가 실측한 조합(Python 3.10.18, CPU 전용)으로 고정되어 있습니다.

## 구성

```text
hedgelab/            공용 패키지
  contracts.py       월물 코드(신·구 체계)·만기일·표준 쌍
  kis.py             한국투자증권 Open API 클라이언트 (시세 조회 전용)
  data.py            Parquet 저장·불러오기, 두 월물 정렬
  stats.py           쌍 단위 통계
  synthetic.py       합성 데이터 생성기
ch02/                헤지비율 손익 시뮬레이터
ch03/                토큰 확인, 시세 조회, 수집, 정렬, 합성 데이터
```

| 장 | 내용 | 결과물 |
|---|---|---|
| 02 | 선물 손익·헤지비율 계산 | 헤지 손익 시뮬레이터 |
| 03 | 시세 수집·합성 데이터 | 데이터셋 준비 스크립트 |
| 04 | 백테스트 엔진 | B0·B1·B2 기준선 전략 (예정) |
| 05 | 강화학습 환경 | `CalendarSpreadEnv` (예정) |
| 06~07 | PPO 학습·위험관리 | PPO 에이전트, 위험관리 엔진 (예정) |
| 08 | 성능 비교·워크포워드 검증 | 비교 실험 보고서 (예정) |
| 09~10 | 실시간 시세·모의운영 | 페이퍼트레이딩, 대시보드 (예정) |

## API 키와 데이터

한국투자증권 Open API를 쓰는 실습(3장, 9장)은 `.env.example`을 `.env`로 복사해 모의투자 앱키를 넣습니다. `.env`, 토큰 캐시(`.kis/`), 수집 데이터(`data/`)는 git에 올라가지 않습니다.

원본 시장 데이터는 재배포 조건 때문에 이 저장소에 포함하지 않습니다. 수집 코드로 직접 받거나 합성 데이터를 사용하세요.
