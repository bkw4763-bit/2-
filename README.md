# 석유 최고가격 결정 도우미

산업통상부가 4주마다 정하는 석유 최고가격(휘발유·경유·등유)을 자료로 계산해 권고하는 도구입니다.
숭실대 경제학과 금융·에너지 데이터 해커톤(2026-10-10) 출품작입니다.

지금은 **자료 수집과 전처리**까지 들어 있습니다. 설계와 남은 작업은 `CLAUDE.md` 에 있습니다.

## 준비

```bash
pip install -r requirements.txt
cp .env.example .env      # 윈도우: copy .env.example .env
```

`.env` 를 열어 발급받은 키 세 개를 넣습니다.

| 변수 | 발급처 |
|---|---|
| `ECOS_API_KEY` | 한국은행 ECOS |
| `KOSIS_API_KEY` | 통계청 KOSIS |
| `OPINET_API_KEY` | 오피넷 유가정보 API |

## 실행

```bash
python test_parsers.py            # 키·인터넷 없이 코드 점검
python run_pipeline.py            # 수집 → 전처리
python run_pipeline.py --offline  # 저장본만 사용 (인터넷 장애 대비)
python counterfactual_check.py    # "없었다면" 모형의 근거 점검과 그림
```

결과는 `data/processed` 에 저장됩니다.

## input 폴더의 파일

오피넷에서 내려받은 과거 자료입니다. 대회 당일 아침에 다시 받아 덮어씁니다.

| 파일 | 오피넷 메뉴 | 설정 |
|---|---|---|
| `retail_daily.csv` | 국내유가통계 > 주유소 > 평균판매가격 > 제품별 | 일간 |
| `supply_weekly_pretax.csv` | 국내유가통계 > 정유사 > 주간공급가격 > 제품별 | 주간, 세전 |
| `supply_weekly_aftertax.csv` | 같은 화면 | 주간, 세후 |
| `intl_daily.csv` | 유가관련정보 > 국제유가 > 석유제품 | 일간, 단위 `$` |

기간은 2015년 1월부터, 제품은 보통휘발유·자동차용경유·실내등유(국제가는 휘발유 92RON·경유 0.001%·등유)입니다.
