"""최종 개별 예측 파이프라인 설명과 프로젝트 코드 전체를 독립 HTML로 보존한다."""
import ast
import base64
import csv
import hashlib
import html
import json
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "pipeline_report.html"
FINAL = ROOT / "딥러닝/output/compare_years_regression_data"
JOINT = ROOT / "딥러닝/output/compare_joint_regression_data"


def esc(value):
    return html.escape(str(value), quote=True)


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(x)}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{esc(x)}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def source_files():
    paths = list(ROOT.glob("*.py")) + list((ROOT / "딥러닝").glob("*.py"))
    paths += [ROOT / "README.md", ROOT / "딥러닝/README.md", ROOT / "requirements.txt", ROOT / "caps.csv",
              ROOT / "모델/timesfm-3.0-pytorch/config.json", ROOT / "모델/timesfm-3.0-pytorch/README.md"]
    paths += list((ROOT / "딥러닝").glob("requirements*.txt"))
    return sorted(set(p for p in paths if p.is_file()))


def role(path):
    if path.name.startswith("test"):
        return "검증 코드"
    if path.name == "compare_timesfm_joint.py":
        return "공동 예측 비교 실험 · 최종 미채택"
    if path.parent == ROOT:
        return "공통 수집·전처리·회귀·진단 또는 참고 설정"
    if path.name in {"run_regression_timesfm.py", "compare_timesfm_years.py", "analyze_policy.py"}:
        return "최종 개별 TimesFM 경로"
    if path.name == "build_pipeline_html.py":
        return "HTML 생성기"
    return "별도/이전 실험 또는 참고 설정 · 최종 진입점에서 미호출"


def build():
    sections = []
    def add(key, title, content):
        sections.append((key, title, f'<section id="{key}"><h2>{title}</h2>{content}</section>'))
    add("decision", "01 · 연구 질문과 최종 결정", '''
<div class="notice"><b>최종 채택: 유종별 개별 TimesFM 예측</b><br>휘발유·경유·등유를 각각 예측한다. 공동 방식은 비교 실험으로 보존한다.</div>
<p>연구 질문은 <b>가격상한제가 없었을 때 정유사 공급가격이 얼마였을까</b>이다.
타겟은 주유소 판매가가 아닌 <code>{fuel}_supply_pretax</code>, 정유사 세전 주간 공급가격(원/L)이다.
회귀의 현재 채택 모형은 <code>recent_margin</code>이고 기본 OLS도 비교한다.</p>
<p>이 문서는 현재 실행 코드와 저장 결과를 정리한 스냅샷이다. 설명에 이어 모든 프로젝트 Python 원문을 수록하므로
주석·docstring·문자열·함수·조건 분기까지 생략하지 않는다. 이전 실험 코드와 오래된 README도 원문 그대로 보존하되 현재 경로와 구분한다.
외부 라이브러리의 구현과 모델 가중치 바이너리는 프로젝트 코드 수록 범위에 포함하지 않는다.</p>''')
    add("flow", "02 · 전체 실행 흐름", '''
<div class="flow"><div>① 수집<small>collect.py · API/파일/캐시</small></div><div>② 공통 전처리<small>prepare.py · weekly_panel.csv</small></div><div>③ 모델 비교<small>회귀 2종 + 개별 TimesFM</small></div><div>④ 정책 경로<small>43주 · 세후 환산·상한 격차</small></div><div>⑤ 민감도<small>국제가격 −10% / 유지 / +10%</small></div></div>
<p><code>run_pipeline.py</code>는 수집 → 전처리 → 회귀를 실행한다.
<code>compare_timesfm_years.py</code>는 공통 주간 자료로 개별 TimesFM과 두 회귀를 동일한 기점에서 비교한다.
<code>analyze_policy.py</code>는 고정 설정으로 정책 경로·그림·민감도를 만든다.</p>
<pre>python -X utf8 run_pipeline.py --offline
.venv-timesfm/Scripts/python.exe -X utf8 딥러닝/compare_timesfm_years.py
.venv-timesfm/Scripts/python.exe -X utf8 딥러닝/analyze_policy.py

# 고정 설정으로 정책 경로만 갱신: 자료 SHA256 일치 필요
.venv-timesfm/Scripts/python.exe -X utf8 딥러닝/compare_timesfm_years.py --policy-only

# 선택적 회귀 진단 / 미채택 공동 예측 비교
python -X utf8 counterfactual_check.py
.venv-timesfm/Scripts/python.exe -X utf8 딥러닝/compare_timesfm_joint.py</pre>
<p>--offline은 API 호출을 생략하고 저장본을 사용한다. 파일 입력의 처리 분기는 collect.py 원문을 따른다.
API 키 이름과 로딩 방식은 config.py에 있으며 실제 .env/키 값은 이 HTML에 포함하지 않는다.</p>''')
    add("data", "03 · 자료와 실제 모델 입력", table(["자료", "역할", "직접 모델 입력"], [
        ["정유사 세전 공급가 · 주간", "공통 타겟", "회귀 y / TimesFM 과거 타겟"],
        ["국제 석유제품가격 · 일간 USD/bbl", "유종별 국제 제품가격", "원/L 환산 후 입력"],
        ["ECOS 원/달러 환율 · 일간", "국제가격 환산", "국제가격에 반영; 별도 feature 아님"],
        ["정유사 세후 공급가 · 주간", "유류세 역산·실제값 대조", "세전 예측 feature 아님"],
        ["caps.csv 상한·시행일·검증 표시", "차수별 상한 경로·격차", "feature 아님"],
        ["주유소 판매가", "최신값·차수 요약", "현재 타겟/feature 아님"],
        ["FRED 브렌트 / ECOS 물가·금리·두바이 / KOSIS 유종 물가", "보조 분석·월별 표", "현재 회귀/TimesFM feature 아님"],
        ["collect_fx.py", "별도 환율 수집 도구", "현재 prepare.py는 ecos_fx_krw_usd 사용"],
    ]) + '''<p>설정 시작일은 2015-01-01, 시행일은 2026-03-13, 수집 종료일은 실행 당시 cfg.TODAY이다.
오피넷 CSV와 ECOS·KOSIS·FRED·오피넷 API를 수집한다. 실제 마지막 관측일·지연일·실패·캐시 상태를 보고한다.
각 API의 통계 코드·품목 코드·재시도·페이지·기간·파싱·중복 병합 로직은 원문에 모두 보존한다.</p>''')
    add("prepare", "04 · 공통 전처리·주간화·결측 처리", '''
<ol><li>일간 D 달력을 만들고 날짜 중복은 마지막 값을 사용한다.</li>
<li>국제가격·환율·브렌트는 최대 7일, 주유소 판매가는 최대 3일 직전 값으로 채운다. 제한을 넘거나 자료가 없는 결측은 남긴다.
현재 공통 경로는 추가 KNN을 사용하지 않는다. 이전 prepare_weekly.py의 KNN과 구분한다.</li>
<li>국제 제품가격을 <code>USD/bbl ÷ 158.987 × KRW/USD</code>로 일별 환산한 뒤 주평균을 낸다.
달러가격 주평균 × 환율 주평균을 계산하는 방식과 다르다.</li>
<li>일~토 W-SAT 주평균을 만든다. 보간된 주말·휴일도 달력 평균에 포함된다.
공급가격은 원래 주간 관측값을 결합한다. days는 달력상 해당 주 포함일수로 유효 관측일수와 다르다.</li>
<li>유류세(부가세 포함)는 <code>세후 − 1.1 × 세전</code>으로 역산한다.
일별 세금은 주 종료일 값에서 최대 6일 bfill 후 ffill한다. 주중 세율 변경일과 어긋날 수 있다.</li>
<li>상한·차수는 시작일부터 다음 시작일 전까지 계단형으로 유지한다. 주간 상한 적용일수도 계산한다.
주간 in_policy는 주 종료일≥시행일로 판단한다.</li>
<li>공급가가 있는 주는 원자료의 역산 tax를 우선 사용하고 공급가가 없는 주는 일별 tax 평균을 사용한다.</li></ol>
<p>daily_panel·weekly_panel·monthly_panel·rounds·latest를 저장한다.
월평균 표에는 물가·금리·두바이유를 결합한다. 차수 표는 직전 14일·해당 차수 기간·차수 안의 완전 주를 요약한다.
최신값의 기준일은 보간 달력이 아닌 원자료의 마지막 관측일이다.</p>''')
    add("regression", "05 · 회귀: 기본 OLS와 최근 마진", '''
<div class="formula">기본 OLS: yₜ = a + b₀xₜ + b₁xₜ₋₁ + b₂xₜ₋₂</div>
<p>x는 해당 유종 국제가격(원/L), y는 세전 공급가격이다. 상수항과 lag0·lag1·lag2를 NumPy lstsq로 추정한다.
정책 전이며 y와 세 시차가 모두 있는 행을 사용하고 최소 52개 학습 행이 필요하다.</p>
<div class="formula">최근 마진: wₖ=bₖ/Σb · m=최근 52주 평균(y−Xw) · ŷ=m+Xw</div>
<p>현재 CHOSEN=recent_margin이다. 시차 비중은 전체 유효 과거로 추정하고 합이 1이 되도록 정규화한다.
마진만 최근 52주 평균으로 정한다. 코드에는 비음수 계수 제약·계수합 0에 대한 별도 보호가 없다.
타겟 AR·달력 계절성·추세항은 별도 feature로 넣지 않는다.</p>
<p>정책 반사실 가격은 시행 전 전체 유효 자료로 적합한다. 기본 OLS는 비교용이다.
세후 가격은 <code>세전 예측 × 1.1 + tax</code>이다.</p>
<p>회귀 90% 구간은 2018년부터 13주마다 32주 경로를 맞힌 backtest 오차(예측−실제)의 q05·q95를 쓴다.
하한=예측−q95, 상한=예측−q05이다. 이 구간 보정 경로와 80:20 성능 비교는 별개이다.</p>
<p>원본 counterfactual.csv는 주간 패널 범위만 출력한다. 연말 연장 회귀 경로는 analyze_policy.py가 만든다.
provisional은 정책 주의 세후 공급가가 세 유종 모두 결측인 경우이다.
gap_table은 미시행 세후 공급가−상한과 잠정 표시를 만든다.</p>''')
    add("timesfm", "06 · 개별 TimesFM: 최종 실행 경로", '''
<ol><li><code>compare_timesfm_years.py → main(fixed_context_years=15)</code>를 실행한다.
run_regression_timesfm.py를 직접 실행하면 기본 후보 52·156·전체에서 선택하는 별도 경로이다.</li>
<li>각 기점 이전의 세전 공급가와 국제가격이 모두 있는 이력을 최대 780주 사용한다.
부족하면 확보된 과거 전체를 사용하고 actual_context_weeks에 실제 길이를 남긴다. 최적 길이를 선택하지 않는다.</li>
<li>유종별 <code>contexts=[과거 공급가 1차원 배열]</code>와
<code>past_future_covariates=[국제가격 (1, 과거+예측 길이)]</code>를 입력한다.
미래 공급가는 입력하지 않는다. 예측 구간의 실제 국제가격을 조건부 입력한다.</li>
<li>주간 연속성·예측 길이·유한값·출력 형태를 검사한다. 결측 타겟을 채우지 않는다.
내부 결측 주를 제거한 뒤 달력이 끊기면 오류를 낸다.</li>
<li>로컬 timesfm-3.0-pytorch, CPU, batch=1, torch threads=4, local_files_only=True,
inference_mode를 사용한다. 모델 가중치 갱신·파인튜닝은 없다.</li>
<li>return_quantiles=True, use_symmetric_averaging=False이다. 점예측은 중앙값이며 q10~q90을 저장한다.</li></ol>
<p>체크포인트의 input patch=32·output patch=64이며 내부 Iterative CPM RevIN·선형 추세 제거·stitching·variate attention이 켜져 있다.
해당 유종의 과거 흐름과 국제가격 관계를 활용한다. 다른 유종은 개별 예측에 입력하지 않는다.
회귀의 수동 국제가격 시차 feature와 달리 과거 타겟 자체도 사용한다. 정확한 모델 설정은 아래 config.json 전체를 참조한다.</p>''')
    add("cv", "07 · 동일한 80:20·rolling-origin 검증", '''
<p>정책 전 완전 주(days=7)에 대해 세 유종 타겟·국제가격의 결측과 날짜 연속성을 검사한다.
마지막 ceil(N×20%)를 테스트, 앞부분을 개발로 나눈다. 고정 15년이므로 최적 입력 길이를 찾지 않는다.</p>
<div class="flow"><div>개발 80%<small>최소 52주 이력 후 검증<br>정답 32주는 개발 안에서 종료</small></div><div>테스트 20%<small>13주마다 기점 이동<br>각 기점에서 다음 32주 예측</small></div></div>
<p>두 회귀와 개별 TimesFM은 동일 기점·정답을 비교한다. 각 기점 이전에 관측된 테스트 타겟은 이후 기점의 과거 입력에 포함된다.
회귀는 매 기점 재적합하고 TimesFM은 같은 사전학습 가중치로 새 이력을 입력한다. 테스트 전체를 처음에 한 번 예측하는 방식과 다르다.</p>
<p>현재 기록: 2015-01-10~2026-03-07의 583주, 개발 466주·테스트 117주,
테스트 시작 2023-12-16, 검증 30개·테스트 7개 기점이다. 정책 예측 실제 과거는 583주로 15년 실측 자료가 아니다.</p>
<p>MAE=평균 절대오차, RMSE=평균 제곱오차의 제곱근, bias=평균(예측−실제).
겹치는 32주 예측 경로의 모든 오차 쌍을 집계하므로 독립 표본이 아니다.
회귀 자체 holdout에는 MAPE·학습 행수·학습 종료일도 저장한다.</p>''')
    add("policy", "08 · 정책 경로·세후 환산·시나리오", '''
<p>시행 전 타겟을 고정하여 2026-03-13~2026-12-31의 43주를 예측한다.
마지막 W-SAT 주 종료일은 2027-01-02이며 표시 날짜·끝 구간은 12/31로 자른다. 첫 주 시작도 시행일로 자른다.</p>
<p>주간 달력을 연장하고 국제가격·tax·cap만 ffill한다. 정책 이후 실제 공급가는 입력하지 않는다.
관측 국제가격은 그대로 사용하고 원자료 마지막 주 이후 국제가격·tax·cap은 마지막 값 유지 가정이다.</p>
<p>international_price_assumed·tax_and_cap_assumed는 원자료 최종 주 이후인지로 판단한다.
최종 주 내부의 부분 결측·발표 지연 전체를 표현하는 표시는 아니다. tax·cap 유지 가정은 확정 정책이 아니다.</p>
<p>cf_pretax·cf_aftertax·실제 세전 공급가·cap·gap_to_cap·세전 q10~q90·full_week_end·잘린 표시 구간을 저장한다.
회귀의 경험적 오차 구간과 TimesFM의 분위수는 생성 방식이 다르다.</p>
<p>analyze_policy.py는 미관측 국제가격만 마지막 주간 가격에서 연말까지 −10%·유지·+10%로 선형 변화시킨다.
--change로 비율을 바꿀 수 있고 0&lt;change&lt;1이어야 한다. 관측 국제가격·tax·cap은 바꾸지 않는다.
고정 15년이면 확보된 실제 문맥 하나만 사용한다. 별도 비고정 경로에는 52·104·156주 등 문맥 비교도 남아 있다.</p>
<p>source_sha256로 원자료와 설정의 정합성을 검사한다. 실제 공급가 격차는 관측 구간에서만 계산한다.
CSV 요약·모델 비교 그림·국제가격 시나리오 그림·문맥 그림·가정 JSON·findings.md를 생성한다.</p>''')
    scores_path = JOINT / "model_comparison.csv"
    if not scores_path.exists():
        scores_path = FINAL / "model_comparison.csv"
    scores = []
    if scores_path.exists():
        with scores_path.open(encoding="utf-8-sig", newline="") as stream:
            scores = list(csv.DictReader(stream))
    rows = [[r["split"], r["fuel"], r["model"], r["context"], *[f'{float(r[x]):.4f}' for x in ("mae", "rmse", "bias")], r["origins"]] for r in scores]
    add("results", "09 · 저장된 성능과 공동 예측 비교", '<p>세전 원/L. 현재 저장 결과를 읽었으며 HTML 생성 중 모델을 재실행하지 않는다.</p>' + table(
        ["구간", "유종", "모델", "명목 문맥", "MAE", "RMSE", "bias", "기점"], rows) + '''
<p>timesfm3가 채택한 개별 방식, timesfm3_joint는 미채택 공동 실험이다.
공동 방식은 공급가 3개·국제가격 3개를 한 표본으로 넣어 함께 예측한다. 동일한 검증 조건이지만 다른 유종의 정보도 추가한다.</p>
<p>테스트에서는 공동 방식이 휘발유만 개선하고 경유·등유는 악화했다. 개발 구간에서는 개선 방향이 달랐다.
일관된 공동 방식 우위를 확인하지 못하여 개별 방식으로 확정했다. 공동 코드는 삭제하지 않고 비교 근거로 보존한다.</p>''')
    add("diagnostics", "10 · 진단과 해석의 범위", '''
<p>counterfactual_check.py는 ADF 단위근·수준/차분·시차·구간 적중률·마진 민감도와 그림을 점검한다.
analyze_vif.py는 별도 자료의 공선성을 분석한다. 정확한 검정 옵션·임계값·분기·출력명은 함수 목록과 전체 원문에 있다.</p>
<p>현재 입력은 공통 표의 모든 변수가 아니다. 유류세와 상한은 세후 환산·대조에 사용한다.
가격 격차는 정부 손실보전액이 아니며 다른 수급·마진 변화도 포함할 수 있다.</p>
<p>성능은 미래 국제가격이 주어진 조건부 정확도이다. 사전 예측에는 국제가격 자체의 오차가 추가된다.
분위수는 지정한 국제가격 경로 조건부이고 국제가격 시나리오 불확실성 전체를 포함하지 않는다.
정책 전 가격 형성 관계가 정책 후에도 유지된다는 가정은 예측 성능만으로 검증할 수 없다.</p>
<p>43주 정책 경로는 32주 CV보다 길다. 공개 지연·사전학습 자료 중복은 반영하지 않았다.
테스트는 과거 실험에서 일부 확인했으며 통계적 유의성 검정은 수행하지 않았다.</p>''')
    artifacts, snapshots = [], []
    for directory in (ROOT / "data/processed", FINAL, FINAL / "analysis", JOINT):
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix not in {".csv", ".json", ".md", ".png"}:
                continue
            schema = ""
            if path.suffix == ".csv":
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream)
                    columns = next(reader, [])
                    schema = f"{sum(1 for _ in reader)}행 · " + ", ".join(columns)
            artifacts.append([path.relative_to(ROOT).as_posix(), f"{path.stat().st_size:,} bytes", schema])
            if path.name in {"locked_settings.json", "policy_scenario.json", "assumptions.json", "findings.md", "candidate_availability.csv"}:
                snapshots.append(f'<details><summary>{esc(path.relative_to(ROOT))}</summary><pre>{esc(path.read_bytes().decode("utf-8-sig"))}</pre></details>')
    add("artifacts", "11 · 산출물과 CSV 열 전체", table(["파일", "크기", "행수·열"], artifacts))
    add("settings", "12 · 실행 설정·가정·보고서 원문", ''.join(snapshots))
    figures = []
    for path in sorted((FINAL / "analysis").glob("*.png")):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        figures.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{esc(path.name)}"><figcaption>{esc(path.name)}</figcaption></figure>')
    add("figures", "13 · 저장된 분석 그림", ''.join(figures) or '<p>저장된 그림 없음.</p>')
    records, manifest, functions, sources = [], [], [], []
    for index, path in enumerate(source_files()):
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        name = path.relative_to(ROOT).as_posix()
        digest = hashlib.sha256(raw).hexdigest()
        records.append({"path": name, "source": text, "sha256": digest, "bytes": len(raw)})
        manifest.append([name, role(path), len(text.splitlines()), digest])
        if path.suffix == ".py":
            for node in ast.walk(ast.parse(text.lstrip("\ufeff"))):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    functions.append([name, node.lineno, node.name, ast.get_docstring(node) or "설명은 코드 원문 참조"])
        sources.append(f'<details class="source" id="file-{index}"><summary>{esc(name)}<small>{esc(role(path))} · {len(text.splitlines())}줄</small></summary><p class="hash">SHA256 {digest}</p><pre><code>{esc(text)}</code></pre></details>')
    add("functions", "14 · 모든 함수·클래스와 docstring 전체", table(["파일", "행", "이름", "코드 내 설명"], functions))
    add("source", "15 · 전체 코드 원문 — 생략 없음", '''
<p>루트·딥러닝 폴더의 모든 Python 코드와 관련 README·requirements·상한표·체크포인트 설정을 수록한다.
테스트·이전 실험·공동 비교·HTML 생성기까지 보존한다. 코드 원문은 생성 시점 파일의 바이트와 SHA256로 검증한다.</p>
<div class="tools"><input id="search" type="search" placeholder="파일명 또는 코드 내용 검색" aria-label="코드 검색"><button id="expand">전체 펼치기</button><button id="collapse">접기</button></div>''' +
        table(["파일", "역할", "줄수", "SHA256"], manifest) + ''.join(sources))
    payload = json.dumps(records, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    style = '''
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f2f5f7;color:#1b303e;font:16px/1.85 'Malgun Gothic','Apple SD Gothic Neo',sans-serif}header{background:#12343e;color:white;padding:44px max(24px,calc((100vw - 1360px)/2))}h1{font-size:clamp(26px,3vw,40px);line-height:1.45;margin:10px 0}header p{color:#c5dce2}h2{font-size:24px;margin:0 0 24px}.layout{max-width:1440px;margin:auto;display:grid;grid-template-columns:220px minmax(0,1fr);gap:24px;padding:28px}nav{position:sticky;top:20px;align-self:start;background:#fff;border:1px solid #dce5ea;border-radius:12px;padding:12px;max-height:94vh;overflow:auto}nav a{display:block;color:#526775;padding:7px 10px;text-decoration:none;font-size:13px;border-radius:6px}nav a:hover{background:#e6f3f0;color:#006c6b}main{min-width:0}section{background:#fff;padding:30px;border:1px solid #dce5ea;border-radius:14px;margin-bottom:24px;scroll-margin-top:20px}.notice{background:#e8f5f1;border-left:5px solid #00726b;padding:20px;border-radius:6px}.notice b{font-size:20px}.flow{display:flex;flex-wrap:wrap;gap:12px;margin:24px 0}.flow div{flex:1;min-width:145px;border:1px solid #dce5ea;background:#edf3f6;padding:14px;border-radius:9px;font-weight:bold}small{display:block;font-size:12px;color:#5a6d79;font-weight:normal}code{background:#eef3f6;padding:2px 4px;border-radius:3px;font:13px/1.8 Consolas,monospace;overflow-wrap:anywhere}pre{background:#102731;color:#dfedf0;padding:20px;border-radius:9px;font:12px/1.85 Consolas,'Malgun Gothic',monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:720px;overflow:auto;tab-size:4}pre code{background:none;color:inherit;padding:0}.formula{background:#edf7f3;padding:20px;border-radius:8px;font:17px/1.85 Consolas,'Malgun Gothic',monospace}.scroll{overflow:auto;border:1px solid #dce5ea;border-radius:8px;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:12px 14px;border-bottom:1px solid #dce5ea;text-align:left;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}th{background:#edf3f6;white-space:nowrap}tr:nth-child(even){background:#fafcfd}li{padding-bottom:12px}details{border:1px solid #dce5ea;border-radius:8px;padding:12px;margin:12px 0}summary{font-weight:bold;cursor:pointer;overflow-wrap:anywhere}.hash{font:11px/1.8 Consolas,monospace;color:#596d79;overflow-wrap:anywhere}.tools{display:flex;gap:10px;flex-wrap:wrap}input{flex:1;min-width:190px;border:1px solid #dce5ea;border-radius:8px;padding:12px;font:inherit}button{background:#006f69;color:white;border:0;border-radius:8px;padding:12px;cursor:pointer}img{max-width:100%;height:auto;border:1px solid #dce5ea;border-radius:8px}figure{margin:24px 0}figcaption,footer{color:#596d79;font-size:13px}footer{padding:24px;text-align:center}@media(max-width:850px){.layout{display:block;padding:12px}nav{position:static;display:flex;flex-wrap:wrap;max-height:none;margin-bottom:16px}section{padding:20px}header{padding:26px 20px}}@media print{body{background:white}.layout{display:block;padding:0}nav,.tools{display:none}header{background:white;color:#172735;padding:0}header p{color:#526775}section{border:0;padding:12px 0}pre{max-height:none;background:#f5f5f5;color:#111;overflow:visible}table{font-size:10px}}
'''
    nav = ''.join(f'<a href="#{key}">{esc(title)}</a>' for key, title, _ in sections)
    document = f'<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>회귀·개별 TimesFM 전체 파이프라인</title><style>{style}</style></head><body><header><div>RESEARCH PIPELINE · SOURCE SNAPSHOT</div><h1>석유 가격상한제 연구<br>회귀와 개별 TimesFM 전체 파이프라인</h1><p>자료 → 전처리 → 검증 → 정책 미시행 가격 → 민감도<br>생성 {datetime.now().isoformat(timespec="seconds")} · 원문 {len(records)}개 파일 · 외부 네트워크 의존 없음</p></header><div class="layout"><nav aria-label="목차">{nav}</nav><main>{"".join(x[2] for x in sections)}</main></div><footer>최종 방식: 유종별 개별 예측 · 전체 원문은 생성 시점 스냅샷</footer><script type="application/json" id="source-snapshot">{payload}</script>'
    document += '''<script>const items=[...document.querySelectorAll('details.source')];document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();items.forEach(x=>{x.hidden=!x.textContent.toLowerCase().includes(q);if(q&&!x.hidden)x.open=true;});});document.getElementById('expand').onclick=()=>items.filter(x=>!x.hidden).forEach(x=>x.open=true);document.getElementById('collapse').onclick=()=>items.forEach(x=>x.open=false);</script></body></html>'''
    DEST.write_text(document, encoding="utf-8")
    return records


def verify(records):
    """원문 바이트·파일 포함 범위·내부 링크를 한 번 검사한다."""
    class Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids, self.links, self.payload, self.active = set(), [], [], False
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs:
                self.ids.add(attrs["id"])
            if tag == "a" and attrs.get("href", "").startswith("#"):
                self.links.append(attrs["href"][1:])
            if tag == "script" and attrs.get("id") == "source-snapshot":
                self.active = True
        def handle_endtag(self, tag):
            if tag == "script":
                self.active = False
        def handle_data(self, data):
            if self.active:
                self.payload.append(data)
    parser = Parser()
    parser.feed(DEST.read_text(encoding="utf-8"))
    recovered = json.loads(''.join(parser.payload))
    assert len(recovered) == len(records)
    assert {r["path"] for r in recovered} == {p.relative_to(ROOT).as_posix() for p in source_files()}
    for record in recovered:
        raw = (ROOT / record["path"]).read_bytes()
        assert record["source"].encode("utf-8") == raw, record["path"]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"], record["path"]
    assert all(link in parser.ids for link in parser.links)
    assert len([x for x in parser.ids if x.startswith("file-")]) == len(records)
    print(f"완료: {DEST.name} · 원문 {len(records)}개 파일 바이트/SHA256·내부 링크 검증 통과")


if __name__ == "__main__":
    verify(build())
