"""새 환율·표준화 분석의 성능 지표 전체를 독립 HTML로 정리합니다."""
import csv
import math
from html import escape
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/model_evaluation_normalized_exchange_rate'
CV = ROOT / 'results/timesfm_cv_normalized_exchange_rate'
POLICY = ROOT / 'results/policy_counterfactual_normalized_exchange_rate'
TARGET = ROOT / '경제대회_환율정규화_예측성능지표_전체정리.html'
NAMES = {'recent_margin': '최근 마진 회귀', 'timesfm_cov': 'TimesFM 딥러닝(국제가·환율)',
         'baseline': '일반 OLS 회귀', 'timesfm_uni': 'TimesFM 단변량', 'no_change': '가격 유지 기준'}


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def value(item):
    try:
        number = float(item)
        return '정의되지 않음' if not math.isfinite(number) else f'{number:,.4f}'
    except (TypeError, ValueError):
        return escape(str(item or '—'))


def table(rows, fields, filterable=False):
    header = ''.join(f'<th scope="col">{escape(label)}</th>' for _, label in fields)
    cells = []
    for row in rows:
        attrs = ''
        if filterable:
            attrs = ' class="metric-row"' + ''.join(f' data-{key}="{escape(row[field], quote=True)}"'
                                                     for key, field in [('model', '모형'), ('fuel', '제품'), ('scope', '범위')])
        cells.append('<tr' + attrs + '>' + ''.join('<td>' + (escape(NAMES.get(row[key], row[key])) if key == '모형'
                     else (f'{int(row[key]):,}' if key in ('N', 'n', 'MAPE_N', '폴드', '문맥길이', '기점수', '예측길이', '채점점수', '무변화승수', '기준보다_좋은_기점', 'HAC시차', '부분집합_n') else value(row.get(key)))) + '</td>' for key, _ in fields) + '</tr>')
    return '<div class="scroll"><table><thead><tr>' + header + '</tr></thead><tbody>' + ''.join(cells) + '</tbody></table></div>'


def main():
    metrics = read(BASE / 'metrics.csv')
    summary = read(CV / 'cv_summary.csv')
    folds = read(CV / 'cv_fold_metrics.csv')
    predictions = read(CV / 'cv_predictions.csv')
    changes = read(BASE / 'before_after.csv')
    policy = read(POLICY / 'summary.csv')
    vif = read(BASE / 'vif.csv')
    checks = read(CV / 'cv_protocol_check.csv')
    principal = [r for r in metrics if r['범위'] in ('전체', '제품') and r['모형'] in ('recent_margin', 'timesfm_cov')]
    all_models = [r for r in metrics if r['범위'] in ('전체', '제품')]
    whole = {r['모형']: r for r in principal if r['제품'] == '전체'}
    regression, neural = whole['recent_margin'], whole['timesfm_cov']
    improvement = (1 - float(regression['MAE']) / float(neural['MAE'])) * 100
    rmse_improvement = (1 - float(regression['RMSE']) / float(neural['RMSE'])) * 100
    fuel_comparison = []
    for fuel in ('경유', '등유', '휘발유'):
        pair = {r['모형']: r for r in principal if r['제품'] == fuel}
        reg, deep = pair['recent_margin'], pair['timesfm_cov']
        row = {'유종': fuel}
        for metric in ('MAE', 'RMSE'):
            a, b = float(reg[metric]), float(deep[metric])
            winner = '회귀' if a < b else '딥러닝'
            row.update({f'회귀_{metric}': a, f'딥러닝_{metric}': b,
                        f'{metric}_우세': f'{winner}: 오차 {(1 - min(a, b) / max(a, b)) * 100:.2f}% 감소'})
        row.update({'회귀_MAPE': reg['MAPE_pct'], '딥러닝_MAPE': deep['MAPE_pct']})
        fuel_comparison.append(row)
    definitions = [
        ('MAE', '평균적으로 몇 원 틀리는가', 'Σ|예측−실제| / N', '원/리터', '작을수록 좋음'),
        ('MSE', '큰 오차에 더 큰 벌점을 주는 평균', 'Σ(예측−실제)² / N', '(원/리터)²', '작을수록 좋음'),
        ('RMSE', '큰 오차의 영향을 가격 단위로 표시', '√MSE', '원/리터', '작을수록 좋음'),
        ('MAPE', '실제 가격 대비 평균 오차 비율', '100 × 평균(|예측−실제| / |실제|)', '%', '작을수록 좋음'),
        ('sMAPE', '실제·예측 가격을 함께 기준으로 본 오차', '평균(200 × |예측−실제| / (|실제|+|예측|))', '%', '작을수록 좋음'),
        ('WAPE', '전체 가격 규모 대비 절대오차 합', '100 × Σ|예측−실제| / Σ|실제|', '%', '작을수록 좋음'),
        ('R²', '실제값의 평균만 예측하는 기준 대비 설명력', '1 − Σ오차² / Σ(실제−실제평균)²', '무단위', '클수록 좋음; 음수 가능'),
        ('Bias', '한쪽 방향으로 높게·낮게 예측하는 정도', '평균(예측−실제)', '원/리터', '0에 가까울수록 좋음; 음수는 과소예측'),
        ('MAE skill', '가격 유지 기준 대비 오차 감소', '1 − 해당 모형의 통합 MAE / 가격 유지 통합 MAE', '무단위', '클수록 좋음; 음수는 기준보다 나쁨'),
        ('최대오차', '가장 크게 벗어난 한 예측의 오차', 'max(|예측−실제|)', '원/리터', '작을수록 좋음'),
        ('오차 q05·q95', '부호 있는 오차 분포의 5%·95% 위치', '예측−실제의 5%·95% 분위수', '원/리터', '분포 요약; 정책 추정의 신뢰구간은 아님'),
        ('MAE 1~4주 / 마지막 4주', '가까운 예측과 먼 예측의 오차', '리드 1~4 / 29~32의 평균 절대오차', '원/리터', '작을수록 좋음'),
        ('기점별 MAE 표준편차', '시점별 성능의 흔들림', '22개 기점의 MAE 표준편차', '원/리터', '안정성 참고; 작다고 정확하다는 뜻은 아님'),
        ('DM t·p', '회귀 대비 기점별 MAE 차이 검정', 'HAC 시차 2 + 소표본 보정, 기존 코드의 양측 t 검정', '무단위', 't 양수면 회귀보다 오차 큼; p<0.05는 개별 검정 기준'),
        ('부분집합 t·p', '겹치지 않는 8개 기점만의 차이 검정', '3개 기점마다 하나를 골라 쌍체 t 검정', '무단위', '검정력이 낮을 수 있음; p가 크다고 동일한 성능은 아님'),
        ('VIF', '설명변수 간 선형 중복', '1 / (1−보조회귀 R²)', '무단위', '성능 지표가 아닌 공선성 진단'),
        ('N / MAPE_N', '채점한 예측 수 / MAPE에 사용한 수', '행 수 / 실제값이 0이 아닌 행 수', '개', '독립적인 날짜 수와 다를 수 있음'),
    ]
    body = ['<header><p class="eyebrow">환율 포함 · Z-score 입력 · 새 분석 결과</p><h1>예측오차와 예측력 지표 전체 정리</h1><p>최근 마진 회귀와 TimesFM 딥러닝을 중심으로 평가 조건, 모든 성능 지표, 통계 검정과 정책 시나리오의 차이를 정리했습니다.</p></header>',
            '<nav><a href="#overview">핵심 결과</a><a href="#implications">쉬운 시사점</a><a href="#definitions">지표 설명</a><a href="#main-results">성능표</a><a href="#statistics">통계 검정</a><a href="#policy">정책 그래프 해석</a><a href="#all-metrics">전체 세부 지표</a><a href="#sources">근거 파일</a></nav>',
            '<section id="overview"><h2>1. 먼저 구분해야 할 두 결과</h2><p class="note"><strong>정책 미시행 가격의 실제 정답은 관측할 수 없습니다.</strong> 따라서 정책 그래프에서 추정선과 실제 가격의 차이를 예측오차로 채점하지 않습니다. 아래 MAE·RMSE·R²는 <strong>정책 이전 교차검증</strong>에서 확인한 예측력이며, 정책 이후 29주 그래프의 차이는 별도의 조건부 시나리오 격차입니다.</p>',
            f'<div class="cards"><article><span>최근 마진 회귀 MAE</span><strong>{float(regression["MAE"]):.4f}</strong><small>원/리터</small></article><article><span>TimesFM 딥러닝 MAE</span><strong>{float(neural["MAE"]):.4f}</strong><small>원/리터</small></article><article><span>딥러닝 대비 회귀의 MAE 감소</span><strong>{improvement:.2f}%</strong><small>RMSE는 {rmse_improvement:.2f}% 감소</small></article></div>',
            '<p>전체 평균에서는 회귀의 오차가 조금 작았습니다. 휘발유·경유는 회귀의 MAE가 낮고, 등유는 딥러닝의 MAE가 낮습니다. 등유의 RMSE는 회귀가 낮아 지표별 순위가 다릅니다. 새 DM 검정에서 두 모델의 차이는 세 유종 모두 5% 수준에서 유의하게 확인되지 않았습니다.</p></section>',
            '<section id="implications"><h2>쉽게 읽는 시사점</h2><p class="note">아래 시사점은 이번 결과를 바탕으로 한 <strong>해석과 추론</strong>입니다. 확인된 숫자와 그 숫자가 의미할 수 있는 내용을 구분했습니다.</p>',
            f'<h3>1. 회귀분석도 충분히 경쟁력이 있습니다</h3><p><strong>결과:</strong> 회귀의 전체 평균 오차는 약 20.94원, 딥러닝은 약 21.19원입니다. 회귀의 오차가 {improvement:.2f}% 작지만, 통계 검정에서는 확실한 우세가 확인되지 않았습니다.</p><p><strong>시사점:</strong> 이번 자료에서는 복잡한 딥러닝을 쓰는 것만으로 예측이 크게 좋아지지는 않았습니다. 계수로 가격 변화를 설명할 수 있는 회귀를 기본 모델로 두고, 딥러닝으로 결과를 함께 확인하는 방식이 합리적입니다. 두 모델의 성능이 완전히 같다고 입증된 것은 아닙니다.</p>',
            '<h3>2. 환율은 중요할 수 있지만, 넣는다고 예측이 자동으로 좋아지지는 않습니다</h3><p><strong>결과:</strong> 환율을 포함한 새 분석의 평균 오차는 기존보다 회귀 약 7.91%, 딥러닝 약 2.37% 커졌습니다. 환율의 VIF는 약 1.64~1.97로 낮습니다.</p><p><strong>시사점:</strong> 환율이 수입가격에 영향을 줄 수 있다는 경제적 설명과, 환율을 추가했을 때 예측오차가 줄어드는지는 서로 다른 문제입니다. 이번 결과만으로 환율이 필요 없다고 단정하기 어렵습니다. 특히 딥러닝은 표준화도 함께 바뀌었으므로 환율만 넣고 뺀 동일 조건의 비교가 더 필요합니다. VIF가 높아서 환율을 제외했다는 설명도 이 결과와 맞지 않습니다.</p>',
            '<h3>3. 유종과 목적에 따라 모델을 선택할 수 있습니다</h3><p><strong>결과:</strong> 휘발유·경유는 회귀의 MAE가 작습니다. 등유는 딥러닝의 MAE가 작지만 회귀의 RMSE가 작습니다.</p><p><strong>시사점:</strong> 등유의 평균적인 오차를 줄이는 데에는 딥러닝이 유리해 보이고, 크게 빗나가는 예측에 더 큰 불이익을 주는 기준에서는 회귀가 유리해 보입니다. 다만 유종별 차이도 통계적으로 확실하지 않아, 현재 순위를 고정된 승자로 받아들이기보다 이후 기간에서도 확인하는 것이 좋습니다.</p>',
            '<h3>4. 높은 R²와 실제 미래의 예측력은 구분해야 합니다</h3><p><strong>결과:</strong> 두 모델의 R²는 약 0.987입니다. 가격이 마지막 관측값에 머문다고 예측하는 기준보다 전체 MAE가 약 81% 작았습니다. 다만 평가할 때 미래 국제가격과 환율의 실제값을 제공했습니다.</p><p><strong>시사점:</strong> 이번 조건에서는 두 모델이 단순한 가격 유지 예측보다 과거 가격 움직임을 잘 따라갔습니다. 그러나 R²가 높다고 가격을 약 99% 정확하게 맞힌다는 뜻은 아닙니다. 실제 운영에서는 국제가격과 환율도 미리 예측해야 하므로 그 오차까지 고려하면 결과가 달라질 수 있습니다.</p>',
            '<h3>5. 정책 그래프는 가격 압력을 살펴보는 자료입니다</h3><p><strong>결과:</strong> 두 모델 모두 세 유종에서 정책 미시행 가정의 평균 가격을 실제 가격보다 높게 추정했습니다. 평균 차이는 휘발유 약 157~177원, 경유 약 343~371원, 등유 약 308~340원/리터입니다.</p><p><strong>시사점:</strong> 과거 국제가격·환율과 국내 가격의 관계가 유지됐다고 가정하면, 실제보다 높은 가격이 예상됐다는 뜻입니다. 두 모델에서 방향이 같다는 점은 정책 이후 가격 상승이 억제됐을 가능성을 검토할 근거가 됩니다. 하지만 다른 가격 결정 요인도 달라졌을 수 있고 정책이 없었을 실제 가격은 관측되지 않으므로, 이 차이를 정책이 낮춘 금액이나 소비자가 절약한 금액으로 확정할 수는 없습니다.</p>',
            '<h3>대회 본문에 넣기 쉬운 문장</h3><p class="note">이번 분석에서는 최근 마진 회귀와 딥러닝의 예측력이 비슷하게 나타났고, 회귀의 평균 오차가 약 1.18% 작았다. 환율을 포함한 새 분석은 기존보다 평균 오차가 커져, 변수 추가가 항상 예측 개선으로 이어지지는 않음을 보여준다. 두 모델 모두 정책 미시행 가정의 가격을 실제보다 높게 추정했지만, 그 차이는 과거 가격 관계가 유지된다는 가정 아래의 추정치이므로 정책의 인과적 효과로 단정하기 어렵다.</p></section>',
            '<section><h2>2. 동일한 평가 조건</h2><ul><li>타겟: 휘발유·경유·등유의 주별 세전 공급가격(원/리터).</li><li>입력: 해당 유종 국제가격의 같은 주·1주 전·2주 전과 같은 주 원달러환율. 새 csv/normalized 파일을 직접 읽었습니다.</li><li>정책 이전 자료의 앞 80%인 462주 안에서 22개 기점, 간격 13주, 예측 32주로 검증했습니다. 유종별 704개, 전체 2,112개 예측입니다.</li><li>각 기점까지의 자료만으로 Z-score를 다시 계산했습니다. 회귀의 국제가격 계수 합 제약은 원래 단위에서 적용했습니다.</li><li>회귀는 기점마다 재적합하고 TimesFM은 기존 설치 모델의 가중치를 고정해 문맥만 확장했습니다. 외부 사전학습 이력은 동일하지 않습니다.</li><li>미래 국제가격·환율의 실제값을 제공한 조건부 예측입니다. 실제 미래 공변량을 모르는 예측 성능은 별도 확인이 필요합니다.</li><li>32주 창이 13주마다 겹쳐 같은 날짜가 다른 기점·리드로 반복 채점됩니다. N=2,112를 독립적인 2,112주로 해석하지 않습니다.</li></ul></section>',
            '<section id="definitions"><h2>3. 지표의 의미와 계산 방법</h2>']
    definition_rows = [dict(zip(('지표', '의미', '계산', '단위', '해석'), row)) for row in definitions]
    body += [table(definition_rows, [(k, k) for k in ('지표', '의미', '계산', '단위', '해석')]),
             '<p>MAPE는 실제값 0을 제외합니다. sMAPE는 실제·예측 모두 0이면 0으로 처리합니다. 실제값 분산이 0이면 R²는 정의되지 않습니다. MAPE가 3%라고 해서 “정확도 97%”라고 표현하지 않습니다. R²=0.9872 역시 정확도 98.72%라는 뜻이 아닙니다.</p>',
             '<p><strong>skill의 정의 차이:</strong> 이번 metrics.csv는 통합 MAE의 비율을 사용합니다. 기존 CV 요약의 skill_vs_무변화는 기점별 MAE 비율의 기하평균을 이용한 1−exp(평균 log 비율)이므로 숫자가 다릅니다. 두 정의를 섞지 않습니다.</p></section>',
             '<section id="main-results"><h2>4. 주된 두 모델의 전체·유종별 성능</h2><p>백분율 오차는 %, MAE·RMSE·Bias는 원/리터, MSE는 (원/리터)²입니다. 모든 값은 저장된 새 예측으로 계산했습니다.</p>']
    score_fields = [('제품', '유종'), ('모형', '모델'), ('N', 'N'), ('MAE', 'MAE ↓'), ('MSE', 'MSE ↓'),
                    ('RMSE', 'RMSE ↓'), ('MAPE_pct', 'MAPE % ↓'), ('sMAPE_pct', 'sMAPE % ↓'),
                    ('WAPE_pct', 'WAPE % ↓'), ('R2', 'R² ↑'), ('Bias', 'Bias → 0'), ('MAE_skill', 'MAE skill ↑')]
    body += [table(principal, score_fields),
             '<h3>경유·등유·휘발유별 예측력 차이</h3><p>각 유종은 같은 704개 예측을 비교했습니다. MAE는 평균적으로 몇 원 틀렸는지, RMSE는 큰 오차를 더 무겁게 반영한 값입니다. 우세 비율은 <strong>(오차가 큰 모델−오차가 작은 모델)÷오차가 큰 모델×100</strong>으로 계산한 오차 감소율이며, 정확도가 그만큼 높다는 뜻은 아닙니다.</p>',
             table(fuel_comparison, [('유종', '유종'), ('회귀_MAE', '회귀 MAE 원/리터'), ('딥러닝_MAE', '딥러닝 MAE 원/리터'), ('MAE_우세', '평균 오차 기준'), ('회귀_RMSE', '회귀 RMSE 원/리터'), ('딥러닝_RMSE', '딥러닝 RMSE 원/리터'), ('RMSE_우세', '큰 오차를 반영한 기준'), ('회귀_MAPE', '회귀 MAPE %'), ('딥러닝_MAPE', '딥러닝 MAPE %')]),
             '<ul><li><strong>경유:</strong> 회귀의 평균 오차가 딥러닝보다 작습니다. MAE와 RMSE 모두 회귀가 낮아 이번 평가에서는 두 기준의 선택이 일치합니다.</li><li><strong>등유:</strong> 평균 오차(MAE)는 딥러닝이 작지만, 큰 오차를 더 무겁게 반영하는 RMSE는 회귀가 작습니다. 평소 평균적인 오차를 줄일지, 크게 빗나가는 예측을 더 중요하게 볼지에 따라 판단이 달라집니다.</li><li><strong>휘발유:</strong> 회귀의 MAE와 RMSE가 모두 작습니다. 다만 평균 오차 차이는 작아서 딥러닝보다 확실히 우수하다고 단정하기 어렵습니다.</li></ul>',
             '<p><strong>유종 간 차이:</strong> 두 모델 모두 등유의 MAE가 가장 큽니다. 즉 이번 평가에서 등유는 평균적으로 더 많이 빗나간 유종입니다. 가격 수준이 다른 유종을 비교할 때는 원 단위 MAE와 함께 MAPE도 확인해야 합니다. 이 결과만으로 등유의 오차가 큰 원인을 공급구조나 정책으로 확정할 수는 없습니다.</p>',
             '<p class="note"><strong>해석:</strong> 이번 수치에서는 휘발유·경유는 회귀를 기본 모델로, 등유는 MAE와 RMSE의 목적에 맞춰 두 모델을 함께 검토할 수 있습니다. 세 유종 모두 DM 검정에서 확실한 성능 차이가 확인되지 않았으므로, 위 우세 비율은 이번 표본의 관측 결과이며 앞으로도 유지되는 우위를 보장하지 않습니다.</p>',
             '<h3>최대오차</h3>']
    maxima = []
    for row in principal:
        sample = [p for p in predictions if p['모형'] == row['모형'] and (row['제품'] == '전체' or p['제품'] == row['제품'])]
        errors = [float(p['예측_세전가격']) - float(p['실제_세전가격']) for p in sample]
        maxima.append({'제품': row['제품'], '모형': row['모형'], '최대오차': max(map(abs, errors))})
        if len(sample) != int(row['N']) or abs(sum(map(abs, errors)) / len(errors) - float(row['MAE'])) > 1e-6:
            raise ValueError('성능표와 예측 원자료 불일치')
    body += [table(maxima, [('제품', '유종'), ('모형', '모델'), ('최대오차', '최대 절대오차 원/리터')]),
             '<details><summary>비교 기준 모형까지 포함한 전체 성능표(5개 모델)</summary><p>일반 OLS는 환율을 사용하지만 최근 마진 보정을 하지 않습니다. 단변량 TimesFM과 가격 유지 기준은 국제가격·환율을 사용하지 않으므로 주된 두 모델과 입력 정보가 다릅니다.</p>', table(all_models, score_fields), '</details></section>',
             '<section><h2>5. 기존 결과와 비교</h2><p>감소율=(기존 MAE−새 MAE)/기존 MAE×100. 음수는 오차 증가입니다. 환율 추가와 표준화를 함께 적용했으며, TimesFM의 환율 효과만 분리한 실험은 아닙니다.</p>',
             table(changes, [('제품', '유종'), ('모형', '모델'), ('기존_MAE', '기존 MAE'), ('환율포함_MAE', '새 MAE'),
                             ('MAE_감소율_pct', 'MAE 감소율 %'), ('기존_RMSE', '기존 RMSE'), ('환율포함_RMSE', '새 RMSE')]),
             '<p>전체 MAE는 기존보다 회귀 약 7.91%, 딥러닝 약 2.37% 증가했습니다. 환율 추가·표준화가 이번 사양에서 평균 오차를 개선하지 않았다는 뜻이며, 환율의 경제적 효과가 없다는 뜻은 아닙니다.</p></section>',
             '<section id="statistics"><h2>6. 예측 길이·기점별 변동과 통계 검정</h2><p>아래 표는 새 CV 요약 전체입니다. 1~4주와 마지막 4주 오차, 기점별 변동, 분포·검정 정보를 포함합니다. 기준은 최근 마진 회귀이며 평균차이는 해당 모형 MAE−회귀 MAE입니다.</p>']
    body += [table(summary, [(key, key) for key in summary[0]]),
             '<p>TimesFM 공변량의 DM p값은 휘발유 0.8022, 경유 0.3293, 등유 0.8571입니다. 모두 0.05보다 크며 통계적으로 확실한 우세를 주장하기 어렵습니다. 비중첩 부분집합은 8개 기점이라 검정력이 낮을 수 있습니다. 표의 p값을 다중검정 보정값으로 제시하지 않습니다.</p>',
             '<p>“기준보다_좋은_기점”은 해당 모형의 기점별 MAE가 최근 마진 회귀보다 낮았던 기점 수입니다. “무변화승수”는 가격 유지 기준보다 낮았던 수입니다. 기점별 MAE평균은 같은 길이의 기점들을 평균한 값이며, MAE표준편차는 기점 간 변동입니다. 오차 q05~q95는 분포 요약이고 새 정책 시나리오의 확률구간으로 검증한 것이 아닙니다.</p></section>',
             '<section><h2>7. VIF: 환율 제외의 근거인가?</h2>', table(vif, [(key, key) for key in vif[0]]),
             '<p>환율 VIF는 약 1.64~1.97로 낮습니다. 높은 VIF는 주로 인접 주 국제가격끼리 나타납니다. 따라서 환율을 VIF가 높아서 제외했다고 설명할 수 없습니다. 예측력이 떨어졌다는 결과와 공선성 진단은 별개의 근거입니다.</p></section>',
             '<section id="policy"><h2>8. 정책 미시행 그래프의 숫자는 예측오차가 아닙니다</h2><p>새 그래프는 정책 이전 전체 578주로 최종 추정하고 2026-03-14~2026-09-26의 29주를 계산했습니다. 교차검증의 462주 안 평가와 학습 범위가 다릅니다. 정책 이후 실제 국내 가격은 비교선으로만 쓰고 학습·문맥에 넣지 않았습니다.</p>',
             table(policy, [(key, key.replace('_', ' ')) for key in policy[0]]),
             '<p class="note">위 차이=정책 미시행 가정의 모형 추정−실제 세전 공급가격입니다. 미시행 가격의 실제 정답이 없으므로 이 차이를 MAE·RMSE·정책효과·정부 손실보전액으로 해석하지 않습니다. 정책이 없어도 과거 가격 관계가 유지된다는 가정과 실제 국제가격·환율을 이용한 조건부 시나리오입니다.</p><a href="경제대회_정책미시행_그래프6개.html">정책 그래프 6개 보기</a></section>',
             '<section id="all-metrics"><h2>9. 전체 세부 지표: 제품·폴드·예측 리드</h2><p>metrics.csv의 '+str(len(metrics))+'행을 모두 포함했습니다. 필터를 선택해 확인할 수 있습니다. 제품_리드의 구간은 몇 주 앞인지, 제품_폴드의 구간은 평가 기점 번호입니다.</p>',
             '<div class="filters"><label>모델 <select id="model"><option value="">전체</option>'+''.join('<option value="'+k+'">'+escape(v)+'</option>' for k,v in NAMES.items())+'</select></label><label>유종 <select id="fuel"><option value="">전체</option><option>경유</option><option>휘발유</option><option>등유</option></select></label><label>범위 <select id="scope"><option value="">전체</option><option>전체</option><option>제품</option><option>제품_폴드</option><option>제품_리드</option></select></label><span id="count"></span></div>',
             table(metrics, [('범위', '범위'), ('구간', '기점/리드'), *score_fields, ('MAPE_N', 'MAPE 표본 수')], True),
             '<details><summary>CV 원본 기점별 지표 전체: '+str(len(folds))+'행</summary>', table(folds, [(key, key) for key in folds[0]]), '</details></section>',
             '<section id="sources"><h2>10. 검증 및 근거 파일</h2><p>회귀 원자료 경로와 표준화 입력 경로의 프로토콜 점검은 MAE 차이 표시값 0.0000원으로 통과했습니다. 같은 모형의 계산 경로를 비교한 것이며, 회귀와 TimesFM의 성능이 같다는 뜻은 아닙니다.</p>', table(checks, [(key, key) for key in checks[0]]),
             '<details><summary>단위테스트 상태</summary><p>관련 단위테스트 1회에서 7개 중 6개가 통과했습니다. 나머지는 약 1e-14의 표준편차 차이를 완전 일치로 비교해 실패했고, 수치 허용오차 비교로 수정했으나 재실행하지 않았습니다. 실제 분석의 데이터·표본·프로토콜 검증은 별도로 통과했습니다.</p></details><ul>']
    paths = [BASE / 'metrics.csv', CV / 'cv_predictions.csv', CV / 'cv_summary.csv', CV / 'cv_fold_metrics.csv',
             BASE / 'before_after.csv', BASE / 'vif.csv', BASE / 'run_manifest.json', BASE / 'validation.md', POLICY / 'predictions.csv']
    for path in paths:
        body.append('<li><a href="'+escape(path.relative_to(ROOT).as_posix())+'">'+escape(path.relative_to(ROOT).as_posix())+'</a></li>')
    body += ['</ul><p>자료나 계수를 새로 추정한 문서가 아니라, 완료된 새 분석의 원자료와 성능 결과를 정리한 문서입니다. 그림·정책 격차와 교차검증 성능은 구분해서 읽어야 합니다.</p></section>']
    css = '''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f2f5fa;color:#24334a;font:16px/1.8 "Malgun Gothic",sans-serif;word-break:keep-all;overflow-wrap:anywhere}main{max-width:1280px;margin:auto;padding:24px}header{background:#17324f;color:white;padding:36px;border-radius:14px}h1{font-size:32px;line-height:1.5}.eyebrow{color:#b7dcff}h2{color:#19517d;font-size:23px}section{background:white;border:1px solid #dbe4ef;padding:28px;border-radius:12px;margin:24px 0;scroll-margin-top:20px}nav,.filters{display:flex;flex-wrap:wrap;gap:15px;margin:22px 0}a{color:#176aa9;text-underline-offset:3px}.note{background:#edf4fd;border-left:4px solid #368cc6;padding:18px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.cards article{padding:22px;background:#edf4fd;border-radius:12px}.cards span,.cards small{display:block}.cards strong{font-size:32px;color:#19517d}.scroll{overflow:auto;max-height:620px;border:1px solid #dce4ef;margin:20px 0}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:11px;border-bottom:1px solid #dce4ef;text-align:left;min-width:105px}th{background:#eaf1fa;position:sticky;top:0;z-index:1}tbody tr:nth-child(even){background:#fafcfe}td{font-variant-numeric:tabular-nums}summary{cursor:pointer;font-weight:bold;padding:14px 0}select{padding:8px;font:inherit;border:1px solid #bccbdd;border-radius:6px}li{margin:9px 0}tr[hidden]{display:none}@media(max-width:750px){main{padding:12px}section,header{padding:20px}.cards{grid-template-columns:1fr}h1{font-size:26px}}@media print{body{background:white;font-size:10pt}main{padding:0;max-width:none}header{background:white;color:black;padding:0}.eyebrow{color:black}nav,.filters{display:none}section{border:0;padding:0;break-inside:auto}.scroll{max-height:none;overflow:visible}table{font-size:8pt}td,th{padding:4px;min-width:0}th{position:static}tr{break-inside:avoid}h2{break-after:avoid}}'''
    js = '''const rows=[...document.querySelectorAll('.metric-row')];const selectors=['model','fuel','scope'].map(id=>document.getElementById(id));function filter(){let n=0;for(const row of rows){const show=selectors.every(s=>!s.value||row.dataset[s.id]===s.value);row.hidden=!show;if(show)n++;}document.getElementById('count').textContent=`${n} / ${rows.length}행`;}selectors.forEach(s=>s.addEventListener('change',filter));filter();'''
    html = '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>환율·정규화 예측성능 지표 전체 정리</title><style>'+css+'</style></head><body><main>'+''.join(body)+'</main><script>'+js+'</script></body></html>'
    TARGET.write_text(html, encoding='utf-8')
    # 관련 검증 1회: 모든 지표 행, HTML 태그, 원자료 일치, 출처 파일.
    class Check(HTMLParser):
        def __init__(self):
            super().__init__(); self.stack=[]; self.ids=set(); self.links=[]; self.metric_rows=0
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag not in ('meta','br','hr','img','input','link'): self.stack.append(tag)
            if 'id' in attrs: self.ids.add(attrs['id'])
            if tag == 'a': self.links.append(attrs['href'])
            if tag == 'tr' and attrs.get('class') == 'metric-row': self.metric_rows += 1
        def handle_endtag(self, tag):
            assert self.stack.pop() == tag, tag
    check = Check(); check.feed(html)
    assert not check.stack and check.metric_rows == len(metrics) == 830
    assert len(folds) == 330 and len(principal) == 8
    for link in check.links:
        assert link[1:] in check.ids if link.startswith('#') else (ROOT / link).is_file(), link
    print(f'검증 통과: 전체 지표 {len(metrics)}행, 기점별 지표 {len(folds)}행, 원자료·HTML·출처 링크 확인. {TARGET}')


if __name__ == '__main__':
    main()
