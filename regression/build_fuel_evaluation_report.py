"""완료된 환율·표준화 결과에서 유종별 상세 평가와 시사점을 작성한다."""
import math
from html.parser import HTMLParser
from build_metrics_html import ROOT, BASE, CV, NAMES, read, table

TARGET = ROOT / '경제대회_유종별_예측력_상세분석.html'
MODELS = ('recent_margin', 'timesfm_cov')
DEFINITIONS = [
    ('MAE', '평균적으로 몇 원 틀리는지', '원/리터', '작을수록 좋음'),
    ('MSE', '오차를 제곱해 평균한 값. 큰 실수에 큰 벌점', '(원/리터)²', '작을수록 좋음'),
    ('RMSE', 'MSE의 제곱근. 큰 실수의 영향을 원 단위로 비교', '원/리터', '작을수록 좋음'),
    ('MAPE_pct', '실제 가격 대비 평균 절대오차 비율', '%', '작을수록 좋음'),
    ('sMAPE_pct', '실제·예측 가격의 평균 크기 대비 절대오차 비율', '%', '작을수록 좋음'),
    ('WAPE_pct', '절대오차 합 ÷ 실제 가격 절댓값 합', '%', '작을수록 좋음'),
    ('R2', '실제값의 변동을 얼마나 잘 설명하는지', '무단위', '클수록 좋음. 정확도 %가 아님'),
    ('Bias', '평균(예측−실제). 방향이 있는 오차', '원/리터', '0에 가까울수록 좋음. 음수는 낮게 예측'),
    ('MAE_skill', '1−모델 MAE÷마지막 가격 유지 기준 MAE', '무단위', '클수록 좋음. 0.8은 기준 대비 오차 80% 감소'),
]


def main():
    metrics = read(BASE / 'metrics.csv')
    summary = read(CV / 'cv_summary.csv')
    predictions = read(CV / 'cv_predictions.csv')
    body = ['<header><h1>경유·등유·휘발유 예측력 상세 분석</h1><p>최근 마진 회귀와 TimesFM 딥러닝 · 환율 포함 · 새 정규화 자료</p></header>',
            '<nav><a href="#diesel">경유</a> · <a href="#kerosene">등유</a> · <a href="#gasoline">휘발유</a> · <a href="#implications">시사점</a></nav>',
            '<section><h2>평가 조건과 읽는 방법</h2><p>정책 이전 앞 80% 자료(462주) 안에서 22개 기점, 13주 간격, 32주 예측으로 평가했습니다. 각 유종·모델은 같은 704개 예측을 비교합니다. 예측 창이 겹쳐 704개의 독립적인 주를 뜻하지 않습니다. 두 모델에 같은 국제가격(현재·1주 전·2주 전)과 환율을 제공하고 각 기점 이전 자료만으로 표준화했습니다.</p><p>회귀는 기점마다 적합하고 최근 52주 마진을 반영합니다. 딥러닝은 기존 사전학습 가중치를 고정하고 과거 가격 문맥을 사용합니다. 외부 사전학습 이력까지 동일한 실험은 아닙니다. 미래 국제가격·환율의 실제값을 제공한 조건부 예측이므로 실제 미래 운영의 성능과 구분해야 합니다.</p><p><strong>오차 감소율</strong>=(큰 오차−작은 오차)÷큰 오차×100. R²·Bias·skill은 같은 방식의 우세 백분율로 표시하지 않습니다. MAPE가 3%라고 정확도 97%인 것은 아닙니다.</p></section>',
            '<section><h2>지표별 의미</h2>', table([dict(zip(('지표','뜻','단위','판단'), d)) for d in DEFINITIONS], [(k,k) for k in ('지표','뜻','단위','판단')]), '</section>']
    verified = 0
    for fuel, anchor in [('경유','diesel'), ('등유','kerosene'), ('휘발유','gasoline')]:
        pair = {r['모형']: r for r in metrics if r['범위'] == '제품' and r['제품'] == fuel and r['모형'] in MODELS}
        detail = {r['모형']: r for r in summary if r['제품'] == fuel and r['모형'] in MODELS}
        reg, deep = pair['recent_margin'], pair['timesfm_cov']
        rows = []
        for key, meaning, unit, rule in DEFINITIONS:
            a, b = float(reg[key]), float(deep[key])
            if key in ('R2', 'MAE_skill'):
                interpretation = f'{"회귀" if a > b else "딥러닝"}가 높음; 차이 {abs(a-b):.6f}'
            elif key == 'Bias':
                interpretation = f'{"회귀" if abs(a) < abs(b) else "딥러닝"}가 0에 가까움'
            else:
                interpretation = f'{"회귀" if a < b else "딥러닝"} 오차 {(1-min(a,b)/max(a,b))*100:.2f}% 감소'
            rows.append({'지표':key,'회귀':a,'딥러닝':b,'단위':unit,'해석':interpretation})
        body += [f'<section id="{anchor}"><h2>{fuel}: 주요 예측력 지표 전체</h2>', table(rows, [(k,k) for k in ('지표','회귀','딥러닝','단위','해석')])]
        diagnostics = []
        for model in MODELS:
            sample = [p for p in predictions if p['제품'] == fuel and p['모형'] == model]
            errors = [float(p['예측_세전가격'])-float(p['실제_세전가격']) for p in sample]
            assert len(errors) == 704
            assert abs(sum(abs(e) for e in errors)/704-float(pair[model]['MAE'])) < 1e-6
            assert abs(sum(e*e for e in errors)/704-float(pair[model]['MSE'])) < 1e-6
            verified += 1
            d = detail[model]
            diagnostics.append({'모형':model,'최대절대오차':max(map(abs,errors)), **{k:d[k] for k in ('MAE_1_4주','MAE_마지막4주','MAE표준편차','오차_q05','오차_q95','편향','기준보다_좋은_기점')}})
        body += ['<h3>예측 길이·안정성·큰 실수</h3>', table(diagnostics, [('모형','모델'),('최대절대오차','최대 오차 원/리터'),('MAE_1_4주','1~4주 MAE'),('MAE_마지막4주','29~32주 MAE'),('MAE표준편차','기점별 MAE 표준편차'),('오차_q05','오차 5% 위치'),('오차_q95','오차 95% 위치'),('편향','평균 편향'),('기준보다_좋은_기점','회귀보다 MAE가 낮은 기점 수')]),
                 '<p>MAE 표준편차는 22개 평가 기점에 따른 성능 변동입니다. 작을수록 기점 간 차이가 작지만 정확도가 높다는 뜻은 아닙니다. 오차 5%·95% 위치는 관측 오차 분포이며 미래 예측구간이나 정책효과의 신뢰구간이 아닙니다. 회귀 행의 “회귀보다 낮은 기점 수” 0은 자기 자신과 비교한 값입니다.</p>']
        near = {m:float(detail[m]['MAE_1_4주']) for m in MODELS}
        far = {m:float(detail[m]['MAE_마지막4주']) for m in MODELS}
        body += [f'<p><strong>기간별 해석:</strong> 1~4주에서는 {NAMES[min(near,key=near.get)]}의 평균 오차가 작고, 29~32주에서는 {NAMES[min(far,key=far.get)]}의 평균 오차가 작습니다. 전체 평균의 순위와 단기·장기 순위가 다를 수 있으므로 실제 필요한 예측 기간에 맞춰 판단해야 합니다.</p>']
        a, b = float(reg['MAE']), float(deep['MAE'])
        winner = '회귀' if a < b else '딥러닝'
        text = 'MAE·RMSE 모두 회귀가 작아 평균 오차와 큰 오차를 반영한 기준의 선택이 일치합니다.' if fuel != '등유' else 'MAE는 딥러닝이 작고 RMSE는 회귀가 작습니다. 평균적인 오차를 줄이는 목적과 큰 실수를 더 무겁게 보는 목적에 따라 선택이 달라집니다.'
        body += [f'<h3>쉽게 이해하는 결과와 시사점</h3><p>{fuel}는 회귀가 평균 약 {a:.2f}원/리터, 딥러닝이 약 {b:.2f}원/리터 빗나갔습니다. {winner}의 평균 오차가 상대 모델보다 {(1-min(a,b)/max(a,b))*100:.2f}% 작았습니다. {text}</p>',
                 f'<p>딥러닝의 평균 편향은 {float(deep["Bias"]):.2f}원/리터, 회귀는 {float(reg["Bias"]):.2f}원/리터입니다. 부호는 가격을 평균적으로 높게(+), 낮게(−) 예측한 방향입니다. 정책 미시행 가격을 해석할 때에도 이 방향을 참고할 수 있지만, 과거 편향을 정책 기간에 그대로 적용해 보정할 근거는 아직 없습니다.</p>',
                 f'<p><strong>통계 검정:</strong> 딥러닝과 회귀의 기점별 MAE 차이에 대한 DM p값은 {float(detail["timesfm_cov"]["DM_p"]):.4f}, 비중첩 8개 기점의 검정 p값은 {float(detail["timesfm_cov"]["부분집합_p"]):.4f}입니다. 5% 기준에서 확실한 차이가 확인되지 않았습니다. 따라서 위 비율은 이번 표본의 결과이며, 미래의 우위나 두 모델의 완전한 동등성을 입증하지 않습니다.</p></section>']
    body += ['<section id="implications"><h2>세 유종을 함께 봤을 때의 시사점</h2><ul><li><strong>등유를 더 주의해서 해석해야 합니다.</strong> 두 모델 모두 등유의 평균 원 단위 오차가 가장 큽니다. MAPE·sMAPE·WAPE도 함께 비교하면 가격 수준 차이를 고려할 수 있습니다. 오차가 큰 원인을 이 결과만으로 특정 정책이나 공급구조 탓으로 확정할 수는 없습니다.</li><li><strong>모델 선택은 목적과 예측 기간에 맞춰야 합니다.</strong> 이번 표본의 전체 MAE 기준으로는 경유·휘발유에 회귀, 등유에 딥러닝이 유리합니다. 등유도 큰 오차를 중시하는 RMSE 기준에서는 회귀가 유리합니다. 단기·장기 지표와 시점별 변동까지 확인한 뒤 선택하는 것이 합리적입니다.</li><li><strong>설명 가능한 회귀를 기본 모델로 사용할 근거가 있습니다.</strong> 딥러닝과의 차이가 작고 통계적으로 확실하지 않으므로, 회귀 계수로 국제가격·환율과 가격의 관계를 설명하면서 딥러닝을 비교 모델로 사용하는 방식을 고려할 수 있습니다. 이는 결과에서 도출한 운영상의 제안입니다.</li><li><strong>실제 미래 예측에는 추가 평가가 필요합니다.</strong> 이번 평가는 미래 국제가격·환율을 알고 계산했습니다. 두 입력 자체의 예측오차까지 반영하면 국내 가격 예측의 성능도 달라질 수 있습니다. 아직 사용하지 않은 이후 기간에서도 성능이 유지되는지 확인할 필요가 있습니다.</li><li><strong>정책 그래프의 격차를 예측오차나 확정 정책효과로 쓰면 안 됩니다.</strong> 여기의 성능 지표는 정책 이전 검증 결과입니다. 정책이 없었을 실제 가격은 관측할 수 없어, 정책 이후 추정선과 실제선의 차이는 과거 관계가 유지된다는 가정 아래의 시나리오 격차입니다.</li></ul><h3>본문용 해석</h3><p>환율을 포함한 동일 평가 조건에서 회귀와 딥러닝은 유종별로 다른 강점을 보였다. 경유와 휘발유는 회귀의 평균 절대오차가 더 작았으며, 등유는 딥러닝의 평균 절대오차가 작았지만 회귀의 RMSE가 낮았다. 따라서 평균적인 예측오차뿐 아니라 큰 오차, 예측 기간, 편향과 안정성을 함께 고려해야 한다. 다만 세 유종 모두 통계적으로 확실한 성능 차이가 확인되지 않아, 이번 결과를 고정된 모델 우위로 일반화하기는 어렵다.</p></section>',
             '<section><h2>근거 자료</h2><p>완료된 새 분석의 저장 결과로 작성했으며 모델을 다시 학습하지 않았습니다. 지표 원자료의 반올림 수준으로 숫자를 표시했습니다.</p><ul><li><a href="results/model_evaluation_normalized_exchange_rate/metrics.csv">전체 성능 지표</a></li><li><a href="results/timesfm_cv_normalized_exchange_rate/cv_summary.csv">기간별·기점별 요약과 통계 검정</a></li><li><a href="results/timesfm_cv_normalized_exchange_rate/cv_predictions.csv">예측 원자료</a></li></ul></section>']
    css = 'body{font:16px/1.8 "Malgun Gothic",sans-serif;background:#f1f5fa;color:#24334a;margin:0;word-break:keep-all}main{max-width:1250px;margin:auto;padding:24px}header,section{background:white;padding:28px;margin:22px 0;border-radius:12px}header{background:#17324f;color:white}h2,h3{color:#19517d}nav a,a{color:#176aa9}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #dbe4ef;padding:12px;text-align:left;min-width:95px}th{background:#edf4fd}li{margin:12px 0}@media print{body{background:white}section{padding:0}header{background:white;color:black}.scroll{overflow:visible}table{font-size:10px}th,td{min-width:0;padding:5px}tr{break-inside:avoid}}'
    html = '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>유종별 예측력 상세 분석</title><style>'+css+'</style></head><body><main>'+''.join(body)+'</main></body></html>'
    class Check(HTMLParser):
        def __init__(self):
            super().__init__(); self.stack=[]; self.links=[]; self.ids=set()
        def handle_starttag(self, tag, attrs):
            attrs=dict(attrs)
            if tag not in ('meta','br','img','hr','link','input'): self.stack.append(tag)
            if 'id' in attrs: self.ids.add(attrs['id'])
            if tag=='a': self.links.append(attrs['href'])
        def handle_endtag(self, tag):
            assert self.stack.pop()==tag
    check=Check(); check.feed(html)
    assert not check.stack and verified==6
    for link in check.links:
        assert link[1:] in check.ids if link.startswith('#') else (ROOT/link).is_file()
    TARGET.write_text(html,encoding='utf-8')
    print('검증 통과: 3개 유종·2개 모델의 MAE/MSE 원자료 일치, HTML 구조·링크 확인.')
    print(TARGET)


if __name__ == '__main__':
    main()
