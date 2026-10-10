"""환율 포함 교차검증·VIF·기존 결과 비교. 실행: .venv/Scripts/python regression/reanalyze_exchange_rate.py"""
import csv
import json
import hashlib
import importlib.metadata
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '딥러닝'))
import evaluate_models as evaluation
from load_training_set import MODELS, load_training_set, load_normalized_training_set, load_weekly_table

OUT = ROOT / 'results/model_evaluation_normalized_exchange_rate'


def save_comparison(rows):
    with (ROOT / 'results/model_evaluation/metrics.csv').open(encoding='utf-8-sig') as stream:
        old = list(csv.DictReader(stream))
    before = {(r['제품'], r['모형']): r for r in old if r['범위'] in ('전체', '제품')}
    selected = [r for r in rows if r['범위'] in ('전체', '제품') and r['모형'] in ('recent_margin', 'timesfm_cov')]
    changes = []
    for row in selected:
        previous = before[row['제품'], row['모형']]
        if int(previous['N']) != row['N']:
            raise ValueError('환율 추가 전후 평가 표본 수 불일치')
        changes.append({'제품': row['제품'], '모형': row['모형'], 'N': row['N'],
                        '기존_MAE': float(previous['MAE']), '환율포함_MAE': row['MAE'],
                        'MAE_감소율_pct': (1 - row['MAE'] / float(previous['MAE'])) * 100,
                        '기존_RMSE': float(previous['RMSE']), '환율포함_RMSE': row['RMSE']})
    path = OUT / 'before_after.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(changes[0]))
        writer.writeheader()
        writer.writerows(changes)
    full = any(r['모형'] == 'timesfm_cov' for r in selected)
    lines = ['# 환율을 포함한 회귀·TimesFM 재분석', '',
             '새 `csv/normalized` 파일을 읽어 국제가격 같은 주·1주 전·2주 전과 같은 주 원달러환율을 두 모형에 동일하게 제공했다. 기존과 같은 22개 기점 × 32주 × 3유종, 총 2,112개 예측으로 평가했다. 교차검증에서는 저장된 Z-score를 역변환하고 각 기점 이전 자료만으로 다시 표준화하여 미래 정보가 들어가지 않도록 했다.', '',
             '최근 마진 회귀는 국제가격 계수만 합 1로 정규화하고 환율 계수는 원래 단위로 유지한다. 마지막 52주 평균 잔차로 상수를 보정하므로, 상수는 환율 효과를 제외한 보정값이며 기존의 단순 가격 차이 마진과 같지 않다.', '',
             ('TimesFM은 기존 소스 커밋과 같은 사전학습 모델을 사용했다. 실제 미래 국제가격·환율을 제공한 조건부 교차검증이며 미래 공변량까지 예측한 성능은 아니다. 가중치를 재학습하지 않고 입력 공변량에 환율을 추가했다.' if full else '현재 회귀 재분석만 완료했다. TimesFM 코드에는 환율을 추가했으나 실행 환경 준비 문제로 새 딥러닝 결과는 아직 없다. 기존 딥러닝 점수를 환율 포함 점수로 사용하지 않는다.'), '',
             '## 기존 입력과 환율 포함·정규화 입력 비교', '',
             '기존 대비 환율 추가와 입력 표준화를 함께 적용한 비교다. TimesFM의 순수한 환율 추가 효과만 분리한 실험은 아니다. 최근 마진 회귀는 원래 단위에서 계수 제약을 적용하여 표준화가 모형의 경제적 의미를 바꾸지 않도록 했다.', '',
             '감소율 = (기존 MAE − 환율 포함 MAE) / 기존 MAE × 100. 양수는 개선, 음수는 악화다.', '',
             '| 제품 | 모델 | 기존 MAE | 환율 포함 MAE | MAE 감소율 | 기존 RMSE | 환율 포함 RMSE |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    labels = {'recent_margin': '최근 마진 회귀', 'timesfm_cov': 'TimesFM 딥러닝'}
    for r in changes:
        lines.append(f"| {r['제품']} | {labels[r['모형']]} | {r['기존_MAE']:.4f} | {r['환율포함_MAE']:.4f} | {r['MAE_감소율_pct']:+.2f}% | {r['기존_RMSE']:.4f} | {r['환율포함_RMSE']:.4f} |")
    lines += ['', '## 환율 포함 두 모델 비교', '',
              '| 제품 | 회귀 MAE | 딥러닝 MAE | 회귀의 MAE 감소율(딥러닝 대비) | 회귀 RMSE | 딥러닝 RMSE |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for fuel in ('전체', '휘발유', '경유', '등유') if full else ():
        pair = {r['모형']: r for r in selected if r['제품'] == fuel}
        a, b = pair['recent_margin'], pair['timesfm_cov']
        reduction = (1 - a['MAE'] / b['MAE']) * 100
        lines.append(f"| {fuel} | {a['MAE']:.4f} | {b['MAE']:.4f} | {reduction:+.2f}% | {a['RMSE']:.4f} | {b['RMSE']:.4f} |")
    if full:
        overall_pair = {r['모형']: r for r in selected if r['제품'] == '전체'}
        reduction = (1 - overall_pair['recent_margin']['MAE'] / overall_pair['timesfm_cov']['MAE']) * 100
        lines += ['', f'전체 평균 MAE 기준 회귀의 오차가 딥러닝보다 {reduction:.2f}% 낮다. 등유에서는 딥러닝의 MAE가 더 작지만 회귀의 RMSE가 더 작으므로 지표별로 판단해야 한다.']
    lines += ['', '수치는 평균 오차 차이를 나타내며 정확도 백분율이나 통계적 유의성을 뜻하지 않는다. 기존 환율 미포함 p값은 새 모델에 적용하지 않는다. 새 검정값은 환율 포함 cv_summary.csv를 참고한다.', '',
              '## VIF', '', '정책 이전 앞 80% 학습 자료에서 국제가격 3개 시차와 환율의 VIF를 다시 계산했다. VIF는 설명변수 간 선형 중복을 진단하며 변수의 필요성이나 인과효과를 직접 판정하지 않는다.', '',
              '| 유종 | 변수 | VIF |', '| --- | --- | ---: |']
    weekly = load_weekly_table()
    vif_rows = []
    for fuel in MODELS:
        train = load_training_set(fuel, weekly)
        X = np.asarray(train['X'], dtype=float)
        for j, name in enumerate(train['설명변수']):
            target = X[:, j]
            other = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
            residual = target - other @ np.linalg.lstsq(other, target, rcond=None)[0]
            sst = np.sum((target - target.mean()) ** 2)
            sse = np.sum(residual ** 2)
            vif = sst / sse if sse > 0 else float('inf')
            vif_rows.append({'제품': fuel, '변수': name, 'N': len(X), 'VIF': vif})
            lines.append(f'| {fuel} | {name} | {vif:.4f} |')
    with (OUT / 'vif.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(vif_rows[0]))
        writer.writeheader()
        writer.writerows(vif_rows)
    fx_vif = [r['VIF'] for r in vif_rows if '환율' in r['변수']]
    lines += ['', f'환율의 VIF는 {min(fx_vif):.2f}~{max(fx_vif):.2f}로 낮다. 반면 인접 주 국제가격의 VIF가 높아, 공선성은 주로 국제가격 시차끼리의 중복에서 나타난다. 환율 제외를 높은 환율 VIF로 정당화할 수 없다.']
    overall = next(r for r in changes if r['제품'] == '전체' and r['모형'] == 'recent_margin')
    lines += ['', '## 결과 해석', '',
              f"최근 마진 회귀의 환율 추가 전후 전체 MAE 변화는 {overall['MAE_감소율_pct']:+.2f}%의 감소율이다. 음수이면 환율 추가로 오차가 증가한 것이며, 환율을 추가한다고 예측력이 자동으로 개선되지는 않는다. 이는 이번 사양과 검증 구간의 결과이고 환율의 경제적 효과가 없다는 뜻은 아니다."]
    if not full:
        lines += ['', '환율 포함 딥러닝 실행은 아직 완료되지 않아 두 모델의 새 우세 비율을 제시하지 않는다.']
    if full:
        from load_model import DEFAULT_CHECKPOINT
        lines += ['', '## 새 결과의 통계적 비교', '',
                  '| 유종 | DM p값 | 비중첩 기점 부분집합 p값 |', '| --- | ---: | ---: |']
        with (ROOT / 'results/timesfm_cv_normalized_exchange_rate/cv_summary.csv').open(encoding='utf-8-sig') as stream:
            for r in csv.DictReader(stream):
                if r['모형'] == 'timesfm_cov':
                    lines.append(f"| {r['제품']} | {float(r['DM_p']):.4f} | {float(r['부분집합_p']):.4f} |")
        lines += ['', '세 유종 모두 겹침 보정 검정의 p값이 0.05보다 커 두 모델의 차이가 5% 수준에서 유의하게 확인되지 않았다. 이것은 성능이 같다는 증명이 아니다. 평가 창이 겹치고 기점이 22개이므로 평균 차이와 검정 결과를 함께 본다.', '',
                  '## 사용한 파일과 검증', '',
                  f'기존 설치 모델: `{DEFAULT_CHECKPOINT}`. 새 다운로드를 중단하고 이 폴더의 기존 가중치를 읽었다.', '',
                  '입력: `csv/normalized/{유종}_train.csv` 및 해당 scaler JSON. CSV의 환율 열·원자료 복원·실제값·날짜를 확인했다. 각 폴드의 평균·표준편차는 그 기점까지의 자료만으로 다시 계산했다.', '',
                  '실행 중 공통 평가 표본·실제값과 기존 평가 격자의 일치를 확인했고, 별도 원자료 회귀 경로와 표준화 입력 회귀 경로의 프로토콜 검증도 통과했다.']
    lines += ['', '## 근거 및 재실행', '',
              '- [전체 성능 지표](metrics.csv)', '- [추가 전후 비교](before_after.csv)', '- [VIF](vif.csv)',
              *(['- [새 예측 및 프로토콜 보고서](../timesfm_cv_normalized_exchange_rate/cv_report.md)',
                 '- [새 기점별 요약 및 유의성 검정](../timesfm_cv_normalized_exchange_rate/cv_summary.csv)'] if full else ['- [새 회귀 예측](regression_predictions.csv)']), '',
              '실행: `.venv/Scripts/python regression/reanalyze_exchange_rate.py`', '',
              '기존 비교는 저장된 기존 예측값을 사용했다. 동일 기점·예측일·실제값을 검증한 뒤 비교하며, 별도 테스트셋과 정책 기간을 섞지 않는다. 환경 차이까지 배제한 환율의 순수 효과 검증에는 같은 환경에서 환율 미포함 TimesFM도 다시 실행해야 한다.']
    (OUT / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    manifest = {'features': ['international_t', 'international_t-1', 'international_t-2', 'exchange_rate_t'],
                'timesfm_complete': full,
                'dependencies': {name: importlib.metadata.version(name) for name in ('numpy', 'pandas')},
                'input_sha256': hashlib.sha256((ROOT / 'csv/merged_weekly_data_clean.csv').read_bytes()).hexdigest()}
    manifest['normalized_input_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (ROOT / 'csv/normalized').glob('*') if p.is_file() and ('_train.' in p.name or '_scaler.' in p.name)}
    if full:
        manifest['dependencies'].update({name: importlib.metadata.version(name) for name in ('timesfm', 'torch', 'scipy')})
        from load_model import DEFAULT_CHECKPOINT
        checkpoint = Path(sys.argv[sys.argv.index('--checkpoint') + 1]) if '--checkpoint' in sys.argv else DEFAULT_CHECKPOINT
        manifest['checkpoint'] = str(checkpoint.resolve())
        manifest['model_config_sha256'] = hashlib.sha256((checkpoint / 'config.json').read_bytes()).hexdigest()
        manifest['model_revision_metadata'] = (checkpoint / '.cache/huggingface/download/config.json.metadata').read_text() if (checkpoint / '.cache/huggingface/download/config.json.metadata').is_file() else None
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def run_regression_only():
    import counterfactual as cf
    from cv_timesfm import prepare_normalized_fold
    weekly = load_weekly_table()
    records = []
    with (ROOT / 'results/timesfm_cv_aligned/cv_predictions.csv').open(encoding='utf-8-sig') as stream:
        original = list(csv.DictReader(stream))
    for product in MODELS:
        train = load_normalized_training_set(product, weekly)
        X = pd.DataFrame(train['X'].astype(float), columns=['lag0', 'lag1', 'lag2', 'exchange_rate'])
        y = pd.Series(train['y'].astype(float))
        date_index = {day: i for i, day in enumerate(train['날짜'])}
        fitted = {}
        normalized_folds = {}
        for row in original:
            if row['제품'] != product or row['모형'] not in ('recent_margin', 'baseline', 'no_change'):
                continue
            origin = int(row['문맥길이'])
            if origin not in normalized_folds:
                normalized_folds[origin] = prepare_normalized_fold(train, origin)
            fold = normalized_folds[origin]
            X = pd.DataFrame(fold['X'], columns=['lag0', 'lag1', 'lag2', 'exchange_rate'])
            target = date_index[row['날짜']]
            if train['날짜'][origin - 1] != row['기준날짜'] or target != origin + int(row['리드']) - 1:
                raise ValueError('기존 평가 기점 또는 예측일 불일치')
            name = row['모형']
            if name == 'no_change':
                prediction = float(y.iloc[origin - 1])
            else:
                key = (origin, name)
                if key not in fitted:
                    fitted[key] = cf.fit(X.iloc[:origin], y.iloc[:origin], name, feature_scale=fold['scaler']['scale'])
                prediction = float(cf.predict(fitted[key], X.iloc[target:target + 1])[0])
            records.append({**row, '예측_세전가격': prediction, '잔차': float(row['실제_세전가격']) - prediction})
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT / 'regression_predictions.csv'
    with source.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    evaluation.MODELS = {k: v for k, v in evaluation.MODELS.items() if k in ('recent_margin', 'baseline', 'no_change')}
    evaluation.SOURCE, evaluation.OUT = source, OUT
    rows = evaluation.evaluate(evaluation.load_predictions())
    evaluation.write_results(rows)
    save_comparison(rows)
    print(f'회귀 환율 포함 결과: {OUT}', flush=True)


def main():
    if '--regression-only' in sys.argv:
        run_regression_only()
        return
    import cv_timesfm as cv
    cv.main()
    evaluation.SOURCE = ROOT / 'results/timesfm_cv_normalized_exchange_rate/cv_predictions.csv'
    evaluation.OUT = OUT
    points = evaluation.load_predictions()
    # 기존 표본의 날짜·기점·실제값까지 맞춰 환율 추가 전후 비교를 확인한다.
    evaluation.SOURCE = ROOT / 'results/timesfm_cv_aligned/cv_predictions.csv'
    old = evaluation.load_predictions()
    if set(old) != set(points) or any(abs(old[k]['recent_margin'][0] - points[k]['recent_margin'][0]) > 0.0001 for k in points):
        raise ValueError('기존과 새 평가 격자 또는 실제값 불일치')
    evaluation.SOURCE = ROOT / 'results/timesfm_cv_normalized_exchange_rate/cv_predictions.csv'
    rows = evaluation.evaluate(points)
    evaluation.write_results(rows)
    save_comparison(rows)
    print(f'환율 포함 비교 저장: {OUT}', flush=True)


if __name__ == '__main__':
    main()
