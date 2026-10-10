"""동일 교차검증 예측으로 회귀·TimesFM 성능 평가. 실행: python regression/evaluate_models.py"""
import csv
import argparse
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/timesfm_cv_normalized_exchange_rate/cv_predictions.csv'
OUT = ROOT / 'results/model_evaluation_normalized_exchange_rate'
MODELS = {
    'baseline': '회귀 전체 적합',
    'recent_margin': '회귀 최근 마진',
    'timesfm_cov': 'TimesFM 공변량 사용',
    'timesfm_uni': 'TimesFM 단변량',
    'no_change': '가격 유지 기준모형',
}


def metrics(actual, predicted):
    n = len(actual)
    error = [p - a for a, p in zip(actual, predicted)]
    mae = sum(abs(e) for e in error) / n
    mse = sum(e * e for e in error) / n
    mean = sum(actual) / n
    sst = sum((a - mean) ** 2 for a in actual)
    nonzero = [(a, e) for a, e in zip(actual, error) if a != 0]
    smape = [0 if a == p == 0 else 200 * abs(p - a) / (abs(a) + abs(p))
             for a, p in zip(actual, predicted)]
    total = sum(abs(a) for a in actual)
    return {'N': n, 'MAE': mae, 'MSE': mse, 'RMSE': math.sqrt(mse),
            'MAPE_pct': sum(abs(e / a) for a, e in nonzero) / len(nonzero) * 100 if nonzero else None,
            'MAPE_N': len(nonzero), 'sMAPE_pct': sum(smape) / n,
            'WAPE_pct': sum(abs(e) for e in error) / total * 100 if total else None,
            'R2': 1 - n * mse / sst if sst else None, 'Bias': sum(error) / n}


def load_predictions():
    points = defaultdict(dict)
    with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['제품'], int(row['폴드']), row['기준날짜'], row['날짜'], int(row['리드']))
            model = row['모형']
            if model not in MODELS or model in points[key]:
                raise ValueError(f'알 수 없거나 중복된 모형: {key}, {model}')
            actual, predicted = float(row['실제_세전가격']), float(row['예측_세전가격'])
            if not all(math.isfinite(v) for v in (actual, predicted)):
                raise ValueError(f'유효하지 않은 예측: {key}, {model}')
            points[key][model] = (actual, predicted)
    if not points:
        raise ValueError('예측 자료가 없습니다.')
    for key, values in points.items():
        if set(values) != set(MODELS):
            raise ValueError(f'모형별 평가 표본 불일치: {key}')
        if len({pair[0] for pair in values.values()}) != 1:
            raise ValueError(f'모형별 실제값 불일치: {key}')
    return points


def evaluate(points):
    groups = defaultdict(list)
    for key, predictions in points.items():
        fuel, fold, _, _, lead = key
        for group in [('전체', '전체', ''), ('제품', fuel, ''),
                      ('제품_폴드', fuel, fold), ('제품_리드', fuel, lead)]:
            groups[group].append(predictions)
    results = []
    for (scope, fuel, segment), samples in groups.items():
        actual = [sample['no_change'][0] for sample in samples]
        reference = metrics(actual, [sample['no_change'][1] for sample in samples])['MAE']
        for model, label in MODELS.items():
            scores = metrics(actual, [sample[model][1] for sample in samples])
            scores['MAE_skill'] = 1 - scores['MAE'] / reference if reference else None
            results.append({'범위': scope, '제품': fuel, '구간': segment,
                            '모형': model, '모형명': label, **scores})
    return results


def write_results(results):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        for row in results:
            writer.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()})
    lines = ['# 회귀·딥러닝 모델 성능 평가', '',
             f'원자료: `{SOURCE.relative_to(ROOT).as_posix()}`. 동일 제품·폴드·기준일·예측일·리드의 실제값과 평가 표본을 확인합니다. 공변량은 국제가격 3개 시차와 같은 주 환율입니다.', '',
             '지표: MAE(평균 절대오차), MSE(평균 제곱오차), RMSE(평균 제곱근 오차), MAPE(평균 절대백분율오차), sMAPE(대칭 절대백분율오차), WAPE(가중 절대백분율오차), R²(결정계수), Bias(평균 예측−실제), MAE skill(가격 유지 기준 대비 MAE 개선율).', '',
             'MAE·RMSE·Bias 단위는 원/리터, MSE는 (원/리터)²입니다. 백분율 오차는 %입니다. 오차 지표는 작을수록, R²·skill은 클수록 좋으며 Bias는 0에 가까울수록 좋습니다. 음수 Bias는 과소예측입니다.', '',
             'MAPE는 실제값 0을 제외하며 MAPE_N에 표본 수를 기록합니다. sMAPE는 실제·예측 모두 0이면 0으로 처리합니다. 정의할 수 없는 지표는 CSV에서 빈칸입니다.', '',
             '폴드의 예측 기간이 겹치므로 같은 날짜라도 기준일·리드가 다른 예측을 각각 채점합니다. 전체 결과는 모든 예측의 통합 지표이며, 제품·폴드·리드별 지표는 metrics.csv에 있습니다. 겹친 예측을 독립 표본으로 해석하거나 이 결과만으로 통계적 유의성을 주장하지 않습니다.', '',
             '미래 국제가격·환율의 실제값을 제공한 조건부 예측입니다. 별도 최종 테스트셋이나 미래 공변량까지 예측하는 실전 성능은 아닙니다.', '',
             '| 제품 | 모형 | N | MAE | RMSE | MAPE(%) | sMAPE(%) | WAPE(%) | R² | Bias | MAE skill |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in results:
        if row['범위'] not in ('전체', '제품'):
            continue
        values = [row['제품'], row['모형명'], str(row['N'])]
        values += ['—' if row[k] is None else f'{row[k]:.4f}' for k in
                   ('MAE', 'RMSE', 'MAPE_pct', 'sMAPE_pct', 'WAPE_pct', 'R2', 'Bias', 'MAE_skill')]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '## 제품별 MAE 최저 모형', '']
    for fuel in sorted({row['제품'] for row in results if row['범위'] == '제품'}):
        candidates = [row for row in results if row['범위'] == '제품' and row['제품'] == fuel and row['모형'] != 'no_change']
        best = min(candidates, key=lambda row: row['MAE'])
        lines.append(f"- {fuel}: {best['모형명']} (MAE {best['MAE']:.4f} 원/리터)")
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out-dir', type=Path, default=OUT)
    args = parser.parse_args()
    SOURCE, OUT = args.source.resolve(), args.out_dir.resolve()
    points = load_predictions()
    results = evaluate(points)
    write_results(results)
    print(f'평가 완료: 공통 예측 {len(points)}개, 지표 행 {len(results)}개 → {OUT}')
