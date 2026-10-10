"""환율·표준화 입력으로 정책 미시행 시나리오를 추정하고 6개 그래프를 저장합니다."""
import base64
import csv
import json
import sys
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '딥러닝'))
from load_model import DEFAULT_CHECKPOINT, load_model
from load_training_set import load_normalized_training_set, load_weekly_table, fit_regression, predict_regression
from run_timesfm import forecast

OUT = ROOT / 'results/policy_counterfactual_normalized_exchange_rate'
PRODUCTS = {'경유': 'diesel', '휘발유': 'gasoline', '등유': 'kerosene'}
LABELS = {'regression': '최근 마진 회귀', 'timesfm': 'TimesFM 딥러닝'}
COLORS = {'regression': '#1769ad', 'timesfm': '#7952b3'}
POLICY_START = pd.Timestamp('2026-03-13')


def policy_inputs(product, weekly):
    train = load_normalized_training_set(product, weekly, 'train')
    test = load_normalized_training_set(product, weekly, 'test')
    policy = load_normalized_training_set(product, weekly, 'policy')
    if not train['scaler'] == test['scaler'] == policy['scaler']:
        raise ValueError('정규화 파일의 척도가 구간별로 다릅니다.')
    prior_dates = train['날짜'] + test['날짜']
    dates = pd.to_datetime(policy['날짜'])
    if not all(pd.Timestamp(d) < POLICY_START for d in prior_dates) or not (dates >= POLICY_START).all():
        raise ValueError('정책 이후 타겟이 학습 자료에 포함되었습니다.')
    prior_y = np.concatenate([train['y'], test['y']]).astype(float)
    saved_mean = np.asarray(train['scaler']['mean'])
    saved_scale = np.asarray(train['scaler']['scale'])
    prior_raw = np.concatenate([train['X'], test['X']]) * saved_scale + saved_mean
    policy_raw = policy['X'] * saved_scale + saved_mean
    mean, std = prior_raw.mean(axis=0), prior_raw.std(axis=0)
    scale = np.where(std == 0, 1.0, std)
    prior_X, policy_X = (prior_raw - mean) / scale, (policy_raw - mean) / scale
    covariates = np.concatenate([prior_X, policy_X]).T.astype(np.float32)
    if covariates.shape != (4, len(prior_y) + len(dates)) or not np.isfinite(covariates).all():
        raise ValueError('국제가 3개 시차 및 환율 입력의 크기 또는 값이 잘못됐습니다.')
    return {'prior_X': prior_X, 'prior_y': prior_y, 'policy_X': policy_X,
            'covariates': covariates, 'scale': scale, 'mean': mean,
            'prior_dates': pd.to_datetime(prior_dates), 'dates': dates,
            'actual': np.asarray(policy['y'], dtype=float)}


def draw(product, name, data, prediction, bounds, path):
    color = COLORS[name]
    fig, ax = plt.subplots(figsize=(12, 6.4), layout='constrained')
    fig.set_facecolor('#fbfcfe')
    ax.set_facecolor('#fbfcfe')
    before = 8
    actual_dates = data['prior_dates'][-before:].append(data['dates'])
    actual_values = np.concatenate([data['prior_y'][-before:], data['actual']])
    ax.axvspan(POLICY_START, data['dates'][-1], color='#eaf0f7', alpha=.5)
    ax.fill_between(data['dates'], data['actual'], prediction,
                    where=prediction >= data['actual'], color=color, alpha=.12, interpolate=True)
    ax.fill_between(data['dates'], data['actual'], prediction,
                    where=prediction < data['actual'], color='#e49b3f', alpha=.15, interpolate=True)
    ax.plot(actual_dates, actual_values, color='#303c4c', lw=2, label='실제 세전 공급가격', zorder=3)
    ax.plot(data['dates'], prediction, color=color, lw=2.5,
            label='정책 미시행 가정의 모형 추정', zorder=4)
    ax.axvline(POLICY_START, color='#8995a5', ls='--', lw=1.2)
    ax.text(POLICY_START + pd.Timedelta(days=3), bounds[1] - .04 * (bounds[1] - bounds[0]),
            '정책 시행 3/13', color='#59677a', va='top', fontsize=10)
    ax.set_title(f'{LABELS[name]} · {product}\n정책 시행 이후, 정책이 없었다면의 가격 추정',
                 loc='left', fontsize=17, fontweight='bold', pad=15)
    ax.set_ylabel('세전 공급가격 (원/리터)', fontsize=11)
    ax.set_ylim(*bounds)
    ax.set_xlim(actual_dates[0], data['dates'][-1] + pd.Timedelta(days=7))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(axis='y', color='#dce3ed', lw=.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#c7d0db')
    ax.tick_params(colors='#526173', labelsize=10)
    ax.legend(loc='upper left', bbox_to_anchor=(0, .90), frameon=False, fontsize=10)
    gap = prediction - data['actual']
    fig.text(.01, -.045, f"정책 이전 {len(data['prior_y'])}주로 추정 · 실제 국제가격·환율 조건부 · 정책 이후 {len(prediction)}주\n"
             '음영은 모형 추정과 실제 가격의 차이이며, 인과적 정책효과나 정부 보전액을 의미하지 않습니다.',
             fontsize=10, color='#526173')
    ax.text(.99, .02, f'평균 가격 차이 {gap.mean():+.1f}원/L', transform=ax.transAxes,
            ha='right', fontsize=11, color=color,
            bbox={'facecolor': 'white', 'edgecolor': '#dce3ed', 'pad': 7})
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_documents(summaries, files, inputs):
    first = inputs[next(iter(PRODUCTS))]
    lines = ['# 정책 미시행 가격 추정: 회귀·딥러닝 그래프 6개', '',
             f"정책 시작일은 2026-03-13, 그래프의 정책 이후 관측 주는 {first['dates'][0].date()}~{first['dates'][-1].date()} ({len(first['dates'])}주)입니다.", '',
             '## 계산 조건', '',
             '- 기존 설치 TimesFM 모델과 새 csv/normalized의 train·test·policy 파일 및 scaler JSON을 사용했습니다. 국제가격 3개 시차와 같은 주 환율을 두 모델에 동일하게 제공했습니다.',
             f"- 모델 선택을 마친 뒤 정책 이전 전체 {len(first['prior_y'])}주(train+test)를 합쳐 사용했습니다. 이는 앞 80% 안에서 수행한 이전 교차검증과 다른 최종 정책 시나리오 추정입니다.",
             '- 저장된 표준화를 역변환하고 정책 이전 전체 자료만으로 Z-score를 다시 계산했습니다. 타겟은 원/리터로 유지했고 회귀의 국제가 계수 합 제약은 원래 단위에서 적용했습니다.',
             '- 회귀는 정책 직전까지의 자료로 한 번 적합하고, TimesFM은 같은 가격 이력을 문맥으로 제공합니다. 정책 이후 실제 국내 가격은 학습·문맥에 넣지 않고 비교선으로만 사용했습니다.',
             '- 정책 이후 실제 국제가격·환율은 두 모델의 입력으로 사용했습니다. 따라서 공변량이 알려진 조건부 시나리오입니다. 같은 유종의 두 그래프는 동일한 y축 범위로 표시했습니다.',
             '- 세전 공급가격을 비교합니다. 실제 세금·세후 가격의 정책 미시행 경로를 따로 추정한 것은 아닙니다.',
             '- 모형 관계가 정책이 없을 때에도 유지된다는 가정에 따른 추정입니다. 실제로 관측할 수 없는 미시행 가격의 정답은 없으므로 이를 MAE로 평가하거나 음영을 인과적 정책효과·정부 보전액으로 단정할 수 없습니다. 확률구간은 표시하지 않았습니다.', '',
             '## 정책 이후 가격 차이 요약', '',
             '차이 = 모형의 정책 미시행 가격 추정 − 실제 세전 공급가격. 단위는 원/리터입니다.', '',
             '| 모델 | 유종 | 평균 추정 가격 | 평균 실제 가격 | 평균 차이 | 마지막 주 추정 | 마지막 주 실제 | 마지막 주 차이 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in summaries:
        lines.append('| ' + ' | '.join([r['모델'], r['유종']] + [f"{r[k]:.2f}" for k in
                     ('평균_추정가격', '평균_실제가격', '평균_차이', '마지막_추정가격', '마지막_실제가격', '마지막_차이')]) + ' |')
    lines += ['', '## 그래프', '']
    for name, product, path in files:
        lines += [f'### {LABELS[name]} · {product}', '', f'![{LABELS[name]} {product}]({path.name})', '']
    lines += ['[주별 예측 및 실제 가격](predictions.csv)', '', f'사용 모델: `{DEFAULT_CHECKPOINT}`', '',
              '재실행: `.venv/Scripts/python regression/plot_policy_counterfactual.py`']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    sections = []
    for name, product, path in files:
        encoded = base64.b64encode(path.read_bytes()).decode('ascii')
        relative = path.relative_to(ROOT).as_posix()
        sections.append(f'<article><h3>{LABELS[name]} · {product}</h3><img src="data:image/png;base64,{encoded}" alt="{LABELS[name]} {product} 정책 미시행 추정"><p><a href="{escape(relative)}" download>PNG 다운로드</a></p></article>')
    intro = '<h2>정책 미시행 가격 추정 · 그래프 6개</h2><p class="note">회귀·딥러닝 각각 경유·휘발유·등유. 환율과 새 정규화 파일을 사용하고 정책 이전 578주로 추정했습니다. 검증 구간의 점수가 아닌 정책 미시행 조건부 시나리오입니다. 음영은 실제와 모형 추정의 차이이며 인과적 정책효과나 정부 보전액은 아닙니다.</p>'
    intro += f'<p>기간: {first["dates"][0].date()}~{first["dates"][-1].date()} · 세전 원/리터 · 같은 유종은 동일 y축 범위.</p><p><a href="{(OUT / "report.md").relative_to(ROOT).as_posix()}">계산 조건과 요약표</a></p>'
    fragment = intro + ''.join(sections)
    target = next(ROOT.glob('*.html'))
    original = target.read_bytes().decode('utf-8')
    start_marker, end_marker = '<!-- POLICY_CHARTS_START -->', '<!-- POLICY_CHARTS_END -->'
    start = original.find(start_marker)
    if start >= 0:
        end = original.index(end_marker, start) + len(end_marker)
        original = original[:start] + original[end:]
    nav_link = '<a href="#policy-charts">정책 미시행 그래프 6개</a>'
    if nav_link not in original:
        original = original.replace('</nav>', nav_link + '</nav>', 1)
    section = start_marker + '<section id="policy-charts">' + fragment + '</section>' + end_marker
    original = original.replace('</nav>', '</nav>' + section, 1)
    target.write_bytes(original.encode('utf-8'))
    gallery = fragment.replace(f'href="{OUT.relative_to(ROOT).as_posix()}/', 'href="')
    (OUT / 'gallery.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>정책 미시행 가격 추정 6개</title><style>body{font:16px/1.8 "Malgun Gothic",sans-serif;max-width:1200px;margin:auto;padding:24px;color:#24334a;background:#f3f6fb}article{background:white;padding:20px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}.note{background:#eaf1fa;padding:18px}a{color:#176aa9}</style><body>' + gallery + '</body></html>', encoding='utf-8')


def main():
    import torch
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False})
    OUT.mkdir(parents=True, exist_ok=True)
    weekly = load_weekly_table()
    model = load_model(DEFAULT_CHECKPOINT, threads=4)
    inputs, predictions = {}, {}
    with torch.inference_mode():
        for product in PRODUCTS:
            data = policy_inputs(product, weekly)
            params = fit_regression(data['prior_X'], data['prior_y'], feature_scale=data['scale'])
            regression = np.asarray(predict_regression(params, data['policy_X']))
            neural, _, _ = forecast(model, data['prior_y'].astype(np.float32), data['covariates'], len(data['dates']))
            inputs[product] = data
            predictions[product] = {'regression': regression, 'timesfm': neural}
            print(f'{product}: 정책 기간 {len(neural)}주 추정 완료', flush=True)
    rows, summaries, files = [], [], []
    for name in LABELS:
        for product, slug in PRODUCTS.items():
            data, predicted = inputs[product], predictions[product][name]
            combined = np.concatenate([data['prior_y'][-8:], data['actual'], *predictions[product].values()])
            padding = max(60, np.ptp(combined) * .12)
            bounds = (float(combined.min() - padding), float(combined.max() + padding))
            path = OUT / f'{name}_{slug}.png'
            draw(product, name, data, predicted, bounds, path)
            files.append((name, product, path))
            gap = predicted - data['actual']
            summaries.append({'모델': LABELS[name], '유종': product, '평균_추정가격': float(predicted.mean()),
                              '평균_실제가격': float(data['actual'].mean()), '평균_차이': float(gap.mean()),
                              '마지막_추정가격': float(predicted[-1]), '마지막_실제가격': float(data['actual'][-1]),
                              '마지막_차이': float(gap[-1])})
            for day, actual, estimate in zip(data['dates'], data['actual'], predicted, strict=True):
                rows.append({'모델': name, '유종': product, '날짜': day.date().isoformat(),
                             '실제_세전가격': actual, '미시행_추정가격': estimate, '차이': estimate - actual})
    for filename, data in [('predictions.csv', rows), ('summary.csv', summaries)]:
        with (OUT / filename).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    save_documents(summaries, files, inputs)
    (OUT / 'run_manifest.json').write_text(json.dumps({'checkpoint': str(DEFAULT_CHECKPOINT),
        'policy_start': '2026-03-13', 'prior_rows': len(next(iter(inputs.values()))['prior_y']),
        'policy_rows': len(next(iter(inputs.values()))['dates']), 'features': ['intl_t', 'intl_t-1', 'intl_t-2', 'exchange_rate_t'],
        'normalized_files': str(ROOT / 'csv/normalized')}, ensure_ascii=False, indent=2), encoding='utf-8')
    # 관련 검증은 이 실행에서 한 번: 그림 수·예측값·학습 경계·두 모형의 평가 날짜.
    assert len(files) == 6 and len({p.name for _, _, p in files}) == 6
    assert all(p.is_file() and p.stat().st_size > 10000 for _, _, p in files)
    assert all(np.isfinite(v).all() for pair in predictions.values() for v in pair.values())
    assert all(data['prior_dates'].max() < POLICY_START for data in inputs.values())
    assert len(rows) == 6 * len(next(iter(inputs.values()))['dates'])
    print(f'검증 통과: 그래프 6개 및 주별 예측 {len(rows)}행. 저장: {OUT}', flush=True)


if __name__ == '__main__':
    main()
