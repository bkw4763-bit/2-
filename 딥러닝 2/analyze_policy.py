"""정책 미시행 가격의 모델 비교·국제가격 시나리오·문맥 민감도를 생성한다."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import run_regression_timesfm as experiment

DEFAULT_INPUT = experiment.OUT
CONTEXTS = (52, 104, 156)
SCENARIOS = {"down": -1, "flat": 0, "up": 1}
NAMES = {"down": "국제가 하락", "flat": "국제가 유지", "up": "국제가 상승"}


def scenario_data(weekly, direction, change):
    """관측 국제가격은 그대로 두고 미관측 주만 연말까지 지정 비율로 선형 변화시킨다."""
    expanded = experiment.policy_input(weekly)
    future = expanded.index > weekly.index.max()
    count = int(future.sum())
    if count:
        scale = 1 + direction * change * np.arange(1, count+1)/count
        for fuel in experiment.cfg.FUELS:
            column = f"{fuel}_intl_krw_l"
            expanded.loc[future, column] *= scale
    return expanded


def build_paths(model, weekly, chosen, change, fixed_context=False):
    """같은 시행 전 타겟 문맥으로 시나리오·문맥 조합을 예측하고 실제 공급가를 대조한다."""
    rows = []
    contexts = [chosen] if fixed_context else sorted(set(CONTEXTS + (chosen,)))
    for scenario, direction in SCENARIOS.items():
        expanded = scenario_data(weekly, direction, change)
        cutoff = expanded.index[expanded.in_policy][0]
        horizon = len(expanded.loc[expanded.index >= cutoff])
        for fuel in experiment.cfg.FUELS:
            X, y = experiment.regression.training_data(weekly, fuel)
            future_X = experiment.regression.lagged_intl(expanded, fuel).loc[cutoff:]
            regression_paths = {name: experiment.regression.predict(
                experiment.regression.fit(X, y, name), future_X) for name in experiment.regression.MODELS}
            for context in contexts:
                dates, values, _ = experiment.forecast(model, expanded, fuel, cutoff, horizon, context)
                for i, date in enumerate(dates):
                    tax = expanded.loc[date, f"{fuel}_tax"]
                    actual_pre = expanded.loc[date, f"{fuel}_supply_pretax"]
                    actual_after = expanded.loc[date, f"{fuel}_supply_aftertax"]
                    cap = expanded.loc[date, f"{fuel}_cap"]
                    after = values[i]*(1+experiment.cfg.VAT)+tax
                    margin = regression_paths["recent_margin"][i]*(1+experiment.cfg.VAT)+tax
                    baseline = regression_paths["baseline"][i]*(1+experiment.cfg.VAT)+tax
                    rows.append({"fuel": fuel, "scenario": scenario, "context_weeks": context,
                        "date": min(date, experiment.POLICY_END),
                        "week_start": max(date-pd.Timedelta(days=6), pd.Timestamp(experiment.cfg.POLICY_START)),
                        "full_week_end": date, "assumed": date > weekly.index.max(),
                        "intl_krw_l": expanded.loc[date, f"{fuel}_intl_krw_l"],
                        "timesfm_pretax": values[i], "timesfm_aftertax": after,
                        "recent_margin_aftertax": margin, "baseline_aftertax": baseline,
                        "actual_pretax": actual_pre, "actual_aftertax": actual_after, "cap": cap,
                        "timesfm_gap_to_actual": after-actual_after,
                        "regression_gap_to_actual": margin-actual_after,
                        "timesfm_gap_to_cap": after-cap, "regression_gap_to_cap": margin-cap})
        print(f"시나리오 {scenario}: {len(contexts)}개 문맥 완료", flush=True)
    return pd.DataFrame(rows)


def summarize(paths):
    """관측 가능한 실제 격차와 연말 가정 결과를 구분해 요약한다."""
    rows = []
    for (fuel, scenario, context), group in paths.groupby(["fuel", "scenario", "context_weeks"]):
        observed = group.dropna(subset=["actual_aftertax"])
        last = group.sort_values("date").iloc[-1]
        rows.append({"fuel": fuel, "scenario": scenario, "context_weeks": context,
            "observed_supply_weeks": len(observed),
            "mean_observed_timesfm_gap": observed.timesfm_gap_to_actual.mean(),
            "mean_observed_regression_gap": observed.regression_gap_to_actual.mean(),
            "year_end_timesfm_aftertax": last.timesfm_aftertax,
            "year_end_regression_aftertax": last.recent_margin_aftertax,
            "year_end_timesfm_gap_to_cap": last.timesfm_gap_to_cap})
    return pd.DataFrame(rows)


def plot_results(paths, chosen, out_dir, observed_end):
    """실제값은 관측 구간에만 표시하고 연말 가정 구간을 음영으로 구분한다."""
    cache = out_dir / ".matplotlib"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})

    def setup(ax, fuel):
        ax.set_title(experiment.cfg.FUELS[fuel]["ko"], loc="left")
        ax.set_ylabel("원/L (세후)")
        ax.axvspan(observed_end, experiment.POLICY_END, color="#eeeeee", alpha=0.7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m월"))
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for j, fuel in enumerate(experiment.cfg.FUELS):
        table = paths[(paths.fuel == fuel) & (paths.scenario == "flat") & (paths.context_weeks == chosen)]
        for col, label, color, style in [("actual_aftertax", "실제 공급가", "#2a78d6", "-"),
            ("recent_margin_aftertax", "최근 마진 회귀", "#555555", "-"),
            ("timesfm_aftertax", f"TimesFM {chosen}주", "#eb6834", "-"),
            ("cap", "상한 (향후 유지 가정)", "#888888", "--")]:
            axes[0, j].plot(table.date, table[col], label=label, color=color, ls=style)
        observed = table.dropna(subset=["actual_aftertax"])
        axes[1, j].plot(observed.date, observed.timesfm_gap_to_actual, label="TimesFM - 실제", color="#eb6834")
        axes[1, j].plot(observed.date, observed.regression_gap_to_actual, label="회귀 - 실제", color="#555555")
        axes[1, j].axhline(0, color="#aaaaaa", lw=0.8)
        for ax in axes[:, j]:
            setup(ax, fuel)
            ax.legend(fontsize=8)
    fig.suptitle("정책 미시행 공급가 비교 / 아래: 실제 공급가와의 격차\n음영: 국제가격·유류세·상한 유지 가정 구간")
    fig.savefig(out_dir / "model_comparison.png", dpi=160)
    plt.close(fig)

    for kind in ("scenarios", "contexts"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
        for ax, fuel in zip(axes, experiment.cfg.FUELS):
            if kind == "scenarios":
                selected = paths[(paths.fuel == fuel) & (paths.context_weeks == chosen)]
                for name, table in selected.groupby("scenario"):
                    ax.plot(table.date, table.timesfm_aftertax, label=NAMES[name])
            else:
                selected = paths[(paths.fuel == fuel) & (paths.scenario == "flat")]
                for context, table in selected.groupby("context_weeks"):
                    ax.plot(table.date, table.timesfm_aftertax, label=f"{context}주" + (" (선택)" if context == chosen else ""))
            setup(ax, fuel)
            ax.legend(fontsize=8)
        fig.suptitle("국제가격 시나리오별 TimesFM 미시행 공급가" if kind == "scenarios" else "입력 길이별 TimesFM 미시행 공급가 (국제가격 유지)")
        fig.savefig(out_dir / f"{kind}.png", dpi=160)
        plt.close(fig)


def write_report(summary, comparison, chosen, change, out_dir):
    """검증 성능과 시나리오 가정을 근거로 발표용 결론을 저장한다."""
    lines = ["# 정책 미시행 공급가격 분석", "",
        f"기간: {experiment.cfg.POLICY_START}~{experiment.POLICY_END.date()}, 선택 문맥: {chosen}주.", "",
        f"미관측 국제가격은 마지막 주간 가격에서 연말까지 -{change:.0%}, 유지, +{change:.0%}로 변화하는 가정이다.",
        "과거 관측 국제가격은 변경하지 않았으며 유류세·상한은 마지막 값을 유지했다.", "",
        "| 유종 | 최근 마진 회귀 테스트 MAE | TimesFM 테스트 MAE | 더 작은 오차 |",
        "|---|---:|---:|---|"]
    for fuel, info in experiment.cfg.FUELS.items():
        test = comparison[(comparison.split == "test") & (comparison.fuel == fuel)].set_index("model")
        reg, deep = test.loc["recent_margin", "mae"], test.loc["timesfm3", "mae"]
        lines.append(f"| {info['ko']} | {reg:.2f} | {deep:.2f} | {'TimesFM' if deep < reg else '최근 마진 회귀'} |")
    lines += ["", "테스트 MAE 단위는 세전 원/L이다. 유종별 테스트 성능을 기준으로 회귀와 TimesFM을 비교한다.",
        "", "| 유종 | 국제가격 하락 시 연말 공급가 | 유지 시 | 상승 시 |", "|---|---:|---:|---:|"]
    for fuel, info in experiment.cfg.FUELS.items():
        selected = summary[(summary.fuel == fuel) & (summary.context_weeks == chosen)].set_index("scenario")
        values = [selected.loc[s, "year_end_timesfm_aftertax"] for s in ("down", "flat", "up")]
        lines.append(f"| {info['ko']} | {values[0]:.0f} | {values[1]:.0f} | {values[2]:.0f} |")
    lines += ["", "연말 가격은 세후 원/L의 시나리오 결과이며 확정 전망이 아니다.",
        "가격 격차는 정부 손실보전액이 아니다. 미시행 가격과 실제 공급가의 차이에 다른 수급·마진 변화도 포함될 수 있다.",
        "공급가 공개 지연·사전학습 자료 중복은 반영하지 않았다."]
    (out_dir / "findings.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    """고정 설정을 재사용해 민감도 분석과 CSV·그림·발표용 요약을 생성한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--lead", type=int, choices=experiment.regression.EVALUATION_LEADS,
                        default=43, help="보고서에 사용할 예측 거리(주)")
    parser.add_argument("--change", type=float, default=0.10, help="연말 국제가격 상승·하락 비율, 기본 0.10")
    args = parser.parse_args()
    if not 0 < args.change < 1:
        parser.error("change는 0과 1 사이여야 합니다.")
    weekly = experiment.regression.load_weekly()
    settings = json.loads((args.input_dir / "locked_settings.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((experiment.cfg.OUT_DIR / "weekly_panel.csv").read_bytes()).hexdigest()
    if digest != settings["source_sha256"]:
        raise ValueError("고정 설정과 원자료가 다릅니다.")
    fixed_context = bool(settings.get("fixed_context_years"))
    chosen = int(settings.get("policy_actual_context_weeks", settings["chosen_context"]))
    out_dir = args.input_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    torch.set_num_threads(4)
    model = TimesFM3Evaluator(ModelConfig(checkpoint_path=str(experiment.ROOT / "모델" / "timesfm-3.0-pytorch"),
        per_core_batch_size=1, device="cpu", local_files_only=True))
    with torch.inference_mode():
        paths = build_paths(model, weekly, chosen, args.change, fixed_context)
    summary = summarize(paths)
    paths.to_csv(out_dir / "sensitivity_paths.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    plot_results(paths, chosen, out_dir, weekly.index.max())
    comparison = pd.read_csv(args.input_dir / "model_comparison.csv")
    comparison = comparison.loc[comparison.lead.eq(args.lead)]
    write_report(summary, comparison, chosen, args.change, out_dir)
    report = out_dir / "findings.md"
    report.write_text(report.read_text(encoding="utf-8").replace(
        "테스트 MAE 단위는", f"{args.lead}주 앞 예측을 비교했다. 테스트 MAE 단위는"), encoding="utf-8")
    if fixed_context:
        path = out_dir / "findings.md"
        text = path.read_text(encoding="utf-8")
        text = text.replace(f"선택 문맥: {chosen}주", f"15년 입력 고정, 확보된 실제 문맥: {chosen}주 (최적 길이 선택 없음)")
        path.write_text(text, encoding="utf-8")
    (out_dir / "assumptions.json").write_text(json.dumps({"international_price_year_end_change": args.change,
        "chosen_context_weeks": chosen, "fixed_context_years": settings.get("fixed_context_years"),
        "sensitivity_contexts_weeks": [chosen] if fixed_context else CONTEXTS,
        "source_sha256": digest, "observed_through_week": str(weekly.index.max().date()),
        "tax_and_cap": "last values held constant", "cv_rerun": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"분석 완료: {out_dir}")


if __name__ == "__main__":
    main()
