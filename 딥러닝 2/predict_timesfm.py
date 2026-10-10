"""로컬 TimesFM-3: 한 기준일 실행 또는 검증 기간 rolling-origin 평가."""
import argparse
import importlib.metadata
import json
import threading
import time

import numpy as np
import pandas as pd

import settings as cfg
from evaluation import score_folds, score_predictions
from make_windows import load_experiment, result_dir
from ai_protocol import lock_settings, load_locked_settings
from evaluation import rank_candidates
from rolling_origin import run_rolling_origin

CHECKPOINT = cfg.ROOT / "모델" / "timesfm-3.0-pytorch"
CODE_REVISION = "e51928e27119cb17bebc005be2696b75e0a9e688"


def model_inputs(context, known_future):
    """미래 국제가격·환율·세금 없이 과거 공변량과 확정된 달력만 전달한다."""
    dates = context.index.append(known_future.index)
    if context.index.max() >= known_future.index.min():
        raise ValueError("과거와 미래 입력 날짜가 겹칩니다.")
    if not dates.equals(pd.date_range(dates.min(), periods=len(dates), freq=cfg.TIME_FREQ)):
        raise ValueError("주간 입력 날짜가 연속적이지 않습니다.")
    target = context.supply_pretax.to_numpy(dtype=np.float32)
    past = context.drop(columns="supply_pretax").to_numpy(dtype=np.float32).T
    # 순환형 달력: 과거와 미래의 동일한 정의를 사용한다.
    week = dates.isocalendar().week.to_numpy(dtype=float)
    month = dates.month.to_numpy() - 1
    calendar = np.array([np.sin(2*np.pi*week/52.1775), np.cos(2*np.pi*week/52.1775),
                         np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)], dtype=np.float32)
    if not all(np.isfinite(x).all() for x in (target, past, calendar)):
        raise ValueError("입력에 결측 또는 무한대가 있습니다.")
    return target, past, calendar


class TimesFMPredictor:
    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.quantiles = []

    def __call__(self, context, known_future, metadata):
        target, past, calendar = model_inputs(context, known_future)
        outputs = list(self.forecaster.predict_batch(
            contexts=[target], horizon=len(known_future),
            past_only_covariates=[past], past_future_covariates=[calendar],
            return_quantiles=True, use_symmetric_averaging=False,
        ))
        if len(outputs) != 1:
            raise ValueError("모델 출력 개수가 잘못되었습니다.")
        output = outputs[0]
        forecast = np.asarray(output.forecast, dtype=float)
        quantiles = np.asarray(output.quantiles, dtype=float)
        levels = self.forecaster.config.quantiles
        if forecast.shape != (len(known_future),) or quantiles.shape != (len(known_future), len(levels)):
            raise ValueError("모델 예측 또는 분위수 배열 크기가 잘못되었습니다.")
        if not np.isfinite(forecast).all() or not np.isfinite(quantiles).all():
            raise ValueError("모델 예측에 결측 또는 무한대가 있습니다.")
        for i, date in enumerate(known_future.index):
            self.quantiles.append({"fuel": metadata["fuel"], "origin": metadata["origin"],
                                   "config_id": metadata["config_id"], "context_years": metadata["context_years"],
                                   "context_weeks": len(context), "date": date, "lead": i+1,
                                   **{f"q{round(q*100):02d}": float(quantiles[i, j]) for j, q in enumerate(levels)}})
        return forecast


def main(default_years="all"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "validation", "test"], default="validation")
    parser.add_argument("--years", default=default_years, help="all 또는 달력 연수 하나; smoke 외에는 전체 후보 사용")
    parser.add_argument("--origin", help="검증 기준일 YYYY-MM-DD; smoke 모드만 지원")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or (args.origin and args.mode != "smoke"):
        parser.error("threads는 양수이며 origin은 smoke 모드에서만 사용합니다.")
    for name in ("config.json", "model.safetensors"):
        if not (CHECKPOINT / name).is_file():
            parser.error(f"로컬 모델 파일이 없습니다: {CHECKPOINT / name}")
    data, all_windows, protocol = load_experiment()
    root = result_dir(protocol) / "timesfm3"
    lock_path = root / "locked_settings.json"
    split = "test" if args.mode == "test" else "validation"
    windows = all_windows.loc[all_windows.split == split].copy()
    if args.mode == "validation":
        if args.years != "all":
            parser.error("전체 후보 비교를 위해 validation은 --years all만 지원합니다.")
        if lock_path.exists():
            parser.error("설정이 이미 고정됐습니다. 기존 검증 결과를 확인하거나 --mode test로 실행하세요.")
    if args.mode == "test":
        if args.years != "all":
            parser.error("test 입력 길이는 고정 파일에서만 읽습니다.")
        if not lock_path.exists():
            parser.error("먼저 전체 validation을 완료해 설정을 고정하세요.")
        locked = load_locked_settings(lock_path, protocol)
        windows = windows.loc[windows.config_id == locked["config_id"]]
    if args.mode == "smoke":
        origin = pd.Timestamp(args.origin) if args.origin else windows.origin.min()
        windows = windows.loc[windows.origin == origin]
        if args.years != "all":
            try:
                year = int(args.years)
            except ValueError:
                parser.error("years는 all 또는 정수여야 합니다.")
            windows = windows.loc[windows.context_years == year]
    if windows.empty:
        parser.error("해당 조건의 검증 기준일이 없습니다.")
    out = root / args.mode
    if args.mode == "smoke":
        out = out / windows.origin.min().strftime("%Y-%m-%d") / f"years_{args.years}"
    if args.mode == "test" and out.exists():
        parser.error("최종 테스트는 한 번만 실행합니다. 기존 test 결과 또는 부분 결과를 확인하세요.")
    try:
        import psutil
        import torch
        from timesfm3 import ModelConfig, TimesFM3Evaluator
    except ImportError as exc:
        parser.error(f"먼저 딥러닝/setup_timesfm.ps1을 실행하세요: {exc}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA GPU를 사용할 수 없습니다. --device cpu를 사용하세요.")
    torch.set_num_threads(args.threads)
    out.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    stop = threading.Event()

    def sample_memory():
        while not stop.wait(0.1):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        forecaster = TimesFM3Evaluator(ModelConfig(checkpoint_path=str(CHECKPOINT),
            per_core_batch_size=1, device=args.device, local_files_only=True))
        loaded = time.perf_counter()
        parts, quantile_parts, timings = [], [], []
        for year, batch in windows.groupby("context_years", sort=True):
            began = time.perf_counter()
            predictor = TimesFMPredictor(forecaster)
            with torch.inference_mode():
                predictions = run_rolling_origin(data, batch, predictor, "timesfm3", split)
            parts.append(predictions)
            quantile_parts.append(pd.DataFrame(predictor.quantiles))
            timings.append({"context_years": int(year), "origins": int(batch.origin.nunique()),
                            "windows": len(batch), "seconds": time.perf_counter()-began})
            # 긴 CV를 실행할 때 완료한 후보 결과부터 확인 가능.
            predictions.to_csv(out / f"years_{year:02d}_predictions.csv", index=False, encoding="utf-8-sig")
            print(f"{year}년 {args.mode} 완료: {batch.origin.nunique()}개 기준일 / {timings[-1]['seconds']:.1f}초", flush=True)
        predictions = pd.concat(parts, ignore_index=True)
        finished = time.perf_counter()
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
    finally:
        stop.set()
        monitor.join()
    for name, table in {"predictions": predictions, "quantiles": pd.concat(quantile_parts, ignore_index=True),
                        "metrics": score_predictions(predictions), "fold_metrics": score_folds(predictions)}.items():
        table.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    manifest_path = CHECKPOINT / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    report = {"model": cfg.MODEL_ID, "checkpoint": str(CHECKPOINT), "checkpoint_manifest": manifest,
              "code_revision": CODE_REVISION, "timesfm_version": importlib.metadata.version("timesfm"),
              "torch_version": torch.__version__, "device": args.device, "threads": args.threads,
              "mode": args.mode, "context_years": sorted(windows.context_years.unique().tolist()), "horizon": max(cfg.HORIZONS),
              "experiment_id": protocol["experiment_id"],
              "time_frequency": cfg.TIME_FREQ, "context_unit": "calendar_years", "horizon_unit": "weekly_observations",
              "origins": int(windows.origin.nunique()), "windows": len(windows), "weight_updates": False,
              "target": "supply_pretax", "past_covariates": ["intl_krw_l"],
              "future_covariates": ["week_sin", "week_cos", "month_sin", "month_cos"],
              "symmetric_averaging": False, "load_seconds": loaded-started,
              "inference_seconds": finished-loaded, "sampled_peak_process_rss_mb": peak_rss[0]/1024**2}
    (out / "run_info.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(timings).to_csv(out / "timings.csv", index=False, encoding="utf-8-sig")
    if args.mode != "test":
        rank_candidates(predictions).to_csv(out / "ranking.csv", index=False, encoding="utf-8-sig")
    if args.mode == "validation":
        record = lock_settings(predictions, lock_path, protocol, all_windows)
        print(f"선택·고정: {record['context_years']}년, 검증 MAE {record['validation_mae']:.3f}")
    print(f"저장: {out}\n예측 {len(windows)}개 창 / 실행 {finished-started:.1f}초 / 최대 RSS 약 {peak_rss[0]/1024**2:.0f}MB")


if __name__ == "__main__":
    main()
