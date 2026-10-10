"""다운로드한 TimesFM 3.0 모델을 로컬 파일에서 불러옵니다.

실행: python 딥러닝/load_model.py --device cpu
모델 및 의존성 준비: download_model.py의 안내 참고.
"""

import argparse
from pathlib import Path


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "모델"
    / "timesfm-3.0-pytorch"
)
EXISTING_CHECKPOINT = Path(__file__).resolve().parents[1].parent / '경제 헤커톤/모델/timesfm-3.0-pytorch'
if (EXISTING_CHECKPOINT / 'model.safetensors').is_file():
    DEFAULT_CHECKPOINT = EXISTING_CHECKPOINT


def load_model(checkpoint_path=DEFAULT_CHECKPOINT, *, device="cpu", threads=4):
    """예측 코드에서 재사용할 TimesFM3Evaluator를 반환합니다."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device는 cpu 또는 cuda여야 합니다.")
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads는 양의 정수여야 합니다.")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    for name in ("config.json", "model.safetensors"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"로컬 모델 파일이 없습니다: {checkpoint / name}")

    try:
        import torch
        from timesfm3 import ModelConfig, TimesFM3Evaluator
    except ImportError as exc:
        raise ImportError(
            "TimesFM 실행 환경이 필요합니다. 딥러닝/download_model.py의 "
            "의존성 설치 안내를 확인하세요."
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA GPU를 사용할 수 없습니다. device='cpu'를 사용하세요.")
    torch.set_num_threads(threads)
    return TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=str(checkpoint),
            per_core_batch_size=1,
            device=device,
            local_files_only=True,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        load_model(args.checkpoint, device=args.device, threads=args.threads)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        parser.exit(1, f"모델 로딩 실패: {exc}\n")
    print(f"TimesFM 3.0 로딩 완료: {args.checkpoint.resolve()} ({args.device})")


if __name__ == "__main__":
    main()
