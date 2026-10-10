"""다른 컴퓨터에서 TimesFM 3.0 모델을 준비합니다.

다운로드 의존성: python -m pip install huggingface_hub
실행: python 딥러닝/download_model.py
다른 저장 위치: python 딥러닝/download_model.py --checkpoint 경로
로딩: python 딥러닝/load_model.py --device cpu
다른 저장 위치로 다운로드했다면 로딩에도 동일한 --checkpoint 경로를 지정하세요.

로딩 의존성(기존 환경과 동일한 TimesFM 소스 버전):
python -m pip install psutil "timesfm[torch] @ https://github.com/google-research/timesfm/archive/e51928e27119cb17bebc005be2696b75e0a9e688.zip"
"""

import argparse
from pathlib import Path

from load_model import DEFAULT_CHECKPOINT


MODEL_ID = "google/timesfm-3.0-pytorch"


def download_model(checkpoint_path=DEFAULT_CHECKPOINT):
    """필수 모델 파일을 다운로드하고 저장 경로를 반환합니다."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "다운로드 의존성이 필요합니다: python -m pip install huggingface_hub"
        ) from exc

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(checkpoint),
        allow_patterns=["config.json", "model.safetensors"],
    )
    for name in ("config.json", "model.safetensors"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"다운로드한 모델 파일이 없습니다: {checkpoint / name}")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    try:
        checkpoint = download_model(args.checkpoint)
    except (ImportError, OSError, ValueError) as exc:
        parser.exit(1, f"모델 다운로드 실패: {exc}\n")
    print(f"TimesFM 3.0 다운로드 완료: {checkpoint}")


if __name__ == "__main__":
    main()
