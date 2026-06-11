"""Скачивает gemma-4-E2B-it с Hugging Face в ./weights.

Запустите ОДИН раз локально перед сборкой образа:

    pip install -U "transformers==5.5.0"
    python3 download_weights.py

После этого папка ./weights попадает в docker image через `COPY . .` и
оказывается в /workspace/weights внутри контейнера. На проверяющем сервере
интернета нет (--network none), поэтому веса обязательно должны быть
внутри образа.
"""
from huggingface_hub import snapshot_download


REPO_ID = "google/gemma-4-E2B-it"
LOCAL_DIR = "weights"


def main() -> None:
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
    )
    print(f"weights downloaded to: {path}")


if __name__ == "__main__":
    main()
