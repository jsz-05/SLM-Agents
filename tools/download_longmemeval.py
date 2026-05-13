from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


FILES = {
    "oracle": (
        "longmemeval_oracle.json",
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    ),
    "s": (
        "longmemeval_s_cleaned.json",
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    ),
    "m": (
        "longmemeval_m_cleaned.json",
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json",
    ),
}


def _download(url: str, output: Path) -> None:
    if output.exists() and output.stat().st_size > 0:
        print(f"Already exists: {output}")
        return

    print(f"Downloading {url}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, output.open("wb") as f:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                pct = 100 * downloaded / total_bytes
                print(f"\r{downloaded / 1_000_000:.1f} MB / {total_bytes / 1_000_000:.1f} MB ({pct:.1f}%)", end="")
        print("")
    print(f"Saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LongMemEval cleaned JSON files from Hugging Face.")
    parser.add_argument(
        "--out-dir",
        default="../LongMemEval/data",
        help="Directory where LongMemEval JSON files should be saved.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        choices=sorted(FILES),
        default=["oracle"],
        help="Which LongMemEval files to download. Start with oracle for quick local experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    for name in args.files:
        filename, url = FILES[name]
        _download(url, out_dir / filename)


if __name__ == "__main__":
    main()
