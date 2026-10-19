"""Download official VSR annotations and, optionally, only referenced COCO images."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse
import requests
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download
from config import write_json
from data import load_rows, audit_splits


def download_image(row, directory):
    target = directory / row["image"]
    if target.exists():
        with Image.open(target) as im:
            im.verify()
        return
    url = row.get("image_link", "")
    if urlparse(url).hostname != "images.cocodataset.org":
        raise ValueError(f"Unexpected COCO image URL: {url}")
    # COCO annotation URLs commonly use HTTP. Preserve the published image origin.
    temporary = target.with_suffix(".part")
    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with open(temporary, "wb") as f:
                for block in response.iter_content(1024 * 1024):
                    f.write(block)
        with Image.open(temporary) as im:
            im.verify()
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--track", choices=["random", "zeroshot"], default="random")
    p.add_argument("--revision", default="main")
    p.add_argument("--download-images", action="store_true")
    p.add_argument(
        "--image-splits",
        nargs="+",
        choices=["train", "dev", "test"],
        default=["train", "dev"],
    )
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    repo = f"cambridgeltl/vsr_{a.track}"
    revision = HfApi().dataset_info(repo, revision=a.revision).sha
    folder = a.data_dir / a.track
    folder.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        cached = Path(
            hf_hub_download(
                repo, f"{split}.jsonl", repo_type="dataset", revision=revision
            )
        )
        dest = folder / f"{split}.jsonl"
        if dest.exists() and dest.read_bytes() != cached.read_bytes():
            raise FileExistsError(
                f"{dest} differs; use a new data directory for this revision."
            )
        dest.write_bytes(cached.read_bytes())
    report = audit_splits(a.data_dir, a.track)
    report.update(dataset_id=repo, revision=revision)
    write_json(folder / "audit.json", report)
    if a.download_images:
        rows = {
            r["image"]: r
            for s in a.image_splits
            for r in load_rows(folder / f"{s}.jsonl")
        }
        images = a.data_dir / "images"
        images.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for i, _ in enumerate(
                pool.map(lambda r: download_image(r, images), rows.values()), 1
            ):
                if i % 100 == 0:
                    print(f"Downloaded/verified {i}/{len(rows)} images", flush=True)
    print(f"Prepared {repo} at {revision}; audit: {folder / 'audit.json'}")


if __name__ == "__main__":
    main()
