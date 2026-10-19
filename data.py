"""VSR JSONL loading. Images stay in native RGB pixel coordinates."""

import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils import sha256


def load_rows(path):
    rows = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if not {"image", "caption", "label"} <= r.keys():
                raise ValueError(f"Missing VSR fields at {path}:{line_no}")
            if r["label"] not in (0, 1):
                raise ValueError(f"Invalid binary label at line {line_no}")
            name = Path(r["image"]).name
            if name != r["image"]:
                raise ValueError("VSR image field must be a filename, not a path.")
            r["id"] = hashlib.sha256(
                f"{name}\0{r['caption']}\0{r['label']}".encode()
            ).hexdigest()[:24]
            rows.append(r)
    return rows


class VSRDataset(Dataset):
    def __init__(self, root, track="random", split="dev", limit=None):
        self.root = Path(root)
        self.manifest = self.root / track / f"{split}.jsonl"
        self.rows = load_rows(self.manifest)
        if limit is not None:
            if limit <= 0:
                raise ValueError("--limit must be positive")
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError("Empty dataset")
        self.manifest_hash = sha256(self.manifest)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = dict(self.rows[index])
        path = self.root / "images" / r["image"]
        if not path.exists():
            raise FileNotFoundError(
                f"{path}: run prepare_data.py --download-images or copy the COCO image here."
            )
        with Image.open(path) as image:
            # No resize or crop here: attacks precede every model-specific view.
            arr = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        r["pixels"] = torch.from_numpy(arr).permute(2, 0, 1).float().div(255)
        return r


def audit_splits(root, track, splits=("train", "dev", "test")):
    sets = {s: load_rows(Path(root) / track / f"{s}.jsonl") for s in splits}
    out = {
        "track": track,
        "counts": {s: len(rs) for s, rs in sets.items()},
        "manifest_sha256": {s: sha256(Path(root) / track / f"{s}.jsonl") for s in sets},
        "within_split_duplicate_examples": {},
        "overlaps": {},
    }
    for s, rs in sets.items():
        out["within_split_duplicate_examples"][s] = len(rs) - len({r["id"] for r in rs})
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out["overlaps"][a + "__" + b] = {
                "exact_examples": len(
                    {r["id"] for r in sets[a]} & {r["id"] for r in sets[b]}
                ),
                "image_caption_pairs": len(
                    {(r["image"], r["caption"]) for r in sets[a]}
                    & {(r["image"], r["caption"]) for r in sets[b]}
                ),
                "images": len(
                    {r["image"] for r in sets[a]} & {r["image"] for r in sets[b]}
                ),
            }
    return out
