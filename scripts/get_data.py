"""Download the benchmark subsets and write manifests. Nothing here is committed (data/ is gitignored).

Edited-vs-authentic (task "edit"), from the Hugging Face mirror
ductai199x/image-manipulation-dataset-compilation (every image is re-saved there as PNG,
original file name kept):
    CASIA v2.0   2 authentic shards + 2 tampered shards   -> calibration/test split
    Columbia     authentic + spliced                       -> external test only
    Coverage     authentic + copy-move                     -> external test only

AI-generated-vs-real (task "ai"), from TheKernel01/Tiny-GenImage (CC BY-NC-SA 4.0):
    train shard 0      -> calibration split
    validation shard 0 -> held-out test split

Usage:  python scripts/get_data.py
Total download is about 1.4 GB.
"""
import csv
import io
import random
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 1337

COMPILATION = "ductai199x/image-manipulation-dataset-compilation"
EDIT_SHARDS = {
    "casia": ["CASIA2.0-auth-0000.tar", "CASIA2.0-auth-0005.tar",
              "CASIA2.0-manip-0000.tar", "CASIA2.0-manip-0005.tar"],
    "columbia": ["Columbia-auth-0000.tar", "Columbia-manip-0000.tar"],
    "coverage": ["Coverage-auth-0000.tar", "Coverage-manip-0000.tar"],
}
GENIMAGE = "TheKernel01/Tiny-GenImage"
GENIMAGE_SHARDS = {"cal": "data/train-00000-of-00014.parquet",
                   "test": "data/validation-00000-of-00004.parquet"}
GENERATORS = ["Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"]

FIELDS = ["path", "label", "source", "group", "orig_ext", "width", "height", "split"]


def fetch(repo, filename):
    return Path(hf_hub_download(repo, filename, repo_type="dataset", local_dir=DATA / "raw" / repo.split("/")[1]))


def prepare_edit():
    rows = []
    for source, shards in EDIT_SHARDS.items():
        for shard in shards:
            label = 1 if "manip" in shard else 0
            out_dir = DATA / "images" / "edit" / source / ("edited" if label else "authentic")
            out_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fetch(COMPILATION, shard)) as tar:
                for member in tar:
                    name = Path(member.name).name
                    if not name.endswith(".png") or name.endswith(".mask.png"):
                        continue
                    target = out_dir / name
                    data = tar.extractfile(member).read()
                    target.write_bytes(data)
                    with Image.open(io.BytesIO(data)) as im:
                        width, height = im.size
                    orig_ext = name[:-4].rsplit(".", 1)[-1].lower()
                    group = name.split("_")[1] if name.startswith("Tp_") else ""
                    rows.append({"path": target.relative_to(ROOT).as_posix(), "label": label,
                                 "source": source, "group": group, "orig_ext": orig_ext,
                                 "width": width, "height": height})
    rng = random.Random(SEED)
    for label in (0, 1):
        casia = sorted((r for r in rows if r["source"] == "casia" and r["label"] == label),
                       key=lambda r: r["path"])
        rng.shuffle(casia)
        cut = int(len(casia) * 0.6)
        for i, r in enumerate(casia):
            r["split"] = "cal" if i < cut else "test"
    for r in rows:
        r.setdefault("split", "external_test")
    write_manifest("edit", rows)


def prepare_ai():
    import pyarrow.parquet as pq
    rows = []
    for split, filename in GENIMAGE_SHARDS.items():
        table = pq.read_table(fetch(GENIMAGE, filename))
        for i, (img, label, gen) in enumerate(zip(table.column("image").to_pylist(),
                                                  table.column("label").to_pylist(),
                                                  table.column("generator").to_pylist())):
            ai = int(label == 1)
            group = GENERATORS[gen]
            with Image.open(io.BytesIO(img["bytes"])) as im:
                fmt, (width, height) = im.format.lower(), im.size
            ext = "jpg" if fmt == "jpeg" else fmt
            target = DATA / "images" / "ai" / split / f"{group}_{i:05d}.{ext}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(img["bytes"])
            rows.append({"path": target.relative_to(ROOT).as_posix(), "label": ai,
                         "source": "tiny-genimage", "group": group, "orig_ext": ext,
                         "width": width, "height": height, "split": split})
    write_manifest("ai", rows)


def write_manifest(task, rows):
    out = DATA / f"manifest_{task}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["path"]))
    print(f"{task}: {len(rows)} images -> {out}")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    prepare_edit()
    prepare_ai()
