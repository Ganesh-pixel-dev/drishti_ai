"""Run every detector (and, for the AI task, the two HF classifiers) over a manifest.

    python scripts/extract_scores.py edit
    python scripts/extract_scores.py ai
    python scripts/extract_scores.py ai --condition jpeg90

Condition "jpeg90" re-saves every image as a JPEG (quality 90) before the detectors
see it. The AI dataset stores real photos as JPEG and generated images as PNG, so the
detectors can separate them on file history alone; this control gives every image the
same treatment.

Output: data/scores_<task>_<condition>.csv (one row per image, one column per detector).
"""
import argparse
import io
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _score_one(args):
    rel_path, condition = args
    from image_module.pipeline import run_detectors, scores_only
    path = ROOT / rel_path
    tmp = None
    try:
        if condition == "jpeg90":
            with Image.open(path) as im:
                handle = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                im.convert("RGB").save(handle, "JPEG", quality=90)
                handle.close()
                tmp = Path(handle.name)
            path = tmp
        results = run_detectors(str(path))
        row = scores_only(results)
        row.update({f"na_{k}": int(not v.get("applicable", True)) for k, v in results.items()})
        row["errors"] = ";".join(k for k, v in results.items() if "error" in v)
        return rel_path, row
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["edit", "ai"])
    parser.add_argument("--condition", default="raw", choices=["raw", "jpeg90"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    manifest = pd.read_csv(ROOT / "data" / f"manifest_{args.task}.csv")
    jobs = [(p, args.condition) for p in manifest["path"]]
    with ProcessPoolExecutor(args.workers) as pool:
        done = dict(pool.map(_score_one, jobs, chunksize=8))
    scores = pd.DataFrame.from_dict(done, orient="index")
    out = manifest.merge(scores, left_on="path", right_index=True)

    if args.task == "ai":
        from image_module.classifiers import MODELS, prob_ai
        for key in MODELS:
            vals = []
            for p in manifest["path"]:
                with Image.open(ROOT / p) as im:
                    if args.condition == "jpeg90":
                        buf = io.BytesIO()
                        im.convert("RGB").save(buf, "JPEG", quality=90)
                        buf.seek(0)
                        im = Image.open(buf)
                    vals.append(prob_ai(key, im))
            out[f"clf_{key}"] = vals

    target = ROOT / "data" / f"scores_{args.task}_{args.condition}.csv"
    out.to_csv(target, index=False)
    print(f"wrote {target} ({len(out)} rows)")


if __name__ == "__main__":
    main()
