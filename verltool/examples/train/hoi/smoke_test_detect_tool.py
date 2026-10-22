"""Smoke test G1 for the `detect` tool (plan/saha-detect-tool/PLAN.md §G1).

Sends each test image to PixelReasonerTool the way the RL agent loop does (the
resized JPEG data URL from verl process_image -> encode_image_url, plus the original
file as `detect_image`), calls `detect`, and compares the result with the offline
YOLOE proposals in hoi-tool-use-checkpoints/test_proposals, which the SFT traces
were built from. `--no_detect_image` drops the path to measure the fallback (the
JPEG copy), which does NOT reproduce the offline proposals.

    cd /workspace/AdaTooler-V/verltool
    CUDA_VISIBLE_DEVICES=0 DETECT_TOOL_DEVICE=0 python examples/train/hoi/smoke_test_detect_tool.py --n 500
"""
import argparse
import asyncio
import glob
import json
import random
import re

from verl.utils.dataset.vision_utils import process_image
from verl_tool.agent_loop.vision_utils import encode_image_url
from verl_tool.servers.tools.pixel_reasoner import PixelReasonerTool

PROPOSALS = "/workspace/hoi-tool-use-checkpoints/test_proposals"
IMAGE_ROOTS = {"/data/hico_20160224_det": "/workspace/data/hico_20160224_det",
               "/data/swig_hoi": "/workspace/data/swig_hoi"}


def local_path(p):
    for old, new in IMAGE_ROOTS.items():
        if p.startswith(old):
            return new + p[len(old):]
    return p


def match(live, offline, tol):
    """Greedy one-to-one match on (label, box within tol). Returns #matched."""
    unused = list(offline)
    n = 0
    for lp in live:
        for op in unused:
            if lp["label"] == op["label"] and max(abs(a - b) for a, b in zip(lp["bbox_2d"], op["bbox_2d"])) <= tol:
                unused.remove(op)
                n += 1
                break
    return n


async def main(args):
    files = sorted(glob.glob(f"{PROPOSALS}/*.json"))
    hico = [f for f in files if "/HICO_" in f]
    swig = [f for f in files if "/HICO_" not in f]
    rng = random.Random(0)
    sample = rng.sample(hico, args.n // 2) + rng.sample(swig, args.n - args.n // 2)

    tool = PixelReasonerTool()
    stats = {"images": 0, "exact_set": 0, "same_order": 0, "same_count": 0,
             "live": 0, "offline": 0, "matched": 0, "errors": 0}
    action = '<tool_call>\n{"name": "detect", "arguments": {}}\n</tool_call>'
    for i, f in enumerate(sample):
        data = json.load(open(f))
        offline = [{"bbox_2d": p["bbox_1000"], "label": p["class_name"]} for p in data["proposals"]]
        img = process_image({"image": local_path(data["image_path"])}, image_patch_size=args.patch)
        traj = f"g1-{i}"
        extra = {"images": [encode_image_url(img)]}
        if not args.no_detect_image:
            extra["detect_image"] = local_path(data["image_path"])
        obs, _, valid = await tool._conduct_action_async(traj, action, extra)
        tool.delete_env(traj)
        m = re.search(r"<candidate_objects>\n(.*)\n</candidate_objects>", obs, re.S) if valid else None
        if m is None:
            stats["errors"] += 1
            print("ERROR", f, str(obs)[:200])
            continue
        live = json.loads(m.group(1))
        stats["images"] += 1
        stats["live"] += len(live)
        stats["offline"] += len(offline)
        stats["same_count"] += len(live) == len(offline)
        k = match(live, offline, args.tol)
        stats["matched"] += k
        stats["exact_set"] += k == len(live) == len(offline)
        stats["same_order"] += [(p["label"], p["bbox_2d"]) for p in live] == [(p["label"], p["bbox_2d"]) for p in offline]
    n = max(stats["images"] + stats["errors"], 1)
    recall = stats["matched"] / max(stats["offline"], 1)
    precision = stats["matched"] / max(stats["live"], 1)
    print(json.dumps(stats))
    print(f"set match (tol {args.tol}/1000): {stats['exact_set'] / n:.1%} of images | "
          f"identical incl. order: {stats['same_order'] / n:.1%} | "
          f"detection recall vs offline: {recall:.1%}, precision: {precision:.1%}")
    ok = stats["errors"] == 0 and min(recall, precision) >= args.min_match
    print("PASS" if ok else f"FAIL: errors={stats['errors']} or recall/precision < {args.min_match:.0%}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--min_match", type=float, default=0.95, help="required recall and precision")
    ap.add_argument("--no_detect_image", action="store_true", help="only send the agent loop's JPEG copy")
    ap.add_argument("--patch", type=int, default=14, help="data.image_patch_size used by the RL dataset")
    asyncio.run(main(ap.parse_args()))
