"""Merge the kept traces from one or more generation output dirs into a final SFT dataset.

Dedups by the verbatim human prompt (unique per parquet row). Copies each source dir's
images/<key>/ into the merged images/ (keys are idx_<uid8>, collision-safe across runs).
Writes merged dataset.json + dataset_info.json (hoi_sft_regen, v1 tags + system).

Usage:
  python examples/data_preprocess/hoi/sft_gen/merge_sft.py \
    --out /workspace/data/hoi_sft_final \
    --src /workspace/data/hoi_sft_regen_v1based /workspace/data/hoi_sft_regen_full
"""
import argparse
import json
import re
import shutil
from pathlib import Path


def _clean_dup_toolcall(conv):
    """Collapse duplicate <tool_call> blocks in a gpt turn (Responses returned the call both as
    visible text and as a function_call). Keep the <think> + the LAST (harness-validated) call."""
    for t in conv:
        if t["from"] != "gpt":
            continue
        v = t["value"]
        tcs = re.findall(r"<tool_call>.*?</tool_call>", v, re.S)
        if len(tcs) > 1:
            think = v[:v.index("<tool_call>")].rstrip()
            t["value"] = (think + "\n\n" + tcs[-1]).strip()
    return conv

DATASET_INFO = {
    "hoi_sft_regen": {
        "file_name": "dataset.json", "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images", "system": "system"},
        "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human",
                 "assistant_tag": "gpt", "observation_tag": "observation"},
    }
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", nargs="+", required=True, help="generation dirs (priority order; first wins on dup)")
    args = ap.parse_args()
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)

    seen, merged, n_tool = set(), [], 0
    for src in args.src:
        src = Path(src)
        jsonl = src / "dataset.jsonl"
        if not jsonl.exists():
            print(f"  skip {src} (no dataset.jsonl)"); continue
        kept_here = 0
        for line in open(jsonl):
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            human = s["conversations"][0]["value"]
            if human in seen:
                continue
            seen.add(human)
            s["conversations"] = _clean_dup_toolcall(s["conversations"])  # fix duplicate <tool_call>
            # copy this sample's image dirs into merged/images/, rewrite paths
            new_imgs = []
            for rel in s["images"]:
                p = Path(rel)  # images/<key>/N.jpg
                key = p.parent.name
                dst_dir = out / "images" / key
                if not dst_dir.exists() and (src / p.parent).exists():
                    shutil.copytree(src / p.parent, dst_dir)
                new_imgs.append(str(Path("images") / key / p.name))
            s["images"] = new_imgs
            merged.append(s)
            kept_here += 1
            if any(t["from"] == "observation" for t in s["conversations"]):
                n_tool += 1
        print(f"  {src.name}: +{kept_here} unique")
    json.dump(merged, open(out / "dataset.json", "w"), ensure_ascii=False, indent=1)
    json.dump(DATASET_INFO, open(out / "dataset_info.json", "w"), indent=2)
    print(f"\nMERGED: {len(merged)} samples ({n_tool} tool / {len(merged)-n_tool} no-tool) -> {out}/dataset.json")
    print(f"dataset_info.json written. Point LLaMA-Factory image_dir at {out}/")


if __name__ == "__main__":
    main()
