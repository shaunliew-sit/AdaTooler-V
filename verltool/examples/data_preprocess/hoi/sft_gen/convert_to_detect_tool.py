"""Rewrite the proposal-injected SFT traces for the detect tool (PLAN §C). No teacher calls.

    # 1) rewrite traces + write the no-detect sampling requests
    python examples/data_preprocess/hoi/sft_gen/convert_to_detect_tool.py rewrite
    # 2) sample_nodetect.py fills OUT/nodetect.json (base model, no tools)
    # 3) merge, split by original image, audit, register for LlamaFactory
    python examples/data_preprocess/hoi/sft_gen/convert_to_detect_tool.py assemble

Rewrite rules, per trace of SRC/dataset.json:
- system and first human turn go through detect_tool_prompt (proposals removed).
- grounding traces, and referring traces whose reasoning cites the proposals, get a
  `detect` call + observation carrying the exact YOLOE JSON that was in the prompt,
  inserted before the original reasoning. Other referring traces stay tool-free.
- traces with two <answer> blocks in one turn are dropped.
GT for scoring comes from data/train.parquet, matched on the original user prompt.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect_tool_prompt import (  # noqa: E402
    DETECT_CALL, convert_system_prompt, convert_user_prompt, detect_observation,
)

SRC = "/workspace/data/hoi_sft_final"
OUT = "/workspace/data/hoi_sft_detect"
TRAIN_PARQUET = os.path.join(os.path.dirname(__file__), "../../../../data/train.parquet")
DETECT_THINK = {
    "grounding": "<think>I need to find the people and objects involved in this interaction. "
                 "I will run the object detector to get candidate boxes for the image.</think>",
    "referring": "<think>The person and object boxes are given. To see the other objects in the "
                 "scene as context, I will run the object detector.</think>",
}
CITES_PROPOSALS = re.compile(r"candidate|proposal|\bid \d", re.I)
VAL_FRAC = 0.1


def is_val(image_path: str) -> bool:
    return int(hashlib.md5(image_path.encode()).hexdigest(), 16) % 1000 < VAL_FRAC * 1000


def gt_by_prompt(prompts: set) -> dict:
    d = pd.read_parquet(TRAIN_PARQUET, columns=["prompt", "reward_model", "extra_info", "images"])
    out = {}
    for p, rm, ei, im in zip(d["prompt"], d["reward_model"], d["extra_info"], d["images"]):
        u = p[1]["content"]
        if u in prompts and u not in out:
            out[u] = {"gt": rm["ground_truth"], "task_type": ei["task_type"],
                      "orig_image": os.path.realpath(im[0]["image"])}
    missing = prompts - out.keys()
    if missing:
        raise ValueError(f"{len(missing)} SFT prompts not found in {TRAIN_PARQUET}")
    return out


def rewrite(args):
    src = json.load(open(os.path.join(SRC, "dataset.json")))
    meta = gt_by_prompt({x["conversations"][0]["value"] for x in src})
    rewritten, requests, stats = [], [], Counter()
    for i, x in enumerate(src):
        conv = x["conversations"]
        if any(m["from"] == "gpt" and m["value"].count("<answer>") > 1 for m in conv):
            stats["dropped_double_answer"] += 1
            continue
        m = meta[conv[0]["value"]]
        task = m["task_type"]
        user, proposals = convert_user_prompt(conv[0]["value"])
        new_conv = [{"from": "human", "value": user}]
        gpt_text = " ".join(t["value"] for t in conv if t["from"] == "gpt")
        if task == "grounding" or CITES_PROPOSALS.search(gpt_text):
            new_conv += [{"from": "gpt", "value": DETECT_THINK[task] + "\n\n" + DETECT_CALL},
                         {"from": "observation", "value": detect_observation(proposals)}]
            stats[f"{task}_detect"] += 1
        else:
            stats[f"{task}_no_tool"] += 1
        new_conv += conv[1:]
        system = convert_system_prompt(x["system"])
        rewritten.append({"system": system, "conversations": new_conv, "images": x["images"],
                          "_orig_image": m["orig_image"], "_task": task, "_source": "rewrite"})
        if task == "grounding":
            requests.append({"id": i, "system": system, "user": user, "gt": m["gt"],
                             "image": os.path.join(SRC, x["images"][0]), "sft_image": x["images"][0],
                             "orig_image": m["orig_image"]})
    os.makedirs(OUT, exist_ok=True)
    json.dump(rewritten, open(os.path.join(OUT, "rewritten.json"), "w"), ensure_ascii=False)
    with open(os.path.join(OUT, "nodetect_requests.jsonl"), "w") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(stats), f"-> {len(rewritten)} traces, {len(requests)} no-detect requests in {OUT}")


def audit(traces):
    """Hard checks before anything trains on the data."""
    bad = Counter()
    for t in traces:
        conv = t["conversations"]
        text = t["system"] + " ".join(m["value"] for m in conv)
        if "<candidate_objects>" in conv[0]["value"] or "<proposals>" in conv[0]["value"]:
            bad["proposals_in_prompt"] += 1
        if '"name": "detect"' not in t["system"]:
            bad["no_detect_schema"] += 1
        for m in conv:
            if m["from"] == "gpt":
                for call in re.findall(r"<tool_call>\n(.*?)\n</tool_call>", m["value"], re.S):
                    try:
                        json.loads(call)
                    except json.JSONDecodeError:
                        bad["bad_tool_json"] += 1
        if conv[-1]["from"] != "gpt" or "<answer>" not in conv[-1]["value"]:
            bad["no_final_answer"] += 1
        n_img_tags = text.count("<image>")
        if n_img_tags != len(t["images"]):
            bad["image_count_mismatch"] += 1
        roles = [m["from"] for m in conv]
        if roles[0] != "human" or any(a == b for a, b in zip(roles, roles[1:])):
            bad["bad_role_order"] += 1
    return bad


def assemble(args):
    rewritten = json.load(open(os.path.join(OUT, "rewritten.json")))
    nodetect = json.load(open(os.path.join(OUT, "nodetect.json")))
    traces = rewritten + nodetect
    bad = audit(traces)
    if bad:
        raise SystemExit(f"audit failed: {dict(bad)}")
    splits = {"train": [], "val": []}
    stats = Counter()
    for t in traces:
        split = "val" if is_val(t["_orig_image"]) else "train"
        uses_detect = any('"name": "detect"' in m["value"] for m in t["conversations"] if m["from"] == "gpt")
        stats[(split, t["_task"], "detect" if uses_detect else "no_detect")] += 1
        splits[split].append({k: v for k, v in t.items() if not k.startswith("_")})
    train_imgs = {t["_orig_image"] for t in traces if not is_val(t["_orig_image"])}
    val_imgs = {t["_orig_image"] for t in traces if is_val(t["_orig_image"])}
    assert not train_imgs & val_imgs
    for name, rows in splits.items():
        json.dump(rows, open(os.path.join(OUT, f"{name}.json"), "w"), ensure_ascii=False)
    images = os.path.join(OUT, "images")
    if not os.path.exists(images):
        os.symlink(os.path.join(SRC, "images"), images)
    entry = {"formatting": "sharegpt",
             "columns": {"messages": "conversations", "images": "images", "system": "system"},
             "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human",
                      "assistant_tag": "gpt", "observation_tag": "observation"}}
    json.dump({"hoi_sft_detect_train": {"file_name": "train.json", **entry},
               "hoi_sft_detect_val": {"file_name": "val.json", **entry}},
              open(os.path.join(OUT, "dataset_info.json"), "w"), indent=2)
    for k in sorted(stats):
        print(k, stats[k])
    print(f"train {len(splits['train'])} val {len(splits['val'])} -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rewrite", "assemble"])
    a = ap.parse_args()
    {"rewrite": rewrite, "assemble": assemble}[a.stage](a)
