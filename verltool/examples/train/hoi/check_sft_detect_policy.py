"""Quick check of a detect-tool SFT model's first move (plan/saha-detect-tool/PLAN.md §C).

    cd /workspace/AdaTooler-V/verltool
    CUDA_VISIBLE_DEVICES=0 python examples/train/hoi/check_sft_detect_policy.py \
        --model /workspace/data/qwen3vl-8b-saha-detect-sft-merged --n 200

Generates the first assistant turn for held-out val_detect.parquet prompts and counts,
per task, how often the model calls detect, zooms, or answers directly, and whether
the turn is well-formed. Wanted after SFT: detect on most grounding prompts but not
all of them, rarely on referring, and well-formed tool-call JSON.
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict

import pandas as pd
from vllm import LLM, SamplingParams

TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.S)


def messages(prompt, image):
    system, user = prompt[0]["content"], prompt[1]["content"]
    before, after = user.split("<image>", 1)
    assert not before.strip()
    return [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file://" + image}},
                                         {"type": "text", "text": after.lstrip("\n")}]}]


def first_move(text):
    m = TOOL_CALL.search(text)
    if m:
        try:
            return "call:" + json.loads(m.group(1))["name"]
        except (json.JSONDecodeError, KeyError):
            return "bad_tool_json"
    if re.search(r"<think>.*?</think>.*<answer>.*?</answer>", text, re.S):
        return "answer"
    return "malformed"


def main(args):
    d = pd.read_parquet(args.parquet)
    rows = d.sample(n=min(args.n, len(d)), random_state=0)
    llm = LLM(model=args.model, max_model_len=16384, gpu_memory_utilization=0.6,
              limit_mm_per_prompt={"image": 1}, allowed_local_media_path="/",
              mm_processor_kwargs={"max_pixels": 2000000, "min_pixels": 200704})
    sp = SamplingParams(n=args.samples, temperature=1.0, max_tokens=2048, stop=["</tool_call>"], seed=0,
                        include_stop_str_in_output=True)
    outs = llm.chat([messages(p, ei["detect_image"]) for p, ei in zip(rows["prompt"], rows["extra_info"])], sp)
    counts = defaultdict(Counter)
    examples = {}
    for task, out in zip(rows["extra_info"].map(lambda e: e["task_type"]), outs):
        for o in out.outputs:
            move = first_move(o.text)
            counts[task][move] += 1
            examples.setdefault((task, move), o.text[:400])
    for task, c in counts.items():
        total = sum(c.values())
        print(task, total, {k: f"{v / total:.1%}" for k, v in c.most_common()})
    for (task, move), text in examples.items():
        print(f"--- {task} / {move}\n{text}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", default="data/val_detect.parquet")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--samples", type=int, default=4, help="samples per prompt at temperature 1 (GRPO-like)")
    main(ap.parse_args())
