"""No-detect grounding SFT traces by rejection sampling a model (PLAN §C2).

    CUDA_VISIBLE_DEVICES=0 python examples/data_preprocess/hoi/sft_gen/sample_nodetect.py --limit 64   # pilot
    CUDA_VISIBLE_DEVICES=0 python examples/data_preprocess/hoi/sft_gen/sample_nodetect.py

Each grounding prompt from convert_to_detect_tool.py's nodetect_requests.jsonl (the
detect-tool prompt, no proposals) is answered `--n` times. Generation stops at a tool
call, so only answers that use no tool remain. A sample is kept when it has the
<think>/<answer> format and scores >= --min_outcome with the RL grounding scorer
(saha_cf minAR10). The best sample per prompt becomes a trace paired with that prompt's
detect trace; at most --max_keep are written to nodetect.json.
"""
import argparse
import json
import os
import random
import re
from collections import Counter

from vllm import LLM, SamplingParams

from verl_tool.workers.reward_manager.saha_cf import compute_grounding_outcome_ar
from verl_tool.workers.reward_manager.sds_grpo import compute_format_reward

OUT = "/workspace/data/hoi_sft_detect"
# A no-detect trace must not talk about detecting, proposals or tool calls.
TOOL_TALK = re.compile(r"detect|proposal|candidate|tool|zoom", re.I)


NO_TOOL_GUIDELINE = ("Guidelines: Reason with the visual information step by step. Place your text reasoning "
                     "process within the <think> </think> tags, and provide your final answer within the "
                     "<answer> </answer> tags.")


def no_tool_prompt(user):
    """The detect-tool prompt with its tool guideline swapped for a tool-free one. With
    tools offered the base model called one on every sample (pilot 256/256), so no-tool
    answers are sampled here and then trained under the detect-tool prompt."""
    head, sep, _ = user.rpartition("Guidelines: Understand the given visual information")
    assert sep, "tool guideline not found"
    return head + NO_TOOL_GUIDELINE


def messages(req, prefill):
    before, after = no_tool_prompt(req["user"]).split("<image>", 1)
    assert not before.strip()
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file://" + req["image"]}},
                                         {"type": "text", "text": after.lstrip("\n")}]}]
    # The base model ignores the tag instructions (pilot: 0/64 formatted); --prefill
    # starts its reply with <think>.
    return msgs + [{"role": "assistant", "content": "<think>"}] if prefill else msgs


def main(args):
    reqs = [json.loads(line) for line in open(os.path.join(OUT, "nodetect_requests.jsonl"))]
    if args.limit:
        reqs = random.Random(0).sample(reqs, args.limit)
    llm = LLM(model=args.model, max_model_len=16384, gpu_memory_utilization=args.gpu_mem,
              limit_mm_per_prompt={"image": 1}, allowed_local_media_path="/workspace/data",
              mm_processor_kwargs={"max_pixels": 2000000, "min_pixels": 200704})
    sp = SamplingParams(n=args.n, temperature=args.temperature, max_tokens=2048, stop=["</tool_call>"], seed=0)
    outs = llm.chat([messages(r, args.prefill) for r in reqs], sp,
                    add_generation_prompt=not args.prefill, continue_final_message=args.prefill)
    stats, kept = Counter(), []
    for req, out in zip(reqs, outs):
        gt = json.loads(req["gt"])
        best = None
        for o in out.outputs:
            text = (("<think>" if args.prefill else "") + o.text).strip()
            stats["samples"] += 1
            if "<tool_call>" in text or o.stop_reason == "</tool_call>":
                stats["tool_call"] += 1
                continue
            if TOOL_TALK.search(text):
                stats["mentions_tool"] += 1
                continue
            if compute_format_reward(text, "grounding") < 1:
                stats["bad_format"] += 1
                continue
            score = compute_grounding_outcome_ar(text, gt)
            if score < args.min_outcome:
                stats["wrong"] += 1
                continue
            stats["pass"] += 1
            if best is None or score > best[0]:
                best = (score, text)
        if best:
            stats["prompts_kept"] += 1
            kept.append({"system": req["system"],
                         "conversations": [{"from": "human", "value": req["user"]},
                                           {"from": "gpt", "value": best[1]}],
                         "images": [req["sft_image"]], "_orig_image": req["orig_image"],
                         "_task": "grounding", "_source": "nodetect", "_score": best[0]})
    random.Random(0).shuffle(kept)
    kept = kept[: args.max_keep]
    print(json.dumps(stats), f"prompts={len(reqs)} written={len(kept)}")
    if not args.limit:
        json.dump(kept, open(os.path.join(OUT, "nodetect.json"), "w"), ensure_ascii=False)
    else:
        for k in kept[:2]:
            print("----", k["_score"], k["conversations"][1]["value"][:800])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # The SAHA SFT model: the base model formats <1/6 of samples even with --prefill
    # and talks about detecting in the rest (pilot, 64 prompts).
    ap.add_argument("--model", default="/workspace/data/qwen3vl-8b-saha-sft-merged")
    ap.add_argument("--prefill", action="store_true", help="prefill '<think>' (for the base model)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--min_outcome", type=float, default=0.5)
    ap.add_argument("--max_keep", type=int, default=1000, help="keeps no-detect at ~30% of grounding traces")
    ap.add_argument("--limit", type=int, default=0, help="pilot on N random prompts; writes nothing")
    ap.add_argument("--gpu_mem", type=float, default=0.6)
    main(ap.parse_args())
