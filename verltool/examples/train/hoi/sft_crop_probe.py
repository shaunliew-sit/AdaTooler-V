#!/usr/bin/env python
"""DECISIVE PROBE: can the reused Qwen3-VL-8B SFT checkpoint turn CORRECT (grid1000)
zoom crops into BETTER grounding on HARD (low proposal-anchor) samples than NOT
using tools?

If yes  -> a full RL rerun with the fixed crop is worthwhile.
If no   -> the SFT itself may have been cold-started on broken crops (regen needed).

Method
------
- Load data/val.parquet, keep grounding rows. Compute s_ref (the no-tool / trust-
  the-proposal baseline) with the SAME scorer the reward uses:
  saha_cf.compute_sref_grounding(proposal_anchor, gt).
- HARD subset = lowest s_ref (s_ref < HARD_THRESH). EASY control = s_ref >= EASY_THRESH.
- Run the SFT model through a multi-turn loop allowing zoom_in/zoom_out, executing
  crops in-process with the FIXED repo crop (pixel_reasoner.crop(..., coordinate_scale=1000),
  exactly what the training tool server does under PIXEL_REASONER_BBOX_MODE=grid1000).
  We REUSE the prompt + image straight out of the parquet so the model sees exactly
  what it saw in training (system prompt, proposals block, image).
- Score the final answer with sds_grpo.compute_grounding_outcome -> s_final.
- Compare on HARD samples: mean s_final vs mean s_ref, %(s_final>s_ref), mean gain
  among tool-using rollouts. Same on EASY.

Run (GPU 0 only):
    cd /workspace/AdaTooler-V/verltool
    source /opt/conda/etc/profile.d/conda.sh && conda activate verl-tool-env
    CUDA_VISIBLE_DEVICES=0 PIXEL_REASONER_BBOX_MODE=grid1000 \
        python examples/train/hoi/sft_crop_probe.py \
        --n-hard 40 --n-easy 15 --max-turns 6
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

# Repo scorers (REUSED, not reinvented).
from verl_tool.workers.reward_manager.sds_grpo import compute_grounding_outcome
from verl_tool.workers.reward_manager.saha_cf import compute_sref_grounding
# Repo crop tool (the FIXED grid1000 crop = exactly what the training tool server runs).
from verl_tool.servers.tools.pixel_reasoner import crop as repo_crop
# Same image post-processing the tool server applies after cropping (Qwen smart-resize),
# so the model sees byte-faithful observations.
from verl_tool.agent_loop.vision_utils import process_image

MODEL_PATH = "/workspace/hoi-tool-use-checkpoints/sft-checkpoints/qwen3VL-8B"
VAL_PARQUET = "data/val.parquet"


# ---------------------------------------------------------------------------
# Parsing helpers (mirror the reference harness + the reward tool-count regex)
# ---------------------------------------------------------------------------
def parse_tool_call(text: str):
    """Extract the LAST <tool_call>...</tool_call> JSON from model output."""
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None


def extract_answer(text: str):
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def gt_from_row(row) -> dict:
    raw = row["reward_model"]["ground_truth"]
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return raw if isinstance(raw, dict) else {}


def image_path_from_row(row):
    """val.parquet images column = list of {"image": "/abs/path.jpg"} (or raw str)."""
    imgs = row["images"]
    if imgs is None or len(imgs) == 0:
        return None
    first = imgs[0]
    if isinstance(first, dict):
        return first.get("image") or first.get("image_url") or first.get("path")
    return first


# ---------------------------------------------------------------------------
# Prompt reconstruction from the parquet chat prompt.
# The parquet `prompt` is a list of {role, content}. content may be a plain str
# (with <image> placeholders) — we rebuild a Qwen3-VL chat with the PIL image(s)
# inserted, then render with the processor's chat template.
# ---------------------------------------------------------------------------
def build_initial_messages(prompt_list, pil_image):
    """Convert the stored chat prompt into vLLM-style multimodal messages.

    The HOI prompt has a single <image> in the user turn. We map it to the PIL
    image. System turn (if any) is kept verbatim.
    """
    messages = []
    used_image = False
    for msg in prompt_list:
        role = msg["role"]
        content = msg.get("content", "")
        if role != "user":
            messages.append({"role": role, "content": content})
            continue
        # user turn: split around the single <image> placeholder
        text = content if isinstance(content, str) else json.dumps(content)
        parts = []
        if "<image>" in text and not used_image:
            before, after = text.split("<image>", 1)
            if before.strip():
                parts.append({"type": "text", "text": before})
            parts.append({"type": "image"})
            used_image = True
            if after.strip():
                parts.append({"type": "text", "text": after})
        else:
            parts.append({"type": "text", "text": text})
            if not used_image:
                # no explicit placeholder -> prepend image
                parts.insert(0, {"type": "image"})
                used_image = True
        messages.append({"role": role, "content": parts})
    return messages


# ---------------------------------------------------------------------------
# Multi-turn rollout using vLLM offline + in-process FIXED crop.
# ---------------------------------------------------------------------------
def run_agent_loop(llm, processor, sampling_params, prompt_list, image_path,
                   max_turns, max_zoom):
    """Returns dict: answer, n_turns, n_zoom_in, n_zoom_out, zoom_bboxes,
    crop_sizes, finished, transcript."""
    from vllm import SamplingParams  # noqa: F401  (type only)

    original = Image.open(image_path).convert("RGB")
    current_images = [original]            # list of PIL images shown so far
    messages = build_initial_messages(prompt_list, original)

    n_zoom_in = n_zoom_out = 0
    zoom_bboxes, crop_sizes = [], []
    transcript = []

    for turn in range(max_turns):
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        mm = {"image": current_images if len(current_images) > 1 else current_images[0]}
        out = llm.generate(
            [{"prompt": prompt_text, "multi_modal_data": mm}],
            sampling_params,
            use_tqdm=False,
        )
        text = out[0].outputs[0].text
        transcript.append({"turn": turn, "text": text})

        ans = extract_answer(text)
        if ans is not None:
            return dict(answer=ans, n_turns=turn + 1, n_zoom_in=n_zoom_in,
                        n_zoom_out=n_zoom_out, zoom_bboxes=zoom_bboxes,
                        crop_sizes=crop_sizes, finished=True, transcript=transcript)

        tool = parse_tool_call(text)
        if not tool:
            return dict(answer=None, n_turns=turn + 1, n_zoom_in=n_zoom_in,
                        n_zoom_out=n_zoom_out, zoom_bboxes=zoom_bboxes,
                        crop_sizes=crop_sizes, finished=False, transcript=transcript)

        name = tool.get("name", "")
        args = tool.get("arguments", {}) or {}
        messages.append({"role": "assistant", "content": text})

        if name in ("zoom_in", "crop_image", "crop_image_normalized"):
            bbox = args.get("bbox_2d", [0, 0, 1000, 1000])
            # FIXED crop: divide by 1000 (grid1000), the training-correct convention,
            # then the SAME process_image() the tool server runs before returning it.
            try:
                cropped = process_image({"image": repo_crop(original, bbox, coordinate_scale=1000.0)})
            except Exception as e:
                cropped = original
                transcript[-1]["crop_error"] = str(e)
            n_zoom_in += 1
            zoom_bboxes.append(bbox)
            crop_sizes.append(list(cropped.size))
            current_images = current_images + [cropped]
            cw, ch = cropped.size
            # Match the tool server's observation string EXACTLY (text then <image>),
            # so the SFT model recognizes the format it was trained on.
            messages.append({"role": "user", "content": [
                {"type": "text", "text": f"Here is the cropped image. (Image Size: {cw}x{ch})\n"},
                {"type": "image"}]})
        elif name == "zoom_out":
            n_zoom_out += 1
            # zoom_out returns TEXT only (no new image) — matches the tool server.
            messages.append({"role": "user", "content": "Returned to the full original image view."})
        else:
            messages.append({"role": "user", "content": "Unknown tool. Provide your "
                             "final answer in <answer>...</answer> tags."})

    # ran out of turns
    return dict(answer=None, n_turns=max_turns, n_zoom_in=n_zoom_in,
                n_zoom_out=n_zoom_out, zoom_bboxes=zoom_bboxes,
                crop_sizes=crop_sizes, finished=False, transcript=transcript)


def summarize(rows, label):
    """rows: list of dicts with s_ref, s_final, i_tool."""
    n = len(rows)
    if n == 0:
        return {"label": label, "n": 0}
    s_ref = np.array([r["s_ref"] for r in rows])
    s_fin = np.array([r["s_final"] for r in rows])
    i_tool = np.array([r["i_tool"] for r in rows])
    tool_mask = i_tool > 0
    better = s_fin > s_ref + 1e-9
    out = {
        "label": label,
        "n": int(n),
        "mean_s_ref": float(s_ref.mean()),
        "mean_s_final": float(s_fin.mean()),
        "frac_s_final_gt_s_ref": float(better.mean()),
        "mean_gain_all": float((s_fin - s_ref).mean()),
        "tool_rate": float(tool_mask.mean()),
        "n_tool": int(tool_mask.sum()),
        "mean_gain_tool_rollouts": (
            float((s_fin[tool_mask] - s_ref[tool_mask]).mean()) if tool_mask.any() else None),
        "mean_s_final_tool": float(s_fin[tool_mask].mean()) if tool_mask.any() else None,
        "mean_s_final_notool": float(s_fin[~tool_mask].mean()) if (~tool_mask).any() else None,
        "frac_finished_answer": float(np.mean([r["finished"] for r in rows])),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--val-parquet", default=VAL_PARQUET)
    ap.add_argument("--n-hard", type=int, default=40)
    ap.add_argument("--n-easy", type=int, default=15)
    ap.add_argument("--hard-thresh", type=float, default=0.5)
    ap.add_argument("--easy-thresh", type=float, default=0.99)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-zoom", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    # IMPORTANT: training rolls out at temperature=1.0. Under greedy (temp=0) this SFT
    # model emits a DEGENERATE empty <tool_call></tool_call>; at 1.0 it emits proper JSON.
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default="examples/train/hoi/sft_crop_probe_results.json")
    args = ap.parse_args()

    # Force the correct crop convention for any in-process tool object too.
    os.environ.setdefault("PIXEL_REASONER_BBOX_MODE", "grid1000")

    import datasets
    ds = datasets.load_dataset("parquet", data_files=args.val_parquet)["train"]

    # ---- select hard / easy grounding samples by s_ref ----
    scored = []
    for i, row in enumerate(ds):
        ei = row["extra_info"]
        if (ei.get("task_type") if hasattr(ei, "get") else None) != "grounding":
            continue
        gt = gt_from_row(row)
        if not gt.get("boxes_1000") or not gt.get("num_pairs"):
            continue
        img_path = image_path_from_row(row)
        if not img_path or not os.path.exists(img_path):
            continue
        s_ref = compute_sref_grounding(ei.get("proposal_anchor"), gt)
        scored.append({"idx": i, "s_ref": s_ref, "gt": gt, "img": img_path,
                       "prompt": row["prompt"]})
    scored.sort(key=lambda x: x["s_ref"])
    hard = [s for s in scored if s["s_ref"] < args.hard_thresh][:args.n_hard]
    easy = [s for s in reversed(scored) if s["s_ref"] >= args.easy_thresh][:args.n_easy]
    print(f"[select] total grounding={len(scored)} | hard(<{args.hard_thresh})={len(hard)} "
          f"| easy(>={args.easy_thresh})={len(easy)}", flush=True)
    if not hard:
        print("[FATAL] no hard samples found — check s_ref distribution.", flush=True)
        sys.exit(2)

    # ---- load model under vLLM (mirror training: trust_remote_code, eager) ----
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": args.max_zoom + 3},
        enforce_eager=True,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=1.0, max_tokens=2048,
        stop=["</tool_call>", "</answer>"], include_stop_str_in_output=True,
    )

    def run_subset(subset, tag):
        rows = []
        for k, s in enumerate(subset):
            res = run_agent_loop(llm, processor, sampling_params, s["prompt"], s["img"],
                                 args.max_turns, args.max_zoom)
            s_final = compute_grounding_outcome(
                f"<answer>{res['answer']}</answer>" if res["answer"] else "", s["gt"])
            i_tool = 1 if (res["n_zoom_in"] + res["n_zoom_out"]) > 0 else 0
            row = {
                "idx": s["idx"], "tag": tag, "s_ref": s["s_ref"], "s_final": s_final,
                "i_tool": i_tool, "n_zoom_in": res["n_zoom_in"],
                "n_zoom_out": res["n_zoom_out"], "n_turns": res["n_turns"],
                "finished": res["finished"], "zoom_bboxes": res["zoom_bboxes"],
                "crop_sizes": res["crop_sizes"], "answer": res["answer"],
                "img": s["img"], "transcript": res["transcript"],
            }
            rows.append(row)
            print(f"[{tag} {k+1}/{len(subset)}] idx={s['idx']} s_ref={s['s_ref']:.3f} "
                  f"s_final={s_final:.3f} tool={i_tool} zin={res['n_zoom_in']} "
                  f"zout={res['n_zoom_out']} fin={res['finished']} "
                  f"crops={res['crop_sizes']}", flush=True)
        return rows

    hard_rows = run_subset(hard, "hard")
    easy_rows = run_subset(easy, "easy")

    hard_summary = summarize(hard_rows, "hard")
    easy_summary = summarize(easy_rows, "easy")
    # tool-only sub-view on hard
    hard_tool = [r for r in hard_rows if r["i_tool"]]
    hard_notool = [r for r in hard_rows if not r["i_tool"]]

    report = {
        "config": vars(args),
        "n_grounding_total": len(scored),
        "hard_summary": hard_summary,
        "easy_summary": easy_summary,
        "hard_tool_n": len(hard_tool),
        "hard_notool_n": len(hard_notool),
        "rows": hard_rows + easy_rows,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("HARD:", json.dumps(hard_summary, indent=2))
    print("EASY:", json.dumps(easy_summary, indent=2))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
