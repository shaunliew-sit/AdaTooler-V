"""SFT trace generator (plan/saha-sft-regen/PLAN.md).

Seeds from verltool/data/train.parquet (verbatim RL prompt + image + GT + proposals),
drives gpt-5.5 via the Responses API tool loop (reasoning=high, previous_response_id
chaining), renders crops with the fixed grid1000 pixel_reasoner crop, applies curation
gates, and writes a ShareGPT dataset aligned to the CURRENT RL format.

Run a tiny smoke test first:
    python examples/data_preprocess/hoi/sft_gen/generate.py --smoke 3
Pilot (sync, concurrent):
    python examples/data_preprocess/hoi/sft_gen/generate.py --n 100 --task both --hard_frac 0.6
Output: /workspace/data/hoi_sft_regen_v2/{dataset.json, images/, stats.json}

NOTE: spends real OpenAI credits. Pilot first, inspect stats.json, then scale (Batch) for 10k.
"""
import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curate  # noqa: E402
from crop import load_image, pil_to_data_url, render_zoom  # noqa: E402
from prompts import (  # noqa: E402
    GEN_DEVELOPER_PROMPT, OBS_ZOOM_IN, OBS_ZOOM_OUT, RESPONSES_TOOLS, split_rl_prompt,
)

ROOT = Path("/workspace/AdaTooler-V")
ENV = dotenv_values(str(ROOT / ".env"))
MODEL = ENV.get("OPENAI_MODEL") or "gpt-5.5"
MAX_TOOL_TURNS = 5  # == RL agent max_turns (train_qwen3vl_v2.sh); a trace must not exceed it
_client = OpenAI(api_key=ENV["OPENAI_API_KEY"], base_url=(ENV.get("OPENAI_BASE_URL") or None))
_usage_lock = threading.Lock()
_USAGE = {"input": 0, "output": 0, "reasoning": 0, "calls": 0}


def _denumpy(o):
    if isinstance(o, np.ndarray):
        return [_denumpy(x) for x in o.tolist()]
    if isinstance(o, (list, tuple)):
        return [_denumpy(x) for x in o]
    if isinstance(o, dict):
        return {k: _denumpy(v) for k, v in o.items()}
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def _extract_proposals(user_text):
    # capture the FULL tag body (the array has nested [..] so a non-greedy [.*?] is wrong)
    m = re.search(r"<(candidate_objects|proposals)>(.*?)</\1>", user_text, re.S)
    if not m:
        return []
    try:
        return [o["bbox_2d"] for o in json.loads(m.group(2).strip()) if "bbox_2d" in o]
    except Exception:
        return []


def _valid_bbox(b):
    if not (isinstance(b, list) and len(b) == 4):
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in b)
    except Exception:
        return False
    import math
    if any(math.isnan(v) or math.isinf(v) for v in (x1, y1, x2, y2)):
        return False
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def _think_and_answer(resp):
    """Return (visible_text, reasoning_summary). visible_text holds <think>/<answer>/etc."""
    vis = resp.output_text or ""
    summ = []
    for it in resp.output:
        if it.type == "reasoning":
            for s in (getattr(it, "summary", None) or []):
                t = getattr(s, "text", None)
                if t:
                    summ.append(t)
    return vis, "\n".join(summ)


def _ensure_think(visible, reasoning_summary):
    """Guarantee a <think> block exists (the final-turn-missing-<think> fix)."""
    if re.search(r"<think>.*?</think>", visible, re.S):
        return visible
    think = reasoning_summary.strip() or "Analyzing the image and proposals to decide the answer."
    return f"<think>{think}</think>\n{visible}".strip()


def _acc_usage(resp):
    u = resp.usage
    rt = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0
    with _usage_lock:
        _USAGE["input"] += u.input_tokens
        _USAGE["output"] += u.output_tokens
        _USAGE["reasoning"] += rt
        _USAGE["calls"] += 1


def process_row(row, out_images_dir, idx):
    """Run the full tool loop for one parquet row. Returns a result dict or None."""
    ei = _denumpy(row["extra_info"]) if isinstance(row["extra_info"], dict) else json.loads(row["extra_info"])
    task_type = ei.get("task_type")
    gt_raw = row["reward_model"]["ground_truth"]
    gt_data = json.loads(gt_raw) if isinstance(gt_raw, str) else _denumpy(gt_raw)
    sys_text, user_text = split_rl_prompt(_denumpy(row["prompt"]))
    img_path = _denumpy(row["images"])[0]
    img_path = img_path["image"] if isinstance(img_path, dict) else img_path
    proposals = _extract_proposals(user_text)
    anchor = _denumpy(ei.get("proposal_anchor")) or []
    gt_boxes = gt_data.get("boxes_1000") or []
    # dir keyed by idx too -> no race/overwrite if a uid repeats across rows
    uid = ei.get("uid") or row.get("uid") or os.urandom(6).hex()
    sample_key = f"{idx:06d}_{str(uid)[:8]}"

    original = load_image(img_path)
    images = [original]                # index 0 = original (target_image=1)
    user_clean = re.sub(r"^<image>\s*", "", user_text)
    api_input = [
        {"role": "developer", "content": GEN_DEVELOPER_PROMPT},
        {"role": "system", "content": sys_text or "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "input_text", "text": user_clean},
            {"type": "input_image", "image_url": pil_to_data_url(original)}]},
    ]
    prev_id = None
    gpt_turns, sft_conv = [], []  # sft_conv built as we go (human added at the end)
    saved_crops = []
    n_tool_turns = 0  # ALL tool turns (zoom_in + zoom_out) — matches RL i_tool = any tool call
    zoomed_in = False  # a zoom_out is only valid while zoomed in (no zoom_out-only / pre-zoom_in)
    zoom_log = []  # [{zoom_in:[bbox], target_image:n} | {zoom_out:True}] — for --save_all diagnostics

    for rnd in range(MAX_TOOL_TURNS + 1):  # rnd 0..MAX_TOOL_TURNS; tools allowed on 0..MAX_TOOL_TURNS-1
        kwargs = dict(model=MODEL, reasoning={"effort": "high", "summary": "detailed"},
                      tools=RESPONSES_TOOLS, store=True, input=api_input)
        if prev_id:
            kwargs["previous_response_id"] = prev_id
        try:
            resp = _client.responses.create(**kwargs)
        except Exception as e:
            return {"uid": uid, "kept": False, "reason": f"api_error:{type(e).__name__}:{e}"}
        _acc_usage(resp)
        prev_id = resp.id
        visible, summ = _think_and_answer(resp)
        fcs = [it for it in resp.output if it.type == "function_call"]

        if fcs:  # tool turn
            if rnd >= MAX_TOOL_TURNS:  # RL would NOT execute a tool here (is_last_step) -> unreproducible
                return {"uid": uid, "kept": False, "reason": "exceeds_max_turns"}
            if len(fcs) > 1:  # SFT format is exactly one tool call/turn; multi-call breaks the chain
                return {"uid": uid, "kept": False, "reason": "multiple_tool_calls"}
            fc = fcs[0]
            try:
                args = json.loads(fc.arguments)
            except Exception:
                return {"uid": uid, "kept": False, "reason": "bad_tool_json"}
            is_zoom_in = fc.name in ("zoom_in", "crop_image_normalized", "crop_image")
            is_zoom_out = fc.name == "zoom_out"
            if not (is_zoom_in or is_zoom_out):
                return {"uid": uid, "kept": False, "reason": f"unknown_tool:{fc.name}"}
            # ---- validate the call BEFORE recording the turn (so saved tool_call is well-formed) ----
            crop_img = None
            if is_zoom_in:
                bbox = args.get("bbox_2d")
                # zoom_in REQUIRES target_image (RL parser rejects it otherwise; old SFT dropped it
                # = the drift we fix). Enforced for zoom_in only; zoom_out legitimately has empty args.
                if "target_image" not in args:
                    return {"uid": uid, "kept": False, "reason": "missing_target_image"}
                ti_raw = args["target_image"]
                if not _valid_bbox(bbox) or float(ti_raw) != int(ti_raw):
                    return {"uid": uid, "kept": False, "reason": f"bad_bbox/target:{args}"}
                ti = int(ti_raw)
                if ti < 1 or ti > len(images):
                    return {"uid": uid, "kept": False, "reason": f"target_oob:{ti}/{len(images)}"}
                try:
                    crop_img = render_zoom(images[ti - 1], bbox)
                except Exception as e:
                    return {"uid": uid, "kept": False, "reason": f"crop_error:{type(e).__name__}"}
            else:  # zoom_out is only valid while zoomed in (no zoom_out-only / pre-zoom_in traces)
                if not zoomed_in:
                    return {"uid": uid, "kept": False, "reason": "zoom_out_without_zoom_in"}
            # ---- record the (now-valid) tool turn ----
            n_tool_turns += 1
            # keep ONLY the <think> from visible text; drop any visible <tool_call>/<answer> the
            # model also wrote (Responses returns the call BOTH as text AND as a function_call ->
            # would duplicate). We then append the single harness-validated tool_call below.
            think_only = re.split(r"<tool_call>|<answer>", _ensure_think(visible, summ))[0].strip()
            tool_json = json.dumps({"name": fc.name, "arguments": args})
            gpt_val = f"{think_only}\n\n<tool_call>\n{tool_json}\n</tool_call>"
            gpt_turns.append(gpt_val)
            sft_conv.append({"from": "gpt", "value": gpt_val})
            if is_zoom_in:
                images.append(crop_img)
                saved_crops.append(crop_img)
                zoomed_in = True
                zoom_log.append({"zoom_in": list(bbox), "target_image": ti})
                sft_conv.append({"from": "observation", "value": OBS_ZOOM_IN})
                api_input = [
                    {"type": "function_call_output", "call_id": fc.call_id,
                     "output": "Here is the zoomed-in view of the requested region."},
                    {"role": "user", "content": [
                        {"type": "input_text", "text": OBS_ZOOM_IN},
                        {"type": "input_image", "image_url": pil_to_data_url(crop_img), "detail": "low"}]}]
            else:  # zoom_out
                zoomed_in = False
                zoom_log.append({"zoom_out": True})
                sft_conv.append({"from": "observation", "value": OBS_ZOOM_OUT})
                api_input = [{"type": "function_call_output", "call_id": fc.call_id, "output": OBS_ZOOM_OUT}]
            continue

        # final answer turn
        final_val = _ensure_think(visible, summ)
        gpt_turns.append(final_val)
        sft_conv.append({"from": "gpt", "value": final_val})
        # ---- curation ----
        s_ref = curate.grounding_sref(anchor, gt_data) if task_type == "grounding" else None
        n_tool = n_tool_turns  # ALL tool turns (RL i_tool semantics), not just zoom_in crops

        def _drop(reason, s_final=None):  # captures full trace for --save_all diagnostics
            return {"uid": uid, "kept": False, "reason": reason, "task_type": task_type,
                    "s_final": s_final, "s_ref": s_ref, "n_tool": n_tool,
                    "trace": {"zoom_log": zoom_log, "turns": sft_conv}}

        ok, why = curate.gate_format(gpt_turns, final_val, task_type)
        if not ok:
            return _drop(f"format:{why}")
        s_final = curate.score_outcome(task_type, final_val, gt_data)
        if s_final < curate.OUTCOME_KEEP:
            return _drop(f"low_outcome:{s_final:.2f}", s_final)
        if task_type == "grounding":
            ok, why = curate.gate_anti_override(final_val, proposals, gt_boxes)
            if not ok:
                return _drop(f"anti_override:{why}", s_final)
            # empty/missing anchor -> grounding_sref silently returns 0.0, which would let a
            # zoom-then-copy tool trace pass gate_tool_gain. A POPULATED anchor that genuinely
            # scores 0.0 (proposal misses GT) is the real hard case and is kept; reject only
            # TOOL traces whose anchor is empty/missing.
            if n_tool and not anchor:
                return _drop("missing_anchor_for_tool_gain", s_final)
            # a tool trace must beat the anchor — reject zoom-then-copy (proposal-copy failure)
            ok, why = curate.gate_tool_gain(task_type, n_tool, s_final, s_ref)
            if not ok:
                return _drop(why, s_final)
        ok, why = curate.gate_no_gt_leak(
            [{"role": "developer", "content": GEN_DEVELOPER_PROMPT},
             {"role": "system", "content": sys_text}, {"role": "user", "content": user_text}], gt_data)
        if not ok:
            return _drop(why, s_final)
        # ---- assemble + persist crops ----
        rel_imgs = []
        sample_dir = out_images_dir / sample_key
        sample_dir.mkdir(parents=True, exist_ok=True)
        original.save(sample_dir / "0.jpg", quality=92)
        rel_imgs.append(str((sample_dir / "0.jpg").relative_to(out_images_dir.parent)))
        for i, c in enumerate(saved_crops, start=1):
            c.convert("RGB").save(sample_dir / f"{i}.jpg", quality=92)
            rel_imgs.append(str((sample_dir / f"{i}.jpg").relative_to(out_images_dir.parent)))
        sample = {"system": sys_text, "conversations": [{"from": "human", "value": user_text}] + sft_conv,
                  "images": rel_imgs}
        assert len(rel_imgs) == 1 + len(saved_crops)
        return {"uid": uid, "kept": True, "task_type": task_type, "s_final": s_final,
                "s_ref": s_ref, "n_tool": n_tool, "sample": sample,
                "trace": {"zoom_log": zoom_log, "turns": sft_conv}}

    return {"uid": uid, "kept": False, "reason": "max_rounds_no_answer"}


def _row_sref(row):
    ei = _denumpy(row["extra_info"]) if isinstance(row["extra_info"], dict) else json.loads(row["extra_info"])
    gt_raw = row["reward_model"]["ground_truth"]
    gt = json.loads(gt_raw) if isinstance(gt_raw, str) else _denumpy(gt_raw)
    return curate.grounding_sref(_denumpy(ei.get("proposal_anchor")) or [], gt)


def _pick_grounding_stratified(df, idxs, n, hard_frac, rng):
    """Sample n grounding rows with ~hard_frac having s_ref<=0.5 (the tool-upside band).
    s_ref computed on a bounded candidate pool to cap cost."""
    pool = rng.choice(idxs, min(len(idxs), 8000, max(2000, 4 * n)), replace=False)  # capped: fast startup
    srefs = np.array([_row_sref(df.loc[i]) for i in pool])
    hard = pool[srefs <= 0.5 + 1e-6]
    easy = pool[srefs > 0.5 + 1e-6]
    n_hard = min(len(hard), round(hard_frac * n))
    n_easy = min(len(easy), n - n_hard)
    n_hard = min(len(hard), n - n_easy)  # backfill if easy short
    chosen = list(rng.choice(hard, n_hard, replace=False)) + list(rng.choice(easy, n_easy, replace=False))
    return chosen


def _load_v1_keys(path):
    """Set of (image_id, task_type) the v1 SFT dataset vetted as solvable (Claude produced a
    correct, rejection-filtered trace). Used to restrict generation to good-quality samples."""
    v1 = json.load(open(path))
    keys = set()
    for s in v1:
        img = s["images"][0]
        image_id = img.split("/")[1]  # images/<image_id>/<action_object>/0.jpg
        human = s["conversations"][0]["value"]
        ttype = "grounding" if "visual grounding task" in human else "referring"
        keys.add((image_id, ttype))
    return keys


def _img_id_fast(im):
    """image_id (basename w/o ext) from the raw 'images' cell (ndarray/list of {image:path}|path)."""
    first = im[0] if (isinstance(im, (list, np.ndarray)) and len(im)) else im
    p = first.get("image") if isinstance(first, dict) else first
    if isinstance(p, dict):
        p = p.get("image")
    return os.path.splitext(os.path.basename(str(p)))[0]


def select_rows(df, n, task, hard_frac, seed, from_v1=None):
    rng = np.random.default_rng(seed)
    tt = df["extra_info"].apply(lambda e: (e if isinstance(e, dict) else json.loads(e)).get("task_type"))
    g_idx = df.index[tt == "grounding"].to_numpy()
    r_idx = df.index[tt == "referring"].to_numpy()
    if from_v1:
        v1keys = _load_v1_keys(from_v1)
        ids = df["images"].apply(_img_id_fast).to_numpy()
        tta = tt.to_numpy()
        keep = np.array([(ids[i], tta[i]) in v1keys for i in range(len(df))])  # RangeIndex: pos==index
        g_idx = g_idx[keep[g_idx]]
        r_idx = r_idx[keep[r_idx]]
        print(f"[from_v1] vetted-sample pool -> grounding {len(g_idx)} | referring {len(r_idx)} "
              f"(v1 keys={len(v1keys)})", flush=True)
    pick = []
    if task in ("grounding", "both"):
        n_g = n // 2 if task == "both" else n
        pick += _pick_grounding_stratified(df, g_idx, min(n_g, len(g_idx)), hard_frac, rng)
    if task in ("referring", "both"):
        n_r = (n - n // 2) if task == "both" else n
        pick += list(rng.choice(r_idx, min(n_r, len(r_idx)), replace=False))
    rng.shuffle(pick)
    return [df.loc[i] for i in pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/train.parquet")
    ap.add_argument("--out", default="/workspace/data/hoi_sft_regen_v2")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--task", choices=["grounding", "referring", "both"], default="both")
    ap.add_argument("--hard_frac", type=float, default=0.6, help="(reserved) weight toward s_ref<=0.5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", type=int, default=0, help="if >0, run this many samples, workers=1")
    ap.add_argument("--save_all", action="store_true",
                    help="write debug.jsonl with EVERY trace (kept+dropped) incl. reasoning + zoom_log")
    ap.add_argument("--from_v1", default=None,
                    help="path to v1 dataset.json; restrict generation to its vetted (image_id,task) samples")
    args = ap.parse_args()
    if args.smoke:
        args.n = args.smoke; args.workers = 1

    df = pd.read_parquet(args.src)
    rows = select_rows(df, args.n, args.task, args.hard_frac, args.seed, from_v1=args.from_v1)
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)
    print(f"model={MODEL} samples={len(rows)} workers={args.workers} out={out}")

    def _safe(row, idx):
        try:
            return process_row(row, out / "images", idx)
        except Exception as e:
            return {"uid": f"row{idx}", "kept": False, "reason": f"worker_crash:{type(e).__name__}:{e}"}

    results = []
    # INCREMENTAL writes (multi-hour run safe vs timeout/crash): kept samples -> dataset.jsonl,
    # every trace -> debug.jsonl, flushed per sample. dataset.json (array) assembled at the end.
    ds_jsonl = open(out / "dataset.jsonl", "w")
    dbg_fh = open(out / "debug.jsonl", "w") if args.save_all else None
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_safe, r, i) for i, r in enumerate(rows)]
        for f in as_completed(futs):
            r = f.result(); results.append(r)
            if r.get("kept"):
                ds_jsonl.write(json.dumps(r["sample"], ensure_ascii=False) + "\n"); ds_jsonl.flush()
            if dbg_fh is not None:
                dbg_fh.write(json.dumps({k: v for k, v in r.items() if k != "sample"}, ensure_ascii=False) + "\n")
                dbg_fh.flush()
            tag = "KEEP" if r.get("kept") else f"drop({r.get('reason','?')[:40]})"
            print(f"  [{len(results)}/{len(rows)}] {str(r.get('uid'))[:8]} {tag} s_final={r.get('s_final')}", flush=True)
    ds_jsonl.close()
    if dbg_fh is not None:
        dbg_fh.close()

    kept = [r for r in results if r.get("kept")]
    with open(out / "dataset.json", "w") as fh:  # final array form for LLaMA-Factory
        json.dump([r["sample"] for r in kept], fh, ensure_ascii=False, indent=1)
    in_t, out_t, rt = _USAGE["input"], _USAGE["output"], _USAGE["reasoning"]
    cost_sync = in_t / 1e6 * 5 + out_t / 1e6 * 30  # gpt-5.5 standard (sync); batch = 50%
    stats = {"attempted": len(rows), "kept": len(kept), "yield": len(kept) / max(1, len(rows)),
             "drop_reasons": _count([r.get("reason") for r in results if not r.get("kept")]),
             "usage": _USAGE, "est_cost_usd_sync": round(cost_sync, 2),
             "est_cost_usd_batch": round(cost_sync * 0.5, 2),
             "sref_tiers_kept": _count([_tier(r.get("s_ref")) for r in kept if r.get("task_type") == "grounding"]),
             "tool_use_kept": _count([("tool" if r.get("n_tool") else "no_tool") for r in kept])}
    with open(out / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print("\n=== STATS ===\n" + json.dumps(stats, indent=2))


def _count(xs):
    d = {}
    for x in xs:
        d[str(x)] = d.get(str(x), 0) + 1
    return d


def _tier(s):
    if s is None:
        return "na"
    if s <= 1e-6:
        return "s_ref=0"
    if s < 0.5 - 1e-6:
        return "s_ref<0.5"
    if abs(s - 0.5) < 1e-6:
        return "s_ref=0.5"
    return "s_ref>0.5"


if __name__ == "__main__":
    main()
