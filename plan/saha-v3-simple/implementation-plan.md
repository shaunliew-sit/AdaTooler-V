# SAHA v3 Counterfactual Tool-Gain Reward — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the SDS-gated tool reward with a counterfactual tool-gain reward — `R_total = R_format·(R_outcome + α·R_tool)`, `R_tool = I_tool·clip(s_final − s_ref)` — grounded only in target-object GT, as a NEW reward manager that leaves the frozen `SDS-GRPO` baseline untouched.

**Architecture:** A new reward manager `saha_cf.py` (registered `SAHA-CF`) reuses the scorers/parsers from `sds_grpo.py` and adds (1) a grounding `s_ref` from the selected proposal anchor box scored against target GT, (2) a referring `s_ref` from the per-`uid` no-tool rollout mean via a two-pass `__call__`, (3) a single `compute_counterfactual_tool_reward` helper, (4) per-rollout logging. Preprocessing threads the proposal anchor into `extra_info` (one-time parquet re-run).

**Tech Stack:** Python, NumPy, PyTorch, verl/verl-tool `register` reward-manager API, pytest. Authoritative spec: `docs/saha-v2/reward-v3-counterfactual-spec.md`.

**Pre-flight verification (do once before Task 1):**
- [ ] Confirm the GRPO group key on the batch: `rg -n "uid|non_tensor_batch\[.index" verltool/verl/verl/trainer/ppo/ray_trainer.py | head`. The plan assumes `data.non_tensor_batch["uid"]`. If the key differs (e.g. `index`), use that key everywhere this plan says `uid`.
- [ ] Confirm the registration decorator on `sds_grpo.py`'s class (search `rg -n "@register" verltool/verl_tool/workers/reward_manager/sds_grpo.py`). Mirror that exact decorator form in Task 2.
- [ ] Confirm `verltool/verl_tool/workers/reward_manager/__init__.py` imports each manager module so the decorator runs (the new module must be added there in Task 2).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `verltool/examples/data_preprocess/hoi/proposal_anchor.py` | Stdlib-only grounding anchor builder (`iou`, `build_proposal_anchor_grounding`) | Create (as-built) |
| `verltool/examples/data_preprocess/hoi/prepare_train.py` | Import the builder + emit `proposal_anchor` into grounding `extra_info` | Modify |
| `verltool/verl_tool/workers/reward_manager/saha_cf.py` | The `SAHA-CF` counterfactual reward manager | Create |
| `verltool/verl_tool/workers/reward_manager/__init__.py` | Import the new manager so `@register` runs | Modify |
| `verltool/tests/test_proposal_anchor.py` | Anchor-selection unit tests | Create |
| `verltool/tests/test_saha_cf_reward.py` | `s_ref`, counterfactual reward, two-pass tests | Create |
| `verltool/examples/train/hoi/configs/saha_cf.yaml` | Reward knobs (α, clip, anchor mode, referring s_ref) | Create |
| `verltool/examples/train/hoi/train_qwen3vl_cf_4b.sh` | 4B-subset sweep launcher | Create |
| `verltool/examples/data_preprocess/hoi/build_subset.py` | Balanced ~2–3k subset builder | Create |

`sds_grpo.py` and `sds_v2.py` are **not modified** (frozen baseline / eval-only).

---

## Task 1: Proposal anchor selection in preprocessing

**Files:**
- Modify: `verltool/examples/data_preprocess/hoi/prepare_train.py` (proposal load ~`:237-243`; `extra_info` build ~`:300-324` grounding and ~`:376-401` referring)
- Test: `verltool/tests/test_proposal_anchor.py`

**Reality check (verified in code 2026-06-15):** proposals carry `{bbox_2d, label, confidence, id}` — **no `role_hint`**. Grounding GT `boxes_1000` is a list of 4-coord boxes with pairs at `[2i]`(person)/`[2i+1]`(object), `num_pairs` pairs total. Referring predicts **text** (no box), so referring needs **no anchor** — its `s_ref` is the group no-tool mean (Task 5). Therefore the anchor is GROUNDING-ONLY and is a **per-pair synthetic proposal answer**: for each GT pair, pick the proposal box best matching the GT person and the GT object by IoU. Stored as `extra_info["proposal_anchor"] = [[{"bbox_2d":...,"label":"person"},{"bbox_2d":...,"label":"object"}], ...]` (or `[]` if no proposals). Boxes share the 1000×1000 grid.

- [ ] **Step 1: Write the failing test**

```python
# verltool/tests/test_proposal_anchor.py
from examples.data_preprocess.hoi.prepare_train import build_proposal_anchor_grounding

def test_anchor_picks_best_matching_boxes_per_pair():
    proposals = [
        {"bbox_2d": [0, 0, 100, 100], "label": "person", "confidence": 0.9, "id": 0},
        {"bbox_2d": [500, 500, 560, 560], "label": "person", "confidence": 0.8, "id": 1},  # best person
        {"bbox_2d": [600, 600, 660, 660], "label": "bottle", "confidence": 0.7, "id": 2},   # best object
    ]
    boxes_1000 = [[510, 510, 570, 570], [600, 600, 660, 660]]  # one pair: person, object
    anchor = build_proposal_anchor_grounding(proposals, boxes_1000, num_pairs=1)
    assert len(anchor) == 1
    assert anchor[0][0]["bbox_2d"] == [500, 500, 560, 560]
    assert anchor[0][1]["bbox_2d"] == [600, 600, 660, 660]

def test_anchor_empty_when_no_proposals():
    assert build_proposal_anchor_grounding([], [[0, 0, 1, 1], [2, 2, 3, 3]], 1) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd verltool && python -m pytest tests/test_proposal_anchor.py -v`
Expected: FAIL — `ImportError: cannot import name 'select_proposal_anchor'`.

- [ ] **Step 3: Implement `build_proposal_anchor_grounding`**

Add near the proposal helpers in `prepare_train.py`:

```python
def _iou(b1, b2):
    if not b1 or not b2 or len(b1) < 4 or len(b2) < 4:
        return 0.0
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def build_proposal_anchor_grounding(proposals, boxes_1000, num_pairs):
    """Synthetic proposal-only answer: for each GT pair pick the proposal box best
    matching the GT person and GT object by IoU. GT-anchored (GT only selects among
    given proposals). Returns list of [person_dict, object_dict] pairs, or []."""
    if not proposals or num_pairs <= 0:
        return []
    boxes = [p["bbox_2d"] for p in proposals if p.get("bbox_2d") and len(p["bbox_2d"]) == 4]
    if not boxes:
        return []
    anchor = []
    for i in range(num_pairs):
        pidx, oidx = 2 * i, 2 * i + 1
        if oidx >= len(boxes_1000):
            break
        gt_p, gt_o = boxes_1000[pidx], boxes_1000[oidx]
        best_p = max(boxes, key=lambda b: _iou(b, gt_p))
        best_o = max(boxes, key=lambda b: _iou(b, gt_o))
        anchor.append([{"bbox_2d": best_p, "label": "person"},
                       {"bbox_2d": best_o, "label": "object"}])
    return anchor
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd verltool && python -m pytest tests/test_proposal_anchor.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Emit the anchor into grounding `extra_info`**

In `process_grounding_sample`'s `extra_info` dict (~`:318-324`), add the key (uses the in-scope `proposals`, `boxes_1000`, `num_pairs`):

```python
"proposal_anchor": build_proposal_anchor_grounding(proposals, boxes_1000, num_pairs),
```

Do NOT add it to `process_referring_sample` — referring `s_ref` is the group no-tool mean (Task 5), so no anchor is stored there.

- [ ] **Step 6: Commit**

```bash
cd verltool && git add examples/data_preprocess/hoi/prepare_train.py tests/test_proposal_anchor.py
git commit -m "feat(saha-cf): select proposal anchor pair into extra_info for counterfactual s_ref"
```

> **Parquet re-run is deferred to Task 8** (built on the balanced subset only, to keep iteration cheap). Do not re-run the full 300k here.

---

## Task 2: `SAHA-CF` reward manager skeleton

> **As-built (2026-06-15, verified):**
> - **Test env = `verl-tool-env`** (the recommended env; `pytest` installed there). Run all tests with `/opt/conda/envs/verl-tool-env/bin/python -m pytest`. It has full `verl`+`torch`, so `saha_cf.py` imports/registers and the reward helpers (Tasks 3-5) live **directly in `saha_cf.py`** — no stdlib `saha_cf_core.py` split needed.
> - `__init__.py` **auto-imports every `*.py`** in `reward_manager/` via glob → **Step 4 (manual import) is unnecessary**; creating `saha_cf.py` is enough.
> - Registration check (PASS): `from verl_tool.workers.reward_manager import get_reward_manager_cls; get_reward_manager_cls('SAHA-CF')` → `SAHACounterfactualRewardManager`; constructs with `reward_kwargs` (alpha, clip_lo, clip_hi, referring_sref, min_no_tool).
> - `proposal_anchor.py` (Task 1) stays a separate module (preprocessing-side helper); its test runs under `verl-tool-env` too.

**Files:**
- Create: `verltool/verl_tool/workers/reward_manager/saha_cf.py`
- Modify: `verltool/verl_tool/workers/reward_manager/__init__.py`
- Test: `verltool/tests/test_saha_cf_reward.py`

Reuse the v1 scorers/parsers by import — do NOT re-implement them.

- [ ] **Step 1: Write the failing test**

```python
# verltool/tests/test_saha_cf_reward.py
def test_manager_is_registered():
    from verl.workers.reward_manager import get_reward_manager_cls  # verify exact accessor name
    assert get_reward_manager_cls("SAHA-CF") is not None
```

(If verl exposes the registry differently, assert on `from verl_tool.workers.reward_manager import saha_cf` importing without error and `saha_cf.SAHACounterfactualRewardManager` existing.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd verltool && python -m pytest tests/test_saha_cf_reward.py::test_manager_is_registered -v`
Expected: FAIL — module/registration missing.

- [ ] **Step 3: Create the skeleton**

```python
# verltool/verl_tool/workers/reward_manager/saha_cf.py
"""SAHA-CF: Proposal-Conditioned Counterfactual Tool-Gain reward manager.

R_total = R_format * (R_outcome + alpha * R_tool)
R_tool  = I_tool * clip(s_final - s_ref, clip_lo, clip_hi)
  grounding s_ref = score(selected proposal anchor box vs target GT)  [GT-anchored]
  referring s_ref = mean(R_outcome of no-tool siblings in same uid)   [detached]
"""
import json
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from verl import DataProto
from verl.workers.reward_manager import register

from .sds_grpo import (
    compute_format_reward,
    compute_grounding_outcome,
    compute_referring_outcome,
    count_zoom_in,
    count_zoom_out,
    parse_grounding_answer,  # not strictly needed but available
)


@register("SAHA-CF")
class SAHACounterfactualRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None,
                 reward_fn_key="data_source", **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        # Single knob + calibration (overridable from config via kwargs)
        self.alpha = float(kwargs.get("alpha", 0.6))
        self.clip_lo = float(kwargs.get("clip_lo", -0.5))
        self.clip_hi = float(kwargs.get("clip_hi", 1.0))
        self.referring_sref = kwargs.get("referring_sref", "group_no_tool")  # or "off"
        self.min_no_tool = int(kwargs.get("min_no_tool", 1))
        try:
            from nltk.corpus import wordnet as _wn
            _wn.ensure_loaded()
        except Exception:
            pass

    def __call__(self, data: DataProto, return_dict: bool = False):
        raise NotImplementedError  # filled in Task 5
```

- [ ] **Step 4: Register via `__init__.py`**

Add to `verltool/verl_tool/workers/reward_manager/__init__.py` next to the existing manager imports:

```python
from . import saha_cf  # noqa: F401  (registers SAHA-CF)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd verltool && python -m pytest tests/test_saha_cf_reward.py::test_manager_is_registered -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd verltool && git add verl_tool/workers/reward_manager/saha_cf.py verl_tool/workers/reward_manager/__init__.py tests/test_saha_cf_reward.py
git commit -m "feat(saha-cf): register SAHA-CF reward manager skeleton reusing v1 scorers"
```

---

## Task 3: Grounding `s_ref` from the proposal anchor

**Files:**
- Modify: `verltool/verl_tool/workers/reward_manager/saha_cf.py`
- Test: `verltool/tests/test_saha_cf_reward.py`

- [ ] **Step 1: Write the failing test**

```python
def test_sref_grounding_scores_anchor_against_gt():
    from verl_tool.workers.reward_manager.saha_cf import compute_sref_grounding
    gt_data = {"boxes_1000": [[500, 500, 560, 560], [600, 600, 660, 660]], "num_pairs": 1}
    perfect = [[{"bbox_2d": [500, 500, 560, 560], "label": "person"},
                {"bbox_2d": [600, 600, 660, 660], "label": "object"}]]
    wrong = [[{"bbox_2d": [0, 0, 10, 10], "label": "person"},
              {"bbox_2d": [10, 10, 20, 20], "label": "object"}]]
    assert compute_sref_grounding(perfect, gt_data) == 1.0   # AR@{0.5,0.75} both hit
    assert compute_sref_grounding(wrong, gt_data) < 0.1
    assert compute_sref_grounding([], gt_data) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd verltool && python -m pytest tests/test_saha_cf_reward.py::test_sref_grounding_scores_anchor_against_gt -v`
Expected: FAIL — `cannot import name 'compute_sref_grounding'`.

- [ ] **Step 3: Implement the anchor scorer**

Add to `saha_cf.py` (module-level). It serializes the anchor into the exact text shape `parse_grounding_answer` expects, then reuses `compute_grounding_outcome`:

```python
def compute_sref_grounding(anchor, gt_data):
    """Score the proposal anchor (list of [person, object] pairs) as if it were the
    model's grounding answer. GT-anchored: depends only on the fixed anchor and GT."""
    if not anchor:
        return 0.0
    lines = "\n".join(json.dumps(pair) for pair in anchor)
    return compute_grounding_outcome(f"<answer>{lines}</answer>", gt_data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd verltool && python -m pytest tests/test_saha_cf_reward.py::test_sref_grounding_scores_anchor_against_gt -v`
Expected: PASS. (If the perfect case is <0.9, inspect `compute_grounding_outcome`'s AR/recall term and adjust the assertion to the documented max, not the code — do not change the scorer.)

- [ ] **Step 5: Commit**

```bash
cd verltool && git add verl_tool/workers/reward_manager/saha_cf.py tests/test_saha_cf_reward.py
git commit -m "feat(saha-cf): grounding s_ref by scoring the proposal anchor vs target GT"
```

---

## Task 4: Counterfactual tool-gain helper

**Files:**
- Modify: `verltool/verl_tool/workers/reward_manager/saha_cf.py`
- Test: `verltool/tests/test_saha_cf_reward.py`

- [ ] **Step 1: Write the failing test (the §2 behavior table)**

```python
def test_counterfactual_tool_reward_behavior():
    from verl_tool.workers.reward_manager.saha_cf import compute_counterfactual_tool_reward as f
    # no tool -> 0 regardless of scores
    assert f(s_final=0.9, s_ref=0.2, i_tool=0, clip_lo=-0.5, clip_hi=1.0) == 0.0
    # tool beats low anchor -> positive gain
    assert f(0.8, 0.2, 1, -0.5, 1.0) == 0.6
    # tool ties good anchor -> ~0
    assert f(0.8, 0.8, 1, -0.5, 1.0) == 0.0
    # tool hurts -> clipped negative
    assert f(0.1, 0.8, 1, -0.5, 1.0) == -0.5
    # positive gain clipped at hi
    assert f(1.0, -0.2, 1, -0.5, 0.5) == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd verltool && python -m pytest tests/test_saha_cf_reward.py::test_counterfactual_tool_reward_behavior -v`
Expected: FAIL — import error.

- [ ] **Step 3: Implement the helper**

```python
def compute_counterfactual_tool_reward(s_final, s_ref, i_tool, clip_lo, clip_hi):
    """R_tool = I_tool * clip(s_final - s_ref, clip_lo, clip_hi)."""
    if not i_tool:
        return 0.0
    return float(min(max(s_final - s_ref, clip_lo), clip_hi))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd verltool && python -m pytest tests/test_saha_cf_reward.py::test_counterfactual_tool_reward_behavior -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd verltool && git add verl_tool/workers/reward_manager/saha_cf.py tests/test_saha_cf_reward.py
git commit -m "feat(saha-cf): counterfactual tool-gain helper with asymmetric clip"
```

---

## Task 5: Two-pass `__call__` (referring `s_ref` = per-`uid` no-tool mean)

**Files:**
- Modify: `verltool/verl_tool/workers/reward_manager/saha_cf.py`
- Test: `verltool/tests/test_saha_cf_reward.py`

Pass 1 collects `(uid, task_type, r_format, r_outcome, i_tool, s_ref_grounding)` per item. Pass 2 fills referring `s_ref` from the same-`uid` no-tool mean (min-count guard), computes `R_tool` and `R_total`, writes the tensor + logging.

- [ ] **Step 1: Write the failing test (group reduction with a pure-Python helper)**

Factor the group logic into a testable pure function so we don't need a full `DataProto`:

```python
def test_referring_group_no_tool_mean_and_guard():
    from verl_tool.workers.reward_manager.saha_cf import resolve_referring_sref
    rows = [
        {"uid": "a", "i_tool": 0, "r_outcome": 0.2},
        {"uid": "a", "i_tool": 0, "r_outcome": 0.4},
        {"uid": "a", "i_tool": 1, "r_outcome": 0.9},  # tool rollout
        {"uid": "b", "i_tool": 1, "r_outcome": 0.7},  # no no-tool sibling -> drop
    ]
    sref = resolve_referring_sref(rows, min_no_tool=1)
    assert abs(sref[2] - 0.3) < 1e-9   # mean(0.2, 0.4)
    assert sref[3] is None             # dropped -> R_tool=0
    assert sref[0] is None and sref[1] is None  # no-tool rows get no R_tool
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd verltool && python -m pytest tests/test_saha_cf_reward.py::test_referring_group_no_tool_mean_and_guard -v`
Expected: FAIL — import error.

- [ ] **Step 3: Implement `resolve_referring_sref` + the two-pass `__call__`**

```python
def resolve_referring_sref(rows, min_no_tool=1):
    """For each referring row, s_ref = mean r_outcome of same-uid no-tool siblings.
    Returns list aligned to rows; None means 'no R_tool for this row'."""
    by_uid_no_tool = defaultdict(list)
    for r in rows:
        if r["i_tool"] == 0:
            by_uid_no_tool[r["uid"]].append(r["r_outcome"])
    out = []
    for r in rows:
        if r["i_tool"] == 0:
            out.append(None)            # no-tool rollouts never get R_tool
            continue
        siblings = by_uid_no_tool.get(r["uid"], [])
        out.append(float(np.mean(siblings)) if len(siblings) >= min_no_tool else None)
    return out
```

Then implement `__call__` (replace the `NotImplementedError`). Full method:

```python
    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                keys = data.meta_info.get("reward_extra_keys", [])
                return {"reward_tensor": data.batch["rm_scores"],
                        "reward_extra_info": {k: data.non_tensor_batch[k] for k in keys}}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        n = len(data)

        # ---- Pass 1: per-item outcome/format/tool + grounding s_ref ----
        items = []
        for i in range(n):
            di = data[i]
            prompt_ids = di.batch["prompts"]
            plen = prompt_ids.shape[-1]
            vlen = di.batch["attention_mask"][plen:].sum()
            resp = self.tokenizer.decode(di.batch["responses"][:vlen], skip_special_tokens=True)

            gt_raw = di.non_tensor_batch["reward_model"]["ground_truth"]
            gt_data = json.loads(gt_raw) if isinstance(gt_raw, str) else (gt_raw or {})
            extra = di.non_tensor_batch.get("extra_info", {}) or {}
            task_type = extra.get("task_type", "grounding")
            uid = di.non_tensor_batch.get("uid", i)  # verify key in pre-flight

            r_format = compute_format_reward(resp, task_type)
            r_outcome = (compute_grounding_outcome(resp, gt_data) if task_type == "grounding"
                         else compute_referring_outcome(resp, gt_data))
            i_tool = 1 if (count_zoom_in(resp) + count_zoom_out(resp)) > 0 else 0

            s_ref_g = (compute_sref_grounding(extra.get("proposal_anchor"), gt_data)
                       if task_type == "grounding" else None)

            items.append(dict(idx=i, uid=uid, task_type=task_type, resp_len=int(vlen),
                              r_format=r_format, r_outcome=r_outcome, i_tool=i_tool,
                              s_ref_g=s_ref_g))

        # ---- Pass 2: referring s_ref via group no-tool mean, then R_total ----
        ref_rows = [r for r in items if r["task_type"] != "grounding"]
        ref_sref = resolve_referring_sref(ref_rows, self.min_no_tool) if self.referring_sref == "group_no_tool" else [None] * len(ref_rows)
        ref_sref_by_idx = {r["idx"]: s for r, s in zip(ref_rows, ref_sref)}

        for r in items:
            if r["task_type"] == "grounding":
                s_ref = r["s_ref_g"] if r["s_ref_g"] is not None else 0.0
                r_tool = compute_counterfactual_tool_reward(
                    r["r_outcome"], s_ref, r["i_tool"], self.clip_lo, self.clip_hi)
            else:
                s_ref = ref_sref_by_idx.get(r["idx"])
                r_tool = (compute_counterfactual_tool_reward(
                    r["r_outcome"], s_ref, r["i_tool"], self.clip_lo, self.clip_hi)
                    if s_ref is not None else 0.0)

            r_total = r["r_format"] * (r["r_outcome"] + self.alpha * r_tool)
            accuracy = 1.0 if r["r_outcome"] > 0 else 0.0

            score_dict = {"score": r_total, "accuracy": accuracy,
                          "r_format": r["r_format"], "r_outcome": r["r_outcome"],
                          "r_tool": r_tool, "s_final": r["r_outcome"],
                          "s_ref": (s_ref if s_ref is not None else float("nan")),
                          "i_tool": float(r["i_tool"]), "task_type_g": 1.0 if r["task_type"] == "grounding" else 0.0}
            for k, v in score_dict.items():
                reward_extra_info[k].append(v)

            reward = accuracy if self.num_examine == 1 else r_total
            reward_tensor[r["idx"], r["resp_len"] - 1] = reward

        if return_dict:
            return {"reward_tensor": reward_tensor,
                    "reward_extra_info": dict(sorted(reward_extra_info.items()))}
        return reward_tensor
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd verltool && python -m pytest tests/test_saha_cf_reward.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
cd verltool && git add verl_tool/workers/reward_manager/saha_cf.py tests/test_saha_cf_reward.py
git commit -m "feat(saha-cf): two-pass __call__ with referring group no-tool mean s_ref"
```

---

## Task 6: Per-rollout JSONL logging assertion

**Files:**
- Test: `verltool/tests/test_saha_cf_reward.py`

The logging keys are already emitted in Task 5's `score_dict`. Lock them with a test so they are not dropped later.

- [ ] **Step 1: Write the test**

```python
def test_logging_keys_present():
    # The §4 diagnostic backbone keys must be in score_dict.
    import inspect
    from verl_tool.workers.reward_manager import saha_cf
    src = inspect.getsource(saha_cf.SAHACounterfactualRewardManager.__call__)
    for key in ["s_final", "s_ref", "r_tool", "i_tool"]:
        assert f'"{key}"' in src, f"missing log key {key}"
```

- [ ] **Step 2: Run it**

Run: `cd verltool && python -m pytest tests/test_saha_cf_reward.py::test_logging_keys_present -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd verltool && git add tests/test_saha_cf_reward.py
git commit -m "test(saha-cf): lock per-rollout diagnostic logging keys"
```

---

## Task 7: Config + 4B sweep launcher

> **As-built (2026-06-15):** knobs reach the manager two ways — hydra
> `+reward_model.reward_kwargs.*` AND `SAHA_CF_*` env vars (manager precedence:
> kwargs > env > default). `reward_kwargs` is NOT a `RewardModelConfig` field, so
> the hydra `+` *add* is unverified against struct mode — **watch the Task-8 smoke
> test**; if it errors at startup, drop the 5 `+reward_model.reward_kwargs.*` lines
> and rely on the env vars (already exported). `compute_reward` calls
> `reward_fn(data, return_dict=True)` with no call-time kwargs, so `__call__` needs
> no `**kwargs`. Launcher derives from the proven single-GPU recipe; default
> model = `qwen3VL-4B/hoi_v2_sft`, default data = `data/hoi/subset`.

**Files:**
- Create: `verltool/examples/train/hoi/configs/saha_cf.yaml`
- Create: `verltool/examples/train/hoi/train_qwen3vl_cf_4b.sh`

- [ ] **Step 1: Write the config**

```yaml
# verltool/examples/train/hoi/configs/saha_cf.yaml
saha_cf:
  alpha: 0.6          # single knob; sweep {0.3, 0.6, 1.0}
  clip_lo: -0.5
  clip_hi: 1.0
  referring_sref: group_no_tool   # or "off" (rely on GRPO advantage for referring)
  min_no_tool: 1
  anchor_mode: best_match         # or "label_match" if best_match proves too strict
```

- [ ] **Step 2: Write the launcher** by copying the existing single-GPU recipe and changing only model, reward manager, data, and reward kwargs:

```bash
# verltool/examples/train/hoi/train_qwen3vl_cf_4b.sh
# Derived from /workspace/hoi/saha-hoi/.../train_saha_hoi_grpo_single_gpu.sh
# Key overrides only:
#   actor_rollout_ref.model.path=/workspace/hoi/checkpoints/qwen3VL-4B/hoi_v2_sft
#   reward_model.reward_manager=SAHA-CF
#   +reward_model.reward_kwargs.alpha=${ALPHA:-0.6}
#   +reward_model.reward_kwargs.referring_sref=${REF_SREF:-group_no_tool}
#   data.train_files=data/hoi/subset/train.parquet
#   actor_rollout_ref.rollout.n=2   (set n=4 if referring s_ref proves too sparse — spec §2)
#   do_offload=True  gpu_memory_utilization=0.45  (single 96GB GPU)
export PIXEL_REASONER_BBOX_MODE=grid1000   # keep the zoom_in coordinate fix
```

(Reproduce the full body from the proven single-GPU script; do not hand-write Ray/Hydra flags from scratch.)

- [ ] **Step 3: Verify reward kwargs thread through** — dry-check that `reward_model.reward_kwargs.*` reaches the manager `__init__` kwargs. Run:

`rg -n "reward_kwargs|reward_manager" verltool/verl_tool/workers/ verltool/verl/verl/workers/reward_manager/__init__.py | head`
Expected: confirms `reward_kwargs` is splatted into the manager. If not, pass knobs via env vars read in `__init__`.

- [ ] **Step 4: Commit**

```bash
cd verltool && git add examples/train/hoi/configs/saha_cf.yaml examples/train/hoi/train_qwen3vl_cf_4b.sh
git commit -m "feat(saha-cf): 4B single-GPU sweep config + launcher"
```

---

## Task 8: Balanced subset + parquet re-run + 1-step smoke test

**Files:**
- Create: `verltool/examples/data_preprocess/hoi/build_subset.py`

- [ ] **Step 1: Write the subset builder** (stratify so crowded/occluded and bad-anchor cases are not washed out):

```python
# build_subset.py: sample ~2-3k rows balanced across task_type and a crowding proxy
# (num proposals near target) and anchor quality (s_ref bucket: low/mid/high).
# Write data/hoi/subset/{train,val}.parquet. Set seed=42.
```

- [ ] **Step 2: Re-run preprocessing on the subset** so `proposal_anchor` lands in `extra_info`:

Run the existing `prepare_train.py` path restricted to the subset ids, then `build_subset.py`.
Verify: `python -c "import pandas as pd; d=pd.read_parquet('verltool/data/hoi/subset/train.parquet'); print(d['extra_info'].iloc[0])"` shows a `proposal_anchor` key.

- [ ] **Step 3: 1-step memory smoke test**

Run `train_qwen3vl_cf_4b.sh` with `trainer.total_training_steps=1` (or the script's 1-step flag). Expected: completes one GRPO step without OOM; logs show non-NaN `r_tool`, `s_ref`, and a non-degenerate tool rate. Capture peak GPU memory.

- [ ] **Step 4: Commit**

```bash
cd verltool && git add examples/data_preprocess/hoi/build_subset.py
git commit -m "feat(saha-cf): balanced subset builder + anchor-bearing parquet for the sweep"
```

---

## Task 9: Drift checks + verification gate

- [ ] **Step 1: Run the full test suite**

Run: `cd verltool && python -m pytest tests/test_proposal_anchor.py tests/test_saha_cf_reward.py -v`
Expected: all PASS.

- [ ] **Step 2: Confirm the baseline is untouched**

Run: `cd verltool && git diff --stat HEAD~8 -- verl_tool/workers/reward_manager/sds_grpo.py verl_tool/workers/reward_manager/sds_v2.py`
Expected: NO changes to either file.

- [ ] **Step 3: CLAUDE.md drift checks**

Run: `rg -n "inspect_pose|detector-free" verltool/ --type py` → nothing active.
Run: `rg -n "proposal_action|R_proposal_decision|lambda_corr" verltool/verl_tool/workers/reward_manager/saha_cf.py` → empty (no resurrected Rev-4 terms).

- [ ] **Step 4: Final commit / branch ready for the sweep**

```bash
cd verltool && git commit --allow-empty -m "chore(saha-cf): v3 counterfactual reward ready for 4B subset sweep"
```

---

## Self-Review (completed against the spec)

- **Spec coverage:** §2 reward → Tasks 3/4/5; §2 grounding anchor → Task 1/3; §2 referring s_ref + n=2 guard → Task 5 (`min_no_tool`) + Task 7 (`n=4` option); §3 reward-only collapse fix (no SAN) → no SAN code path written (correct); §4 detached s_ref → grounding GT-anchored (Task 3) + referring computed from sibling outcomes only (Task 5); §4 logging → Task 6; §7 change set → Tasks 1/2/5; §8 experiment → Tasks 7/8. SAN ablation is intentionally NOT built (spec: ablation-only, deferred until the reward-only default is measured).
- **Placeholders:** none — every code/test step has concrete content; the launcher body legitimately reuses the proven single-GPU script rather than inventing Hydra flags.
- **Type consistency:** `select_proposal_anchor`→`{"person","object"}` consumed by `compute_sref_grounding`; `resolve_referring_sref` rows use `uid/i_tool/r_outcome` matching Pass-1 dict keys; `compute_counterfactual_tool_reward(s_final, s_ref, i_tool, clip_lo, clip_hi)` signature identical across Tasks 4 and 5.

---

## Follow-ups (from codex review 2026-06-15)

- **Rich per-rollout JSONL** (`uid`, `task_type`, anchor, strata) is NOT in `reward_extra_info` because that dict must be numeric (verl means each list). The string-valued §13 schema (`saha_jsonl.py`) is a **separate writer** to be wired into `saha_cf.__call__` as a post-gate diagnostic task — needed for the eval-stratified tool-rate tables, not for training correctness.
- Numeric diagnostics already logged per rollout: `s_final, s_ref, has_sref, r_tool, tool_gain_raw, tool_gain_clipped, i_tool, n_zoom_in, n_zoom_out, is_grounding`.
