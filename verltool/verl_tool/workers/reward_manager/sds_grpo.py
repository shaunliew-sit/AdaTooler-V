"""
SDS-GRPO: Spatial Difficulty-Gated GRPO Reward Manager for HOI Adaptive Tool Use.

Reward formula:
    R_total = R_format * (R_outcome + alpha * R_tool)

Where:
    - R_format: Binary gate (1 if valid <think>/<answer> tags, 0 otherwise)
    - R_outcome: Task-specific accuracy (IoU+AR for grounding, ROUGE-L+METEOR for referring)
    - R_tool: Adaptive tool reward with separate zoom_in/zoom_out treatment:
        R_tool = R_zoom_in + R_zoom_out_hygiene

Tool reward design rationale (grounded in pain point analysis):
    1. zoom_in is the information-gathering action — gated by object size (SDS)
    2. zoom_out is a navigation action (context restore) — free, with small hygiene bonus
    3. Object area (not min of person/object) drives SDS, because the pain point data
       shows objects are the perceptual bottleneck (e.g., baseball 6px², tie 8px²)
    4. Optimal zoom_in count is dynamic: n_opt = round(2 * SDS_obj)
       - SDS high (tiny objects): n_opt=2, multi-region zoom encouraged
       - SDS medium: n_opt=1, single zoom usually sufficient
       - SDS low (large objects): n_opt=0, no zoom needed
"""
import json
import math
import re
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from verl import DataProto
from verl.workers.reward_manager import register


# ---------------------------------------------------------------------------
# IoU & Greedy Matching
# ---------------------------------------------------------------------------

def calculate_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    if len(box1) < 4 or len(box2) < 4:
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def match_pairs_greedy(
    pred_pairs: list[list[dict]],
    gt_pairs: list[list[dict]],
    threshold: float,
) -> int:
    """Greedy-match predicted pairs to GT pairs.

    Each pair is [person_dict, object_dict] with "bbox_2d" keys.
    Returns count of matched pairs whose pair_score >= threshold.
    """
    if not pred_pairs or not gt_pairs:
        return 0

    # Compute all pair scores
    scores: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(pred_pairs):
        for gi, gt in enumerate(gt_pairs):
            person_iou = calculate_iou(pred[0]["bbox_2d"], gt[0]["bbox_2d"])
            object_iou = calculate_iou(pred[1]["bbox_2d"], gt[1]["bbox_2d"])
            pair_score = 0.5 * person_iou + 0.5 * object_iou
            scores.append((pair_score, pi, gi))

    # Sort descending by score
    scores.sort(key=lambda x: x[0], reverse=True)

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    count = 0
    for score, pi, gi in scores:
        if pi in matched_pred or gi in matched_gt:
            continue
        if score >= threshold:
            count += 1
            matched_pred.add(pi)
            matched_gt.add(gi)
    return count


# ---------------------------------------------------------------------------
# Grounding answer parsing & outcome
# ---------------------------------------------------------------------------

def parse_grounding_answer(text: str) -> list[list[dict]]:
    """Parse grounding answer text into list of [person, object] pairs.

    Handles both the <answer>...</answer> wrapped format and raw JSON lines.
    Each line should be a JSON array of two dicts with "bbox_2d" and "label".
    """
    # Extract content from <answer> tags if present
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    content = match.group(1).strip() if match else text.strip()

    pairs: list[list[dict]] = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, list) and len(parsed) == 2:
                if all(
                    isinstance(d, dict)
                    and "bbox_2d" in d
                    and isinstance(d["bbox_2d"], list)
                    and len(d["bbox_2d"]) == 4
                    for d in parsed
                ):
                    pairs.append(parsed)
        except (json.JSONDecodeError, TypeError):
            continue
    return pairs


def compute_grounding_outcome(pred_text: str, gt_data: dict) -> float:
    """Compute grounding outcome: Average Recall at IoU {0.5, 0.75}.

    gt_data should have "boxes_1000" (flat list of coords) and "num_pairs".
    """
    pred_pairs = parse_grounding_answer(pred_text)

    # Reconstruct GT pairs from boxes_1000
    boxes_1000 = gt_data.get("boxes_1000", [])
    num_pairs = gt_data.get("num_pairs", 0)

    gt_pairs: list[list[dict]] = []
    for i in range(num_pairs):
        person_idx = i * 2
        object_idx = i * 2 + 1
        if object_idx < len(boxes_1000):
            gt_pairs.append([
                {"bbox_2d": boxes_1000[person_idx]},
                {"bbox_2d": boxes_1000[object_idx]},
            ])

    if not gt_pairs:
        # No GT pairs — if pred also empty, perfect; otherwise 0
        return 1.0 if not pred_pairs else 0.0

    recall_05 = match_pairs_greedy(pred_pairs, gt_pairs, 0.5) / len(gt_pairs)
    recall_075 = match_pairs_greedy(pred_pairs, gt_pairs, 0.75) / len(gt_pairs)
    return (recall_05 + recall_075) / 2.0


# ---------------------------------------------------------------------------
# Referring answer parsing & outcome
# ---------------------------------------------------------------------------

def parse_referring_answer(text: str) -> str:
    """Extract action phrase from referring answer."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    content = match.group(1).strip() if match else text.strip()
    # Normalize
    return content.lower().strip()


def _rouge_l_f1(pred: str, ref: str) -> float:
    """Compute ROUGE-L F1 between two strings (word-level LCS)."""
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    m = len(pred_tokens)
    n = len(ref_tokens)

    # LCS via DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0
    precision = lcs_len / m
    recall = lcs_len / n
    return 2.0 * precision * recall / (precision + recall)


def _meteor_score(pred: str, ref: str) -> float:
    """Compute a simplified METEOR-like score.

    Uses unigram precision/recall with harmonic mean and fragmentation penalty.
    Falls back to nltk.meteor_score if available.
    """
    try:
        import nltk
        from nltk.translate.meteor_score import single_meteor_score
        return single_meteor_score(ref.split(), pred.split())
    except (ImportError, LookupError, TypeError, AttributeError, AssertionError):
        pass

    # Fallback: unigram F1 with fragmentation penalty
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    matches = len(pred_set & ref_set)
    if matches == 0:
        return 0.0

    precision = matches / len(pred_tokens)
    recall = matches / len(ref_tokens)
    f_mean = (10.0 * precision * recall) / (9.0 * precision + recall)

    # Simple chunk penalty
    chunks = 1
    prev_in_ref = False
    for token in pred_tokens:
        if token in ref_set:
            if not prev_in_ref:
                chunks += 1
            prev_in_ref = True
        else:
            prev_in_ref = False
    chunks = max(1, chunks - 1)

    penalty = 0.5 * (chunks / matches) ** 3 if matches > 0 else 0.0
    return f_mean * (1.0 - penalty)


def compute_referring_outcome(pred_text: str, gt_data: dict) -> float:
    """Compute referring outcome: max(exact_match, 0.5*ROUGE-L + 0.5*METEOR)."""
    pred = parse_referring_answer(pred_text)
    gt = gt_data.get("response", "").lower().strip()

    if not gt:
        return 1.0 if not pred else 0.0

    exact_match = 1.0 if pred == gt else 0.0
    rouge_l = _rouge_l_f1(pred, gt)
    meteor = _meteor_score(pred, gt)
    text_sim = 0.5 * rouge_l + 0.5 * meteor
    return max(exact_match, text_sim)


# ---------------------------------------------------------------------------
# Format Gate
# ---------------------------------------------------------------------------

def compute_format_reward(response: str, task_type: str) -> float:
    """Check that response has valid <think>...</think> and <answer>...</answer> tags."""
    pattern = re.compile(r"<think>(.*?)</think>.*<answer>(.*?)</answer>", re.DOTALL)
    match = pattern.search(response)
    if not match:
        return 0.0

    # For both task types, valid <think> + <answer> tags are sufficient
    return 1.0


# ---------------------------------------------------------------------------
# Spatial Difficulty Score (SDS) — Object-Area Based
# ---------------------------------------------------------------------------

def compute_sds(boxes_1000: list[list[float]], num_pairs: int) -> float:
    """Compute SDS from GT **object** bounding box areas in 1000-grid coordinates.

    Uses object area only (not min of person/object) because the pain point
    analysis shows objects are the perceptual bottleneck:
      - baseball: 6 px², person: 166,579 px² (ratio 1:27,763)
      - tie: 8 px², person: 60,000 px² (ratio 1:7,500)
      - screw: 26 px², person: 221,774 px² (ratio 1:8,530)
    The person is almost never the resolution bottleneck.

    boxes_1000: flat list of [x1,y1,x2,y2] boxes, alternating person/object.
    num_pairs: number of person-object pairs.

    Returns SDS in [0, 1]. High = small objects = zoom likely helps.
    """
    if num_pairs == 0 or not boxes_1000:
        return 0.5  # neutral for empty

    min_obj_area = float("inf")
    for i in range(num_pairs):
        object_idx = i * 2 + 1
        if object_idx >= len(boxes_1000):
            break

        object_box = boxes_1000[object_idx]
        obj_area = (
            max(0.0, object_box[2] - object_box[0])
            * max(0.0, object_box[3] - object_box[1])
            / (1000.0 * 1000.0)
        )
        min_obj_area = min(min_obj_area, obj_area)

    if min_obj_area == float("inf"):
        return 0.5

    # Linear mapping: obj_area <= 0.01 -> SDS=1.0, obj_area >= 0.15 -> SDS=0.0
    sds = (0.15 - min_obj_area) / 0.14
    return float(np.clip(sds, 0.0, 1.0))


def compute_sds_referring(gt_data: dict) -> float:
    """Compute SDS for referring task from object_bbox only.

    For referring, the person and object bboxes are given explicitly.
    We use object area only, consistent with the grounding SDS.
    """
    object_bbox = gt_data.get("object_bbox")
    if not object_bbox:
        return 0.5

    obj_area = (
        max(0.0, object_bbox[2] - object_bbox[0])
        * max(0.0, object_bbox[3] - object_bbox[1])
        / (1000.0 * 1000.0)
    )
    sds = (0.15 - obj_area) / 0.14
    return float(np.clip(sds, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Tool Counting — Separate zoom_in and zoom_out
# ---------------------------------------------------------------------------

_ZOOM_IN_PATTERN = re.compile(
    r"<tool_call>\s*\{[^}]*\"name\"\s*:\s*\"(zoom_in|crop_image)\"",
    re.DOTALL,
)

_ZOOM_OUT_PATTERN = re.compile(
    r"<tool_call>\s*\{[^}]*\"name\"\s*:\s*\"zoom_out\"",
    re.DOTALL,
)


def count_zoom_in(response: str) -> int:
    """Count zoom_in and crop_image tool calls (information-gathering actions)."""
    return len(_ZOOM_IN_PATTERN.findall(response))


def count_zoom_out(response: str) -> int:
    """Count zoom_out tool calls (context-restore actions)."""
    return len(_ZOOM_OUT_PATTERN.findall(response))


# ---------------------------------------------------------------------------
# Tool Reward — Redesigned with Separate Treatment
# ---------------------------------------------------------------------------

def compute_tool_reward(
    sds: float,
    n_zoom_in: int,
    n_zoom_out: int,
    tau: float = 0.3,
    gamma: float = 2.0,
    lambda_abstain: float = 0.1,
    lambda_hygiene: float = 0.05,
) -> tuple[float, float, float, int]:
    """Compute adaptive tool reward with separate zoom_in/zoom_out treatment.

    Returns (r_tool, r_zoom_in, r_zoom_out_hygiene, n_opt) for logging.

    Design:
        R_tool = R_zoom_in + R_zoom_out_hygiene

    R_zoom_in: Object-size-gated reward for information-gathering.
        - n_opt = round(2 * SDS): dynamic optimal zoom_in count
        - When n_opt > 0: R_zoom_in = (SDS - tau) * G(n_zoom_in, n_opt)
          where G is a Gaussian efficiency function
        - When n_opt == 0: R_zoom_in = -(1 - SDS) * lambda_abstain * n_zoom_in
          (linear penalty for unnecessary zooming on easy images)

    R_zoom_out_hygiene: Small bonus for pairing zoom_out with zoom_in.
        - R_zoom_out_hygiene = lambda_hygiene * min(n_zoom_out, n_zoom_in) / max(n_zoom_in, 1)
        - Encourages context restoration without penalizing its absence
          (38% of referring SFT uses zoom_in only — this must not be penalized)
    """
    # Dynamic optimal zoom_in count based on object size difficulty
    n_opt = round(2.0 * sds)

    # --- R_zoom_in ---
    if n_opt > 0:
        # Gaussian decay around optimal zoom_in count
        deviation = (n_zoom_in - n_opt) / max(n_opt, 0.5)
        gaussian = math.exp(-gamma * deviation * deviation)
        r_zoom_in = (sds - tau) * gaussian
    else:
        # Easy image (large objects): penalize unnecessary zooming
        if n_zoom_in > 0:
            r_zoom_in = -(1.0 - sds) * lambda_abstain * n_zoom_in
        else:
            # No zoom on easy image: neutral (reward comes from R_outcome)
            r_zoom_in = 0.0

    # --- R_zoom_out_hygiene ---
    if n_zoom_in > 0 and n_zoom_out > 0:
        r_zoom_out_hygiene = lambda_hygiene * min(n_zoom_out, n_zoom_in) / n_zoom_in
    else:
        r_zoom_out_hygiene = 0.0

    r_tool = r_zoom_in + r_zoom_out_hygiene
    return r_tool, r_zoom_in, r_zoom_out_hygiene, n_opt


# ---------------------------------------------------------------------------
# Reward Manager
# ---------------------------------------------------------------------------

@register("SDS-GRPO")
class SDSGRPORewardManager:
    """Reward manager for HOI tasks using Spatial Difficulty-Gated GRPO.

    Key design decisions (grounded in pain point analysis):
    1. SDS uses object area only — objects are the resolution bottleneck
       (ARs = 3.76% Qwen3VL, 0.10% Groma on HICO small objects)
    2. zoom_in and zoom_out counted separately — they serve different functions
       (SFT: 38% of referring uses zoom_in only; 45% of grounding uses in→out)
    3. Dynamic n_opt = round(2*SDS) — matches SFT distribution
       (69% use 1 zoom_in, 28% use 2 zoom_ins)
    4. zoom_out is free with small hygiene bonus — not an information action
    """

    name = "sds_grpo"

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Any = None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

        # Hyperparameters
        self.alpha = 0.6              # tool reward weight in R_total
        self.tau = 0.3                # SDS threshold for zoom_in gating
        self.gamma = 2.0              # Gaussian decay steepness
        self.lambda_abstain = 0.1     # per-zoom_in penalty on easy images
        self.lambda_hygiene = 0.05    # zoom_out hygiene bonus

        # Pre-warm NLTK WordNet corpus so it is loaded into memory before
        # concurrent RewardManagerWorker threads call single_meteor_score.
        # NLTK's ZipFilePathPointer.read() is not thread-safe; concurrent access
        # triggers "assert self.fp is None" (AssertionError).  Loading once here
        # caches the corpus data and prevents the race condition.
        try:
            from nltk.corpus import wordnet as _wn
            _wn.ensure_loaded()
        except Exception:
            try:
                from nltk.corpus import wordnet as _wn
                _wn.synsets("test")  # fallback for NLTK versions without ensure_loaded
            except Exception:
                pass

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)
        already_printed: dict[str, int] = {}

        for i in range(len(data)):
            data_item = data[i]

            # Decode prompt and response
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth_raw = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            # Parse ground truth
            if isinstance(ground_truth_raw, str):
                try:
                    gt_data = json.loads(ground_truth_raw)
                except (json.JSONDecodeError, TypeError):
                    gt_data = {"response": ground_truth_raw}
            elif isinstance(ground_truth_raw, dict):
                gt_data = ground_truth_raw
            else:
                gt_data = {}

            task_type = extra_info.get("task_type", "grounding")

            # --- R_format ---
            r_format = compute_format_reward(response_str, task_type)

            # --- R_outcome ---
            if task_type == "grounding":
                r_outcome = compute_grounding_outcome(response_str, gt_data)
            else:
                r_outcome = compute_referring_outcome(response_str, gt_data)

            # --- SDS (object-area based) ---
            sds = extra_info.get("spatial_difficulty_score", None)
            if sds is None:
                if task_type == "grounding":
                    sds = compute_sds(
                        gt_data.get("boxes_1000", []),
                        gt_data.get("num_pairs", 0),
                    )
                else:
                    sds = compute_sds_referring(gt_data)

            # --- Tool counts (separate zoom_in and zoom_out) ---
            n_zoom_in = count_zoom_in(response_str)
            n_zoom_out = count_zoom_out(response_str)

            # --- R_tool ---
            r_tool, r_zoom_in, r_zoom_out_hygiene, n_opt = compute_tool_reward(
                sds,
                n_zoom_in,
                n_zoom_out,
                self.tau,
                self.gamma,
                self.lambda_abstain,
                self.lambda_hygiene,
            )

            # --- R_total ---
            r_total = r_format * (r_outcome + self.alpha * r_tool)

            accuracy = 1.0 if r_outcome > 0 else 0.0

            # Store metrics
            score_dict = {
                "score": r_total,
                "accuracy": accuracy,
                "r_format": r_format,
                "r_outcome": r_outcome,
                "r_tool": r_tool,
                "r_zoom_in": r_zoom_in,
                "r_zoom_out_hygiene": r_zoom_out_hygiene,
                "sds": sds,
                "n_opt": float(n_opt),
                "n_zoom_in": float(n_zoom_in),
                "n_zoom_out": float(n_zoom_out),
            }

            for key, value in score_dict.items():
                reward_extra_info[key].append(value)

            if accuracy > 0:
                reward_extra_info["correct_response_length"].append(valid_response_length)
            else:
                reward_extra_info["wrong_response_length"].append(valid_response_length)

            # Use accuracy for validation (num_examine==1)
            reward = score_dict["accuracy"] if self.num_examine == 1 else r_total
            reward_tensor[i, valid_response_length - 1] = reward

            # Debug printing
            if data_source not in already_printed:
                already_printed[data_source] = 0
            if already_printed[data_source] < self.num_examine:
                already_printed[data_source] += 1
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                print(f"[prompt] {prompt_str[:200]}...")
                print(f"[response] {response_str[:300]}...")
                print(f"[ground_truth] {ground_truth_raw}")
                for key, value in score_dict.items():
                    print(f"[{key}] {value}")

            # Crop tool_interact_info images for logging
            tool_interact_info = data_item.non_tensor_batch.get("tool_interact_info", None)
            if tool_interact_info is not None:
                for tool_interact in tool_interact_info:
                    if "image" in tool_interact:
                        if isinstance(tool_interact["image"], list):
                            tool_interact["image"] = [x[:50] for x in tool_interact["image"]]
                        elif isinstance(tool_interact["image"], str):
                            tool_interact["image"] = tool_interact["image"][:50]

        # Aggregate response lengths
        correct_len_mean = (
            np.mean(reward_extra_info["correct_response_length"])
            if reward_extra_info["correct_response_length"]
            else 0.0
        )
        wrong_len_mean = (
            np.mean(reward_extra_info["wrong_response_length"])
            if reward_extra_info["wrong_response_length"]
            else 0.0
        )
        reward_extra_info["correct_response_length"] = [correct_len_mean] * len(reward_tensor)
        reward_extra_info["wrong_response_length"] = [wrong_len_mean] * len(reward_tensor)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(sorted(reward_extra_info.items())),
            }
        return reward_tensor
