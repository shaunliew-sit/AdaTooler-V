"""
Preprocess HOI grounding and referring datasets to parquet format for SDS-GRPO RL training.

Loads 4 train files (HICO/SWIG x grounding/referring), applies quality checks,
loads YOLO-world proposals, constructs prompts matching SFT format, pre-computes SDS,
and outputs train/val parquet files.

Usage:
    cd verltool
    python examples/data_preprocess/hoi/prepare_train.py \
        --data_dir /workspace/Groma/groma_data/benchmarks_simplified \
        --hico_img_dir /workspace/data/hico_20160224_det/images/train2015 \
        --swig_img_dir /workspace/data/swig_hoi/images_512 \
        --proposals_dir /workspace/hoi-tool-use-checkpoints/test_proposals \
        --local_dir data/hoi \
        --max_pairs 15 \
        --filter_len 8192
"""
import importlib.util
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import datasets
import fire
import numpy as np

# SAHA v3: grounding proposal-anchor builder (stdlib-only sibling module). Imported
# robustly so this works whether prepare_train.py is run as a script or imported.
try:
    from proposal_anchor import build_proposal_anchor_grounding
except ImportError:
    _pa_spec = importlib.util.spec_from_file_location(
        "proposal_anchor", str(Path(__file__).resolve().parent / "proposal_anchor.py")
    )
    _pa_mod = importlib.util.module_from_spec(_pa_spec)
    _pa_spec.loader.exec_module(_pa_mod)
    build_proposal_anchor_grounding = _pa_mod.build_proposal_anchor_grounding

# ---------------------------------------------------------------------------
# Proposal-decision convention (SAHA v2 Rev 4, latest-decision.md §10.0)
# ---------------------------------------------------------------------------
# Single source of truth lives in the reward_manager package
# (proposal_action.CONVENTION_TEXT). It is loaded by absolute file path via
# importlib so that this preprocess script does NOT pull in the full training
# stack (the reward_manager package __init__ imports torch/verl). The block is
# only appended to prompts when --proposal_action_convention is passed; the
# default behavior is byte-identical to before.

def _load_convention_text() -> str:
    """Load CONVENTION_TEXT from proposal_action.py by absolute path (no torch)."""
    repo_root = Path(__file__).resolve().parents[3]
    module_path = (
        repo_root
        / "verl_tool"
        / "workers"
        / "reward_manager"
        / "proposal_action.py"
    )
    spec = importlib.util.spec_from_file_location("_saha_proposal_action", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load proposal_action.py at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONVENTION_TEXT

# ---------------------------------------------------------------------------
# System prompt — HOI-specific, only zoom tools (no video tools)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "zoom_in", "description": "Zoom in on the image based on the bounding box coordinates.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "coordinates for bounding box of the area you want to zoom in, in the same 1000x1000 normalized space as the candidate boxes (minimum value 0, maximum value 1000).", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."}}, "required": ["bbox_2d", "target_image"]}}}
{"type": "function", "function": {"name": "zoom_out", "description": "Return to the full original image view after zooming in.", "parameters": {"type": "object", "properties": {}, "required": []}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


GROUNDING_GUIDELINE = """Guidelines: Understand the given visual information and the user query. Determine if it is beneficial to employ the available visual operations (tools).

- You can zoom in using `zoom_in` to get a closer look at specific regions.
- You can zoom out using `zoom_out` to return to the full image view.

Reason with the visual information step by step, iteratively refining your solution using the visual feedback from the tools. Place your text reasoning process within the <think> </think> tags, put any function calls within the <tool_call></tool_call> tags, and provide your final answer within the <answer> </answer> tags."""


# ---------------------------------------------------------------------------
# Prompt templates matching SFT format
# ---------------------------------------------------------------------------

GROUNDING_USER_TEMPLATE = """<image>
You will be performing a visual grounding task. You will be given a JSON array of candidate object proposals detected in an image, each with a bounding box (``bbox_2d`` in 1000x1000 normalized coordinates), label, and confidence score. Your goal is to identify specific objects and their spatial relationships based on the task description provided.

Here are the candidate object proposals:

<candidate_objects>
{proposals_json}
</candidate_objects>

Here is the task you need to complete:

<task_description>
{query}
</task_description>

Your job is to carefully analyze the candidate objects and identify which ones satisfy the relationship or interaction described in the task. Pay close attention to:
- Spatial proximity (are objects close enough to be interacting?)
- Semantic relationships (does the pairing make sense for the described interaction?)
- Object labels and their relevance to the task
- **Objects visible in the image but missing from proposals** — the detector may not have found every relevant person or object. If you see additional interacting pairs in the image, estimate their bounding boxes from visual inspection and include them.

Before providing your final answer, reason through your analysis step by step in a <think> block.

In your <think>, you should:
1. Identify what object types are relevant to the task
2. List out the candidate objects of each relevant type
3. Analyze spatial relationships between candidates (e.g., is object A close to object B? Are their bounding boxes overlapping or adjacent?)
4. Determine which pairs or groups of objects satisfy the interaction or relationship described in the task
5. Check if there are additional interacting pairs visible in the image that are not covered by the proposals
6. Note any edge cases (e.g., objects that are close but likely not interacting, ambiguous cases)

After your reasoning, provide your final answer.

Important formatting guidelines:
- Bounding boxes use 1000x1000 normalized coordinates as [x1, y1, x2, y2]
- Each object should include both its bounding box and label
- **IMPORTANT: Each line must contain EXACTLY 2 objects** — one person and one interacted object. If one person interacts with multiple objects, write each pair on a separate line. If multiple people interact with the same object, write each pair on a separate line.
- If no valid pairs are found, write "no valid pairs found" in the answer.

Example output format for a single pair:
<answer>
[{{"bbox_2d": [x1, y1, x2, y2], "label": "person"}}, {{"bbox_2d": [x1, y1, x2, y2], "label": "object"}}]
</answer>

Example output format for multiple pairs:
<answer>
[{{"bbox_2d": [x1, y1, x2, y2], "label": "person"}}, {{"bbox_2d": [x1, y1, x2, y2], "label": "object"}}]
[{{"bbox_2d": [x1, y1, x2, y2], "label": "person"}}, {{"bbox_2d": [x1, y1, x2, y2], "label": "object"}}]
</answer>

{guideline}"""


REFERRING_USER_TEMPLATE = """<image>
You will be identifying and describing the action that a person is performing with a specific object in an image.

Here are the candidate object proposals detected in the image:

<proposals>
{proposals_json}
</proposals>

The person you need to analyze is located at: **{person_json}**

The object they are interacting with is located at: **{object_json}**

Your task is to describe the action the person is performing with this object. Analyze the spatial relationship between the person and object, their positioning, and any visible interaction patterns to determine what action is taking place.

Note: The proposals may not cover every object in the scene. Use the proposals as context but focus your analysis on the person and object at the specified bounding boxes.

Before providing your final answer, reason about what you observe in a <think> block.

Your response must follow these formatting rules:
- Use the format: "{{verb+ing}} {{object}}" where the verb is in present participle (-ing) form
- Do not include articles (a, an, the)
- Be concise and specific
- Use only the action phrase, nothing else

Examples of correct responses:
- "riding bicycle"
- "holding umbrella"
- "sitting on bench"
- "throwing frisbee"
- "petting dog"

Write your final answer inside <answer> tags. Your answer should contain only the action phrase in the specified format, with no additional explanation or commentary.

{guideline}"""


# ---------------------------------------------------------------------------
# SDS computation (matches reward manager)
# ---------------------------------------------------------------------------

def compute_sds(boxes_1000: list, num_pairs: int) -> float:
    """Compute Spatial Difficulty Score from GT **object** bounding box areas.

    Uses object area only (not min of person/object) because objects are the
    perceptual bottleneck in HOI detection (e.g., baseball 6px², tie 8px²).
    """
    if num_pairs == 0 or not boxes_1000:
        return 0.5

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

    sds = (0.15 - min_obj_area) / 0.14
    return float(np.clip(sds, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Proposal loading
# ---------------------------------------------------------------------------

def load_proposals(proposals_dir: str, file_name: str, dataset_type: str) -> list[dict] | None:
    """Load pre-computed YOLO-world proposals for an image.

    For HICO: proposal file is named like the image file without extension.
    For SWIG: proposal file is named like the image file without extension.
    """
    stem = Path(file_name).stem
    proposal_path = os.path.join(proposals_dir, f"{stem}.json")

    if not os.path.exists(proposal_path):
        return None

    try:
        with open(proposal_path) as f:
            data = json.load(f)
        proposals = data.get("proposals", [])
        # Format for prompt: use bbox_1000 coordinates
        formatted = []
        for idx, p in enumerate(proposals):
            formatted.append({
                "bbox_2d": p.get("bbox_1000", p.get("bbox", [])),
                "label": p.get("class_name", "unknown"),
                "confidence": round(p.get("confidence", 0.0), 2),
                "id": idx,
            })
        return formatted
    except (json.JSONDecodeError, IOError):
        return None


# ---------------------------------------------------------------------------
# Sample processing
# ---------------------------------------------------------------------------

def process_grounding_sample(
    sample: dict,
    img_dir: str,
    proposals_dir: str,
    dataset_type: str,
    guideline: str = GROUNDING_GUIDELINE,
) -> dict | None:
    """Process a single grounding sample into RL format."""
    file_name = sample["file_name"]
    img_path = os.path.join(img_dir, file_name)

    if not os.path.exists(img_path):
        return None

    # Validate response
    response = sample.get("response", "")
    if not response:
        return None

    boxes_1000 = sample.get("boxes_1000", [])
    num_pairs = sample.get("num_pairs", 0)

    # Load proposals
    proposals = load_proposals(proposals_dir, file_name, dataset_type)
    if proposals is None:
        proposals = []

    proposals_json = json.dumps(proposals, indent=2)

    query = sample.get("query", "")
    if not query:
        action = sample.get("action", "")
        obj_cat = sample.get("object_category", "")
        query = (
            f"Locate every person who is **{action} {obj_cat}** and the "
            f"**{obj_cat}** they interact with in this image."
        )

    user_content = GROUNDING_USER_TEMPLATE.format(
        proposals_json=proposals_json,
        query=query,
        guideline=guideline,
    )

    # Pre-compute SDS
    sds = compute_sds(boxes_1000, num_pairs)

    # Ground truth for reward manager
    gt_data = json.dumps({
        "boxes_1000": boxes_1000,
        "num_pairs": num_pairs,
    })

    return {
        "data_source": "hoi_grounding",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "images": [{"image": os.path.abspath(img_path)}],
        "ability": "hoi_grounding",
        "reward_model": {
            "style": "rule",
            "ground_truth": gt_data,
        },
        "extra_info": {
            "task_type": "grounding",
            "spatial_difficulty_score": sds,
            "num_gt_pairs": num_pairs,
            # SAHA v3: proposal-trust anchor for the counterfactual s_ref (grounding only).
            "proposal_anchor": build_proposal_anchor_grounding(proposals, boxes_1000, num_pairs),
            "images": [os.path.abspath(img_path)],
            "is_video": False,
        },
    }


def process_referring_sample(
    sample: dict,
    img_dir: str,
    proposals_dir: str,
    dataset_type: str,
    guideline: str = GROUNDING_GUIDELINE,
) -> dict | None:
    """Process a single referring sample into RL format."""
    file_name = sample["file_name"]
    img_path = os.path.join(img_dir, file_name)

    if not os.path.exists(img_path):
        return None

    response = sample.get("response", "")
    if not response:
        return None

    boxes_1000 = sample.get("boxes_1000", [])
    person_box_idx = sample.get("person_box_idx", 0)
    object_box_idx = sample.get("object_box_idx", 1)

    if person_box_idx >= len(boxes_1000) or object_box_idx >= len(boxes_1000):
        return None

    person_bbox = boxes_1000[person_box_idx]
    object_bbox = boxes_1000[object_box_idx]

    # Load proposals
    proposals = load_proposals(proposals_dir, file_name, dataset_type)
    if proposals is None:
        proposals = []

    proposals_json = json.dumps(proposals, indent=2)

    person_json = json.dumps({"bbox_2d": person_bbox, "label": "person"})
    object_json = json.dumps({"bbox_2d": object_bbox, "label": "object"})

    user_content = REFERRING_USER_TEMPLATE.format(
        proposals_json=proposals_json,
        person_json=person_json,
        object_json=object_json,
        guideline=guideline,
    )

    # Pre-compute SDS (object-area based: object is at index 1 in the pair)
    sds = compute_sds([person_bbox, object_bbox], num_pairs=1)

    # Ground truth
    gt_data = json.dumps({
        "response": response,
        "person_bbox": person_bbox,
        "object_bbox": object_bbox,
    })

    return {
        "data_source": "hoi_referring",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "images": [{"image": os.path.abspath(img_path)}],
        "ability": "hoi_referring",
        "reward_model": {
            "style": "rule",
            "ground_truth": gt_data,
        },
        "extra_info": {
            "task_type": "referring",
            "spatial_difficulty_score": sds,
            "num_gt_pairs": 1,
            "images": [os.path.abspath(img_path)],
            "is_video": False,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_dir: str = "/media/shaun/workspace/hoi/dataset/benchmarks_simplified",
    hico_img_dir: str = "/media/shaun/workspace/hoi/dataset/hico_20160224_det/images/train2015",
    swig_img_dir: str = "/media/shaun/workspace/hoi/dataset/swig_hoi/images_512",
    proposals_dir: str = "/media/shaun/workspace/hoi-dataset-curation/output/proposals",
    local_dir: str = "data/hoi",
    max_pairs: int = 15,
    filter_len: int | None = 8192,
    val_size: int = 500,
    seed: int = 42,
    proposal_action_convention: bool = False,
) -> None:
    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)

    # SAHA v2 Rev 4 (§10.0): optionally append the prompt-level proposal-decision
    # convention. Default False -> guideline is unchanged and generated prompts
    # are byte-identical to prior behavior.
    guideline = GROUNDING_GUIDELINE
    if proposal_action_convention:
        guideline = GROUNDING_GUIDELINE + "\n\n" + _load_convention_text()
        print("Proposal-decision convention ENABLED (appended to task guideline).")

    # Define input files
    input_files = [
        ("hico_ground_train_simplified.json", "grounding", hico_img_dir, "hico"),
        ("swig_ground_train_simplified.json", "grounding", swig_img_dir, "swig"),
        ("hico_referring_train_simplified.json", "referring", hico_img_dir, "hico"),
        ("swig_referring_train_simplified.json", "referring", swig_img_dir, "swig"),
    ]

    all_samples: list[dict] = []
    stats: dict[str, dict[str, int]] = {}

    for filename, task_type, img_dir, dataset_type in input_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"WARNING: {filepath} not found, skipping")
            continue

        with open(filepath) as f:
            raw_data = json.load(f)

        file_stats = {"total": len(raw_data), "skipped_pairs": 0, "skipped_img": 0, "skipped_response": 0, "kept": 0}
        print(f"Processing {filename}: {len(raw_data)} samples")

        for sample in raw_data:
            # Quality check: pair count cap for grounding
            if task_type == "grounding" and sample.get("num_pairs", 0) > max_pairs:
                file_stats["skipped_pairs"] += 1
                continue

            if task_type == "grounding":
                result = process_grounding_sample(sample, img_dir, proposals_dir, dataset_type, guideline)
            else:
                result = process_referring_sample(sample, img_dir, proposals_dir, dataset_type, guideline)

            if result is None:
                file_stats["skipped_img"] += 1
                continue

            all_samples.append(result)
            file_stats["kept"] += 1

        stats[filename] = file_stats
        print(f"  Kept: {file_stats['kept']}, Skipped (pairs): {file_stats['skipped_pairs']}, "
              f"Skipped (img/response): {file_stats['skipped_img']}")

    print(f"\nTotal samples: {len(all_samples)}")

    if not all_samples:
        print("ERROR: No samples produced. Check paths and data files.")
        return

    # Convert to HuggingFace dataset
    dataset = datasets.Dataset.from_list(all_samples)

    # Optional: filter by token length
    if filter_len and filter_len > 0:
        try:
            from transformers import AutoProcessor
            from qwen_vl_utils import process_vision_info
            from collections import defaultdict
            import regex as re

            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

            def get_mm_content_len(example: dict) -> int:
                messages = deepcopy(example["prompt"])
                for message in messages:
                    content = message["content"]
                    content_list = []
                    segments = re.split(r"(<image>)", content)
                    segments = [item for item in segments if item]
                    img_idx = 0
                    for segment in segments:
                        if segment == "<image>":
                            content_list.append({"type": "image", "image": example["images"][img_idx]["image"]})
                            img_idx += 1
                        else:
                            content_list.append({"type": "text", "text": segment})
                    message["content"] = content_list
                raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[raw_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                return inputs.input_ids.shape[1]

            def add_content_len(example: dict) -> dict:
                try:
                    example["extra_info"]["mm_content_len"] = get_mm_content_len(example)
                except Exception as e:
                    print(f"Warning: failed to compute content length: {e}")
                    example["extra_info"]["mm_content_len"] = 0
                return example

            print(f"\nComputing token lengths (filter_len={filter_len})...")
            dataset = dataset.map(add_content_len, num_proc=32)
            before = len(dataset)
            dataset = dataset.filter(
                lambda x: 0 < x["extra_info"]["mm_content_len"] <= filter_len,
                num_proc=8,
            )
            print(f"Filtered {before - len(dataset)}/{before} samples exceeding {filter_len} tokens")
        except ImportError as e:
            print(f"WARNING: Cannot filter by token length (missing dependency: {e}). Skipping filter.")

    # Split train/val
    split = dataset.train_test_split(test_size=min(val_size, len(dataset) // 10), seed=seed)
    train_dataset = split["train"]
    val_dataset = split["test"]

    print(f"\nFinal: {len(train_dataset)} train, {len(val_dataset)} val")

    # Print SDS distribution
    sds_values = [s["extra_info"]["spatial_difficulty_score"] for s in train_dataset]
    print(f"\nSDS distribution (train):")
    for lo, hi in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
        count = sum(1 for v in sds_values if lo <= v < hi)
        print(f"  [{lo:.1f}, {hi:.1f}): {count} ({100*count/len(sds_values):.1f}%)")

    # Print task type distribution
    grounding_count = sum(1 for s in train_dataset if s["extra_info"]["task_type"] == "grounding")
    referring_count = len(train_dataset) - grounding_count
    print(f"\nTask distribution: grounding={grounding_count}, referring={referring_count}")

    # Save
    train_path = local_dir_path / "train.parquet"
    val_path = local_dir_path / "val.parquet"
    train_dataset.to_parquet(str(train_path))
    val_dataset.to_parquet(str(val_path))
    print(f"\nSaved to {train_path} and {val_path}")

    # Print example
    print(f"\nExample training sample:")
    example = train_dataset[0]
    print(f"  data_source: {example['data_source']}")
    print(f"  task_type: {example['extra_info']['task_type']}")
    print(f"  SDS: {example['extra_info']['spatial_difficulty_score']:.3f}")
    print(f"  image: {example['images'][0]['image']}")
    print(f"  prompt (user, first 200 chars): {example['prompt'][1]['content'][:200]}...")


if __name__ == "__main__":
    fire.Fire(main)
