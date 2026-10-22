"""Proposal-injected HOI prompts -> detect-tool prompts (plan/saha-detect-tool/PLAN.md §B).

One converter shared by the GRPO parquet migration, the SFT rewrite and eval, so all
three see the same text. It edits the prompts actually stored in the data (exact
template segments), not prepare_train.py's constants, which have drifted from them.

convert_user_prompt(text) -> (new_text, proposals_json): the offline YOLOE proposals
block is removed from the prompt and returned verbatim, so the SFT rewrite can replay
it as the observation of a `detect` call.
"""
import json
import re

DETECT_TOOL_SCHEMA = json.dumps({
    "type": "function",
    "function": {
        "name": "detect",
        "description": (
            "Run an object detector on the full original image. Returns a JSON array of candidate "
            "object proposals, each with a bounding box (bbox_2d in the original image's 1000x1000 "
            "normalized coordinates), a label, a confidence score and an id."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
})
DETECT_CALL = '<tool_call>\n{"name": "detect", "arguments": {}}\n</tool_call>'
DETECT_GUIDELINE = "- You can run an object detector using `detect` to get candidate object boxes for the full image.\n"

_ZOOM_GUIDELINE = "- You can zoom in using `zoom_in` to get a closer look at specific regions.\n"
_PROPOSALS = re.compile(r"<(candidate_objects|proposals)>\n(.*?)\n</\1>", re.S)

_GROUNDING_EDITS = [
    ("You will be performing a visual grounding task. You will be given a JSON array of candidate object "
     "proposals detected in an image, each with a bounding box (``bbox_2d`` in 1000x1000 normalized "
     "coordinates), label, and confidence score. Your goal is to identify specific objects and their spatial "
     "relationships based on the task description provided.\n\nHere are the candidate object proposals:\n\n"
     "<candidate_objects>\n{proposals}\n</candidate_objects>\n\n",
     "You will be performing a visual grounding task. Your goal is to identify specific objects and their "
     "spatial relationships based on the task description provided. You can call the `detect` tool to get a "
     "JSON array of candidate object proposals detected in the image, each with a bounding box (``bbox_2d`` in "
     "1000x1000 normalized coordinates), label, and confidence score.\n\n"),
    ("Your job is to carefully analyze the candidate objects and identify which ones satisfy",
     "Your job is to carefully analyze the image and identify which people and objects satisfy"),
    ("- **Objects visible in the image but missing from proposals** — the detector may not have found every "
     "relevant person or object.",
     "- **Objects the detector misses** — if you call `detect`, it may not find every relevant person or object."),
    ("2. List out the candidate objects of each relevant type",
     "2. Locate the people and objects of each relevant type (call `detect` if candidate boxes would help)"),
    ("3. Analyze spatial relationships between candidates (", "3. Analyze spatial relationships between them ("),
    ("5. Check if there are additional interacting pairs visible in the image that are not covered by the proposals",
     "5. Check if there are additional interacting pairs visible in the image that you have not covered yet"),
]
_REFERRING_EDITS = [
    ("Here are the candidate object proposals detected in the image:\n\n<proposals>\n{proposals}\n</proposals>\n\n", ""),
    ("Note: The proposals may not cover every object in the scene. Use the proposals as context but focus your "
     "analysis on the person and object at the specified bounding boxes.",
     "Note: You can call `detect` to see the other objects in the scene as context, but focus your analysis on "
     "the person and object at the specified bounding boxes."),
]


_ZOOM_BBOX_OLD = ("coordinates for bounding box of the area you want to zoom in. minimum value is 0 and "
                  "maximum value is the width/height of the image.")
_ZOOM_BBOX_NEW = ("coordinates for bounding box of the area you want to zoom in, in the same 1000x1000 "
                  "normalized space as the detected boxes (minimum value 0, maximum value 1000).")


def convert_system_prompt(text: str) -> str:
    """Add the detect function to the <tools> list, after zoom_out. Also fixes the inherited
    zoom_in wording: the tool server reads zoom boxes on the 0-1000 grid (grid1000)."""
    if text.count(_ZOOM_BBOX_OLD) == 1:
        text = text.replace(_ZOOM_BBOX_OLD, _ZOOM_BBOX_NEW)
    lines = text.split("\n")
    idx = [i for i, line in enumerate(lines) if '"name": "zoom_out"' in line]
    if len(idx) != 1:
        raise ValueError("system prompt has no single zoom_out tool line")
    return "\n".join(lines[: idx[0] + 1] + [DETECT_TOOL_SCHEMA] + lines[idx[0] + 1:])


def convert_user_prompt(text: str) -> tuple[str, str]:
    """Return (detect-tool prompt, the removed proposals JSON text)."""
    found = _PROPOSALS.findall(text)
    if len(found) != 1:
        raise ValueError(f"expected one proposals block, found {len(found)}")
    tag, proposals = found[0]
    edits = _GROUNDING_EDITS if tag == "candidate_objects" else _REFERRING_EDITS
    out = text
    for old, new in edits:
        old = old.replace("{proposals}", proposals)
        if out.count(old) != 1:
            raise ValueError(f"template segment not found exactly once: {old[:80]!r}")
        out = out.replace(old, new)
    if out.count(_ZOOM_GUIDELINE) != 1:
        raise ValueError("zoom_in guideline line not found exactly once")
    out = out.replace(_ZOOM_GUIDELINE, DETECT_GUIDELINE + _ZOOM_GUIDELINE)
    leftover = re.search(r"proposal|candidate_objects|<proposals>", out.replace("candidate object proposals detected in the image", ""))
    if leftover:
        raise ValueError(f"proposal reference left in converted prompt near {out[max(0, leftover.start() - 60):leftover.end() + 20]!r}")
    return out, proposals


def detect_observation(proposals: str) -> str:
    """The `detect` observation for an offline proposals block. Must equal
    pixel_reasoner.format_detect_observation for the same detections."""
    return "Here are the detected object proposals:\n<candidate_objects>\n" + proposals + "\n</candidate_objects>"
