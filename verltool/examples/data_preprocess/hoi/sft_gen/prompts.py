"""Prompts for SFT trace generation (plan/saha-sft-regen/PLAN.md).

Design: the model is STEERED at generation time by a `developer` message
(GEN_DEVELOPER_PROMPT) that fixes the behaviors we want (per-turn visible <think>,
trust-good-proposals/anti-override, zoom-discipline). The SAVED SFT trace, however,
uses the *verbatim RL system+user messages* from the parquet — the developer steering
is generation-only and dropped — so SFT inputs match the live RL pipeline exactly.

The developer prompt NEVER contains ground truth (anti-leak, PLAN §6).
"""

# Generation-only steering. NOT saved into the SFT trace.
GEN_DEVELOPER_PROMPT = """You are generating gold demonstration traces for a visual HOI (human-object interaction) tool-use agent. The user message defines the task, the candidate object proposals, and the exact required answer format. Follow these rules EXACTLY on EVERY turn:

1. ALWAYS reason inside a <think>...</think> block in your visible text on EVERY assistant turn — including the final turn that contains <answer>. The <think> must be substantive: what you actually see, which proposals match which object, and why you do or do not zoom. Never emit a tool call or an <answer> without a preceding visible <think> in the SAME message.

2. TRUST GOOD PROPOSALS. If a candidate proposal's bounding box already accurately covers the target object, USE ITS COORDINATES VERBATIM in your answer (you may correct only a wrong label). Do NOT "tighten", nudge, or re-estimate a box that is already correct — needlessly re-estimating a good proposal is the single most common failure and is forbidden. If a zoom CONFIRMS that a proposal was already correct, copy that original proposal box EXACTLY — do not replace it with a crop-derived box.

3. ZOOM ONLY WHEN IT HELPS. Call zoom_in only when the target object is small, occluded, ambiguous, or NOT accurately covered by any proposal. If the proposals already clearly cover the answer, do NOT zoom — go straight to <answer>. When you DO zoom (because a proposal is wrong or missing), read accurate coordinates off what you actually see in the zoomed crop; do not guess. Emit at most ONE tool call per turn.

4. Tool schema: zoom_in requires bbox_2d (0-1000 normalized [x1,y1,x2,y2]) AND target_image (1 = the original image, 2 = the first crop, 3 = the second crop, ...). zoom_out takes no arguments and returns to the original full view.

5. All coordinates are on the 0-1000 normalized grid (the same scale as the proposals). Produce the FINAL ANSWER in the EXACT format demanded by the user message (grounding: each line exactly 2 objects [{person},{object}] as JSON, or the no-pairs sentence; referring: only the bare "{verb}ing {object}" phrase).

6. Your boxes are scored by IoU against a ground truth you are NOT shown. Be accurate and decisive; never hallucinate objects that are not visible. Prefer a correct proposal box over a hand-estimated one."""


def split_rl_prompt(prompt_messages):
    """parquet 'prompt' -> (system_text, user_text). Robust to ndarray/list of dicts."""
    sys_text, user_text = None, None
    for m in list(prompt_messages):
        role = m["role"] if isinstance(m, dict) else m.get("role")
        content = m["content"] if isinstance(m, dict) else m.get("content")
        if role == "system":
            sys_text = content
        elif role == "user":
            user_text = content
    if user_text is None:
        raise ValueError("no user message in RL prompt")
    return sys_text, user_text


# The two tool definitions, in Responses-API function-tool schema (verified by probe).
RESPONSES_TOOLS = [
    {"type": "function", "name": "zoom_in",
     "description": "Zoom in on the image based on the bounding box coordinates.",
     "parameters": {"type": "object", "properties": {
         "bbox_2d": {"type": "array", "items": {"type": "number"},
                      "description": "0-1000 normalized [x1,y1,x2,y2]"},
         "target_image": {"type": "number",
                           "description": "Index from 1 (1 = original image)."}},
         "required": ["bbox_2d", "target_image"]}},
    {"type": "function", "name": "zoom_out",
     "description": "Return to the full original image view after zooming in.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
]

# Exact pixel_reasoner observation strings (must match RL byte-for-byte).
OBS_ZOOM_IN = "<image>\nHere is the zoomed-in view of the requested region."
OBS_ZOOM_OUT = "Returned to the full original image view."
