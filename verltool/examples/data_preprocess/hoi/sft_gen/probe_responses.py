"""Responses-API probe — the surface we must use (Chat Completions rejects
reasoning_effort+tools for gpt-5.5). Confirms before building the harness:
  - reasoning={"effort":"high"} + function tools + image input works
  - the model emits a VISIBLE <think> block in output_text (for the SFT trace)
  - 2-round multi-turn with reasoning-item PASSBACK + crop returned as input_image
  - token usage incl. reasoning tokens (cost calibration)

Run: python examples/data_preprocess/hoi/sft_gen/probe_responses.py   (~a few cents)
"""
import base64
import sys
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

ENV = dotenv_values("/workspace/AdaTooler-V/.env")
client = OpenAI(api_key=ENV["OPENAI_API_KEY"], base_url=(ENV.get("OPENAI_BASE_URL") or None))
MODEL = ENV.get("OPENAI_MODEL") or "gpt-5.5"
IMG = "/workspace/data/hico_20160224_det/images/train2015/HICO_train2015_00000001.jpg"


def data_url(p):
    return f"data:image/jpeg;base64,{base64.b64encode(Path(p).read_bytes()).decode()}"


# Responses API function-tool schema: name/description/parameters at TOP level (no "function" wrapper)
TOOLS = [
    {"type": "function", "name": "zoom_in",
     "description": "Zoom in on the image based on the bounding box coordinates.",
     "parameters": {"type": "object", "properties": {
         "bbox_2d": {"type": "array", "items": {"type": "number"},
                      "description": "0-1000 normalized [x1,y1,x2,y2]"},
         "target_image": {"type": "number", "description": "Index from 1; use 1 for original."}},
         "required": ["bbox_2d", "target_image"]}},
    {"type": "function", "name": "zoom_out",
     "description": "Return to the full original image view after zooming in.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
]
SYS = ("You are a visual grounding assistant. Reason step by step INSIDE a <think>...</think> block "
       "in your visible text BEFORE any tool call or final answer. Use zoom_in/zoom_out to inspect. "
       "Give the final answer inside <answer>...</answer>.")
USER = [{"type": "input_text", "text": "Locate the person and the object they interact with. "
         "Zoom in on the main subject first, then answer. Coordinates 0-1000 normalized."},
        {"type": "input_image", "image_url": data_url(IMG)}]


def dump(tag, resp):
    print(f"\n--- {tag} ---")
    kinds = [it.type for it in resp.output]
    print(f"  output item types: {kinds}")
    vis = resp.output_text or ""
    print(f"  has <think> in output_text: {'<think>' in vis}  output_text_len={len(vis)}")
    print(f"  output_text[:320]: {vis[:320]!r}")
    fcs = [it for it in resp.output if it.type == "function_call"]
    for fc in fcs:
        print(f"  function_call: name={fc.name} args={fc.arguments} call_id={fc.call_id}")
    u = resp.usage
    rt = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", None)
    print(f"  usage: input={u.input_tokens} output={u.output_tokens} reasoning={rt} total={u.total_tokens}")
    return fcs


def main():
    print(f"model={MODEL}  surface=/v1/responses")
    try:
        r1 = client.responses.create(
            model=MODEL, reasoning={"effort": "high"}, tools=TOOLS, store=True,
            input=[{"role": "system", "content": SYS}, {"role": "user", "content": USER}])
    except Exception as e:
        print(f"[FAIL] responses round1: {type(e).__name__}: {e}")
        sys.exit(1)
    print("[VERDICT] Responses API accepts reasoning=high + tools + image: YES")
    fcs = dump("round 1", r1)
    if not fcs:
        print("\n[note] no tool call on round 1 — check <think> presence above; may need prompt tweak.")
        return

    # Round 2 via previous_response_id (store=True): server keeps full prior context (incl. images),
    # so we send ONLY the new items — function_call_output (text ack) + the crop as a user image.
    fc = fcs[0]
    next_input = [{"type": "function_call_output", "call_id": fc.call_id,
                   "output": "Here is the zoomed-in view of the requested region."},
                  {"role": "user", "content": [
                      {"type": "input_text", "text": "<image>\nHere is the zoomed-in view of the requested region."},
                      {"type": "input_image", "image_url": data_url(IMG), "detail": "low"}]}]
    try:
        r2 = client.responses.create(model=MODEL, reasoning={"effort": "high"}, tools=TOOLS,
                                     store=True, previous_response_id=r1.id, input=next_input)
        print("\n[VERDICT] multi-turn via previous_response_id (only new crop sent): YES")
        print(f"  (round-2 input_tokens should be SMALL if server reused context: see usage)")
        dump("round 2 (after crop)", r2)
    except Exception as e:
        print(f"\n[FAIL] previous_response_id chaining: {type(e).__name__}: {e}")
    print("\nDONE. If both verdicts YES and <think> is present+rich -> build harness on Responses API.")


if __name__ == "__main__":
    main()
