"""De-risk probe for the SFT regen pipeline (plan/saha-sft-regen/PLAN.md §5).

Confirms, with a handful of cheap synchronous calls, the decisions codex flagged RISKY:
  (2) Does gpt-5.5 accept `reasoning_effort` + tools + image input in CHAT COMPLETIONS?
      (if not -> we must use the Responses API and pass back encrypted reasoning items)
  (3) Does it emit a VISIBLE <think> block in content alongside a tool call?
  (g) Multi-turn: return a crop as a follow-up user image message (role:"tool" is text-only)
  (6) Prints token usage so we can calibrate cost before the 100-sample pilot.

Run: python examples/data_preprocess/hoi/sft_gen/probe_openai.py
Reads OPENAI_API_KEY from /workspace/AdaTooler-V/.env. Spends ~a few cents.
"""
import base64
import json
import sys
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

ENV = dotenv_values("/workspace/AdaTooler-V/.env")
API_KEY = ENV.get("OPENAI_API_KEY")
MODEL = ENV.get("OPENAI_MODEL") or "gpt-5.5"
if not API_KEY:
    sys.exit("OPENAI_API_KEY missing in .env")
client = OpenAI(api_key=API_KEY, base_url=(ENV.get("OPENAI_BASE_URL") or None))

IMG = "/workspace/data/hico_20160224_det/images/train2015/HICO_train2015_00000001.jpg"


def data_url(path):
    b = Path(path).read_bytes()
    return f"data:image/jpeg;base64,{base64.b64encode(b).decode()}"


TOOLS = [
    {"type": "function", "function": {
        "name": "zoom_in",
        "description": "Zoom in on the image based on the bounding box coordinates.",
        "parameters": {"type": "object", "properties": {
            "bbox_2d": {"type": "array", "description": "0-1000 normalized [x1,y1,x2,y2]",
                         "items": {"type": "number"}},
            "target_image": {"type": "number", "description": "Index from 1. Use 1 for original."}},
            "required": ["bbox_2d", "target_image"]}}},
    {"type": "function", "function": {
        "name": "zoom_out",
        "description": "Return to the full original image view after zooming in.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

SYS = ("You are a visual grounding assistant. Reason step by step INSIDE a <think>...</think> block "
       "in your text output BEFORE any tool call or final answer. You may call zoom_in/zoom_out to "
       "inspect regions. When done, give the answer inside <answer>...</answer>.")
USER = [
    {"type": "text", "text": "Locate the person and the object they interact with. "
                              "First zoom in on the main subject to inspect it, then answer. "
                              "Coordinates are 0-1000 normalized."},
    {"type": "image_url", "image_url": {"url": data_url(IMG)}},
]


def show(tag, resp):
    m = resp.choices[0].message
    tcs = m.tool_calls or []
    content = m.content or ""
    print(f"\n--- {tag} ---")
    print(f"  finish_reason: {resp.choices[0].finish_reason}")
    print(f"  has <think> in content: {'<think>' in content}  content_len={len(content)}")
    print(f"  content[:300]: {content[:300]!r}")
    print(f"  tool_calls: {[(t.function.name, t.function.arguments) for t in tcs]}")
    u = resp.usage
    print(f"  usage: prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens}")
    return m


def main():
    print(f"model={MODEL}  sdk=openai")
    # (A) Chat Completions: reasoning_effort + tools + image
    try:
        r1 = client.chat.completions.create(
            model=MODEL, reasoning_effort="high", tools=TOOLS, tool_choice="auto",
            messages=[{"role": "system", "content": SYS},
                      {"role": "user", "content": USER}],
        )
        print("\n[VERDICT 2] Chat Completions ACCEPTS reasoning_effort + tools + image: YES")
        m1 = show("A: chat.completions round 1", r1)
    except Exception as e:
        print(f"\n[VERDICT 2] Chat Completions reasoning_effort FAILED: {type(e).__name__}: {e}")
        print("  -> fall back to Responses API in the harness.")
        # try without reasoning_effort to see if it's just that param
        try:
            r1b = client.chat.completions.create(
                model=MODEL, tools=TOOLS, tool_choice="auto",
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": USER}])
            print("  (tools+image work WITHOUT reasoning_effort -> only the reasoning param is unsupported here)")
            show("A': chat.completions no-effort", r1b)
        except Exception as e2:
            print(f"  also failed without reasoning_effort: {type(e2).__name__}: {e2}")
        return

    # (g) Multi-turn: feed a (fake) crop back as a follow-up USER image message
    tcs = m1.tool_calls or []
    if not tcs:
        print("\n[VERDICT 3] model did NOT call a tool on round 1 (may need stronger prompt). "
              "has_think=", "<think>" in (m1.content or ""))
        return
    tc = tcs[0]
    assistant_msg = {"role": "assistant", "content": m1.content or "",
                     "tool_calls": [{"id": tc.id, "type": "function",
                                     "function": {"name": tc.function.name,
                                                  "arguments": tc.function.arguments}}]}
    tool_ack = {"role": "tool", "tool_call_id": tc.id,
                "content": "Here is the zoomed-in view of the requested region."}
    crop_user = {"role": "user", "content": [
        {"type": "text", "text": "<image>\nHere is the zoomed-in view of the requested region."},
        {"type": "image_url", "image_url": {"url": data_url(IMG), "detail": "low"}}]}
    try:
        r2 = client.chat.completions.create(
            model=MODEL, reasoning_effort="high", tools=TOOLS, tool_choice="auto",
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": USER},
                      assistant_msg, tool_ack, crop_user])
        print("\n[VERDICT g] multi-turn (assistant tool_call -> tool ack -> user crop image) WORKS: YES")
        show("B: chat.completions round 2 (after crop)", r2)
    except Exception as e:
        print(f"\n[VERDICT g] multi-turn flow FAILED: {type(e).__name__}: {e}")

    print("\nDONE. If verdicts 2/3/g are all YES -> build the pilot harness on Chat Completions.")


if __name__ == "__main__":
    main()
