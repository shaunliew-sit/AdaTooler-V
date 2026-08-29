"""In-place migration: add ``extra_info["proposal_anchor"]`` to an existing HOI
RL parquet that predates the SAHA-CF (v3 counterfactual) reward.

Why this exists: SAHA-CF's grounding s_ref = score of the proposal-only answer
(``proposal_anchor``), built by ``proposal_anchor.build_proposal_anchor_grounding``
from (proposals, GT boxes, num_pairs). Older parquets (SDS-GRPO era) lack the
field, and ``saha_cf.compute_sref_grounding`` then silently falls back to
s_ref=0.0 for every grounding row -> a broken reward.

The proposals are already embedded in each grounding prompt (the
``<candidate_objects>[ ... ]</candidate_objects>`` block) and the GT boxes live in
``reward_model.ground_truth``. ``proposal_anchor`` is a pure deterministic function
of those, so reconstruction is *exact* -- byte-identical to re-running
prepare_train.py with the original train proposals (which need not be present on
this machine).

Implementation: rewrite ONLY the ``extra_info`` struct column, appending a
``proposal_anchor`` field. Every other column's Arrow data is carried over
untouched, so image data / prompts cannot drift.

Usage:
    python migrate_add_proposal_anchor.py --parquet data/train.parquet
    python migrate_add_proposal_anchor.py --parquet data/val.parquet
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Load the SAME anchor builder used by prepare_train.py so the reconstruction is
# identical to the canonical preprocessing path.
_pa_path = Path(__file__).resolve().parent / "proposal_anchor.py"
_spec = importlib.util.spec_from_file_location("proposal_anchor", _pa_path)
_pa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pa)
build_proposal_anchor_grounding = _pa.build_proposal_anchor_grounding

# Proposals are embedded between these tags in grounding prompts.
_PROPOSALS_RE = re.compile(r"<candidate_objects>\s*(.*?)\s*</candidate_objects>", re.S)

# Arrow type for the new field: list of [person, object] pairs, each a struct.
PAIR_STRUCT = pa.struct([("bbox_2d", pa.list_(pa.int64())), ("label", pa.string())])
ANCHOR_TYPE = pa.list_(pa.list_(PAIR_STRUCT))


def _user_content(prompt) -> str:
    """Concatenate the user-turn text from a prompt (list of {role, content})."""
    out = []
    for msg in prompt:
        if msg.get("role") == "user":
            out.append(msg.get("content") or "")
    return "\n".join(out)


def _extract_proposals(user_txt: str):
    m = _PROPOSALS_RE.search(user_txt)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def build_anchors(table: pa.Table):
    """Return (anchors:list, stats:dict) aligned to table rows."""
    n = table.num_rows
    ei = pa.concat_arrays(table.column("extra_info").chunks)  # StructArray
    task_types = ei.field("task_type").to_pylist()
    gt_strs = pa.concat_arrays(table.column("reward_model").chunks).field("ground_truth").to_pylist()
    prompts = table.column("prompt").to_pylist()

    anchors = [[] for _ in range(n)]
    stats = {"grounding": 0, "referring": 0, "built": 0, "parse_miss": 0,
             "empty_anchor": 0, "gt_error": 0}

    for i in range(n):
        if task_types[i] != "grounding":
            stats["referring"] += 1
            continue
        stats["grounding"] += 1
        props = _extract_proposals(_user_content(prompts[i]))
        if props is None:
            stats["parse_miss"] += 1
            continue
        try:
            gt = json.loads(gt_strs[i])
            boxes_1000 = gt["boxes_1000"]
            num_pairs = int(gt.get("num_pairs", 0))
        except (json.JSONDecodeError, KeyError, TypeError):
            stats["gt_error"] += 1
            continue
        anchor = build_proposal_anchor_grounding(props, boxes_1000, num_pairs)
        if anchor:
            anchors[i] = anchor
            stats["built"] += 1
        else:
            stats["empty_anchor"] += 1
    return anchors, stats


def migrate(parquet_path: str, backup: bool = True):
    path = Path(parquet_path)
    table = pq.read_table(path)
    if "proposal_anchor" in table.schema.field("extra_info").type.names:
        print(f"[skip] {path} already has extra_info.proposal_anchor")
        return

    anchors, stats = build_anchors(table)
    print(f"[{path.name}] rows={table.num_rows} {stats}")

    # Rebuild extra_info struct: original children + proposal_anchor, same order.
    ei = pa.concat_arrays(table.column("extra_info").chunks)
    names = list(ei.type.names)
    children = [ei.field(name) for name in names]
    children.append(pa.array(anchors, type=ANCHOR_TYPE))
    names.append("proposal_anchor")
    new_ei = pa.StructArray.from_arrays(children, names=names)

    idx = table.schema.get_field_index("extra_info")
    new_field = pa.field("extra_info", new_ei.type)
    table = table.set_column(idx, new_field, new_ei)

    if backup:
        bak = path.with_suffix(path.suffix + ".pre_saha_cf.bak")
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"[backup] {bak}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    print(f"[written] {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="path to the parquet to migrate in place")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    migrate(args.parquet, backup=not args.no_backup)
