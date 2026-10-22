"""Write a detect-tool copy of a proposal-injected HOI GRPO parquet (PLAN §B1).

    python examples/data_preprocess/hoi/migrate_to_detect_tool.py data/val.parquet data/train_resampled_moderate.parquet

For each input `X.parquet`, writes `X_detect.parquet` (the input is never modified):
- prompt: system prompt gains the `detect` tool; user prompt loses the offline YOLOE
  proposals (detect_tool_prompt.convert_*).
- extra_info.detect_image: the real path of the row's original image, which the agent
  loop forwards to the tool server so YOLOE sees the original file, not the resized
  JPEG copy. A path only, never GT.
All other columns, and extra_info.proposal_anchor (eval/metric strata only), are kept.
"""
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_tool_prompt import convert_system_prompt, convert_user_prompt  # noqa: E402


def convert_prompt(messages):
    out = []
    for m in messages:
        if m["role"] == "system":
            out.append({**m, "content": convert_system_prompt(m["content"])})
        elif m["role"] == "user":
            out.append({**m, "content": convert_user_prompt(m["content"])[0]})
        else:
            out.append(m)
    return out


def migrate(path):
    out_path = path[: -len(".parquet")] + "_detect.parquet"
    table = pq.read_table(path)
    prompts = [convert_prompt(p) for p in table.column("prompt").to_pylist()]
    detect_images = []
    for imgs in table.column("images").to_pylist():
        if len(imgs) != 1:
            raise ValueError(f"expected one image per row, got {len(imgs)}")
        real = os.path.realpath(imgs[0]["image"])
        if not os.path.isfile(real):
            raise FileNotFoundError(real)
        detect_images.append(real)

    ei = pa.concat_arrays(table.column("extra_info").chunks)
    fields = [ei.type.field(i) for i in range(ei.type.num_fields) if ei.type.field(i).name != "detect_image"]
    children = [ei.field(f.name) for f in fields] + [pa.array(detect_images, pa.string())]
    new_ei = pa.StructArray.from_arrays(children, fields=fields + [pa.field("detect_image", pa.string())])

    table = table.set_column(table.schema.get_field_index("prompt"), table.schema.field("prompt"),
                             pa.array(prompts, table.schema.field("prompt").type))
    table = table.set_column(table.schema.get_field_index("extra_info"), pa.field("extra_info", new_ei.type), new_ei)
    tmp = out_path + ".tmp"
    pq.write_table(table, tmp)
    os.replace(tmp, out_path)
    print(f"{path} -> {out_path}: {table.num_rows} rows")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        migrate(p)
