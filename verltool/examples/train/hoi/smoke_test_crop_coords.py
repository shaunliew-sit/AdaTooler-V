#!/usr/bin/env python
"""
Smoke test for the zoom_in / crop coordinate-space fix in
verl_tool/servers/tools/pixel_reasoner.py.

Background
----------
The HOI/SAHA-CF pipeline is Qwen3-VL and emits bbox in 0-1000 NORMALIZED coords
(GT field `boxes_1000`, candidate proposals 0-1000, answers scored 0-1000). The
crop tool used to divide those coords by the NATIVE image pixel size, treating
them as absolute pixels (the Qwen2.5-VL / AdaTooler-V convention). On any image
not exactly 1000 px, every zoom landed on the wrong region; coords > img-dim
clamped to the edge, producing mislocated thin sliver crops -> tools were useless
-> the counterfactual reward correctly suppressed tool use -> GRPO tool collapse.

The fix makes crop() convention-aware via `coordinate_scale`, configured per
tool-server process by PIXEL_REASONER_BBOX_MODE ("grid1000" -> divide by 1000;
"pixel"/unset -> original absolute-pixel behaviour).

This test verifies:
  1. grid1000 crops LAND ON the intended region (synthetic colour-marker test).
  2. The old pixel-mode interpretation of 0-1000 coords is wrong/degenerate (regression).
  3. Backward compat: pixel-mode is byte-identical to the original formula.
  4. The all-<1 float branch is unaffected by coordinate_scale.
  5. PIXEL_REASONER_BBOX_MODE parsing (grid1000 / gridN / pixel / unset / garbage).
  6. Real HICO image (if present): grid1000 crop is non-degenerate & centred.
  7. Full async tool path reports a sane (non-sliver) "Image Size".

Run:  python examples/train/hoi/smoke_test_crop_coords.py
"""
import os
import sys
import asyncio

from PIL import Image

from verl_tool.servers.tools.pixel_reasoner import crop, PixelReasonerTool

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ::  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def has_red(img):
    """True if the image contains a clearly-red pixel (False for empty crops)."""
    if min(img.size) == 0:
        return False
    rgb = img.convert("RGB")
    rgb.thumbnail((64, 64))  # cheap scan
    for r, g, b in rgb.getdata():
        if r > 150 and g < 100 and b < 100:
            return True
    return False


def make_marked_image(W, H, cx_norm, cy_norm, half=30, color=(255, 0, 0)):
    """White WxH image with a `color` square centred at the given 0-1000 coords."""
    img = Image.new("RGB", (W, H), (255, 255, 255))
    cx = int(cx_norm / 1000 * W)
    cy = int(cy_norm / 1000 * H)
    for x in range(max(0, cx - half), min(W, cx + half)):
        for y in range(max(0, cy - half), min(H, cy + half)):
            img.putpixel((x, y), color)
    return img


def aspect(sz):
    w, h = sz
    return (w / h) if h else float("inf")


# ---------------------------------------------------------------------------
print("\n=== 1. grid1000 crop lands on the intended region (synthetic) ===")
# Red marker at the CENTRE (500,500). A centred crop must capture it; an off-centre
# crop (e.g. top-left or right-edge) must not.
W, H = 800, 400
center_img = make_marked_image(W, H, 500, 500)
c_center = crop(center_img, [450, 450, 550, 550], coordinate_scale=1000)
check("grid1000 centre-box captures centre marker", has_red(c_center),
      f"crop size={c_center.size}")

# Marker now at top-left (150,150). Centre-box must NOT capture it; a TL box must.
tl_img = make_marked_image(W, H, 150, 150)
c_tl_wrong = crop(tl_img, [450, 450, 550, 550], coordinate_scale=1000)
c_tl_right = crop(tl_img, [50, 50, 300, 300], coordinate_scale=1000)
check("grid1000 centre-box misses top-left marker", not has_red(c_tl_wrong))
check("grid1000 top-left box captures top-left marker", has_red(c_tl_right),
      f"crop size={c_tl_right.size}")

# ---------------------------------------------------------------------------
print("\n=== 2. Old pixel-mode interpretation of 0-1000 coords is wrong (regression) ===")
# Same centred marker; interpreting [450,450,550,550] as PIXELS on an 800x400 image
# sends y to 450/400=1.125 -> clamps -> degenerate/empty crop that misses the marker.
c_buggy = crop(center_img, [450, 450, 550, 550], coordinate_scale=None)
degenerate = (min(c_buggy.size) == 0) or (not has_red(c_buggy))
check("pixel-mode on 0-1000 coords is degenerate/off-target", degenerate,
      f"buggy crop size={c_buggy.size}, has_red={has_red(c_buggy)}")
check("fixed vs buggy crops differ", c_center.size != c_buggy.size,
      f"fixed={c_center.size} buggy={c_buggy.size}")

# ---------------------------------------------------------------------------
print("\n=== 3. Backward compat: pixel-mode == original absolute-pixel formula ===")
# Reproduce the original else-branch math exactly and compare.
def original_pixel_crop(image, bbox_2d, padding=(0.1, 0.1)):
    img_x, img_y = image.size
    padding_tr = (600.0 / img_x, 600.0 / img_y)
    padding = (min(padding[0], padding_tr[0]), min(padding[1], padding_tr[1]))
    nb = (bbox_2d[0] / img_x - padding[0], bbox_2d[1] / img_y - padding[1],
          bbox_2d[2] / img_x + padding[0], bbox_2d[3] / img_y + padding[1])
    nx1, ny1, nx2, ny2 = (min(max(0, v), 1) for v in nb)
    return image.crop((int(nx1 * img_x), int(ny1 * img_y),
                       int(nx2 * img_x), int(ny2 * img_y)))

px_bbox = [120, 60, 520, 300]  # absolute pixels on an 800x400 image
c_new = crop(center_img, px_bbox, coordinate_scale=None)
c_old = original_pixel_crop(center_img, px_bbox)
check("pixel-mode crop size unchanged from original", c_new.size == c_old.size,
      f"new={c_new.size} old={c_old.size}")

# ---------------------------------------------------------------------------
print("\n=== 4. all-<1 float branch is unaffected by coordinate_scale ===")
f_none = crop(center_img, [0.1, 0.1, 0.9, 0.9], coordinate_scale=None)
f_grid = crop(center_img, [0.1, 0.1, 0.9, 0.9], coordinate_scale=1000)
check("float (0.0-1.0) bbox ignores coordinate_scale", f_none.size == f_grid.size,
      f"none={f_none.size} grid1000={f_grid.size}")

# ---------------------------------------------------------------------------
print("\n=== 5. PIXEL_REASONER_BBOX_MODE parsing ===")
def scale_for(mode):
    if mode is None:
        os.environ.pop("PIXEL_REASONER_BBOX_MODE", None)
    else:
        os.environ["PIXEL_REASONER_BBOX_MODE"] = mode
    tool = PixelReasonerTool(num_workers=1)
    return tool.coordinate_scale

check("grid1000 -> 1000", scale_for("grid1000") == 1000.0)
check("grid500  -> 500",  scale_for("grid500") == 500.0)
check("pixel    -> None", scale_for("pixel") is None)
check("unset    -> None", scale_for(None) is None)
check("garbage  -> None", scale_for("banana") is None)
check("grid(bad)-> None", scale_for("gridXYZ") is None)
check("grid0    -> None (no div-by-zero)", scale_for("grid0") is None)
check("grid-5   -> None (no inverted box)", scale_for("grid-5") is None)

# ---------------------------------------------------------------------------
print("\n=== 6. Real HICO image (if present) ===")
hico = "/workspace/data/hico_20160224_det/images/train2015/HICO_train2015_00004173.jpg"
if os.path.exists(hico):
    im = Image.open(hico)
    iw, ih = im.size
    img_aspect = iw / ih
    # Centred square normalized box -> crop aspect should track image aspect, not collapse.
    c_fixed = crop(im, [250, 250, 750, 750], coordinate_scale=1000)
    c_old = crop(im, [250, 250, 750, 750], coordinate_scale=None)
    ar = aspect(c_fixed.size)
    check("HICO grid1000 crop non-degenerate (aspect tracks image)",
          0.5 * img_aspect < ar < 2.0 * img_aspect,
          f"img {iw}x{ih} (ar={img_aspect:.2f}) -> crop {c_fixed.size} (ar={ar:.2f})")
    check("HICO grid1000 differs from buggy pixel-mode", c_fixed.size != c_old.size,
          f"fixed={c_fixed.size} buggy={c_old.size}")
    # The right-centre sliver case from the bug report.
    sliver = crop(im, [590, 108, 1000, 575], coordinate_scale=None)
    fixed2 = crop(im, [590, 108, 1000, 575], coordinate_scale=1000)
    check("buggy right-edge box was a thin sliver, fix widens it",
          fixed2.size[0] > sliver.size[0],
          f"buggy={sliver.size} fixed={fixed2.size}")
else:
    print(f"  [SKIP] {hico} not present on this box")

# ---------------------------------------------------------------------------
print("\n=== 7. Full async tool path reports a sane Image Size ===")
os.environ["PIXEL_REASONER_BBOX_MODE"] = "grid1000"
tool = PixelReasonerTool(num_workers=1)
env = {"images": [center_img]}
obs, valid = asyncio.run(
    tool.conduct_zoom_in_action_async({"bbox_2d": [400, 400, 600, 600], "target_image": 1}, env)
)
ok = valid and isinstance(obs, dict) and "Image Size" in obs.get("obs", "")
size_str = obs.get("obs", "").split("Image Size:")[-1].split(")")[0].strip() if ok else "n/a"
check("async zoom_in returns a valid cropped image", ok, f"Image Size: {size_str}")
if ok:
    w, h = (int(v) for v in size_str.lower().split("x"))
    check("async crop is not a degenerate sliver", 0.2 < (w / h) < 5.0,
          f"{w}x{h} (aspect={w/h:.2f})")

# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILED -> {FAILS}")
    sys.exit(1)
print("RESULT: ALL PASSED")
sys.exit(0)
