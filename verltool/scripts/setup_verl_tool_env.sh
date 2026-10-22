#!/bin/bash
# Rebuild the verl-tool conda env (HOI / SAHA GRPO pipeline) from scratch.
#
# The env lives on /workspace (persistent volume), NOT /opt/conda/envs (container
# overlay), so it survives the container being removed. Activate with:
#   conda activate /workspace/envs/verl-tool-env
#
# Versions come from verl_tool_env.lock.txt (a pip freeze of the verified env:
# torch 2.8.0+cu128, vllm 0.11.0, transformers 4.57.6, numpy 1.26.4), installed
# BEFORE the editable packages so nothing unpinned is ever resolved. flash-attn
# is the prebuilt 2.8.3 wheel for cu12/torch2.8/cp310, pinned by sha256.
set -euo pipefail

ENV_PREFIX=${ENV_PREFIX:-/workspace/envs/verl-tool-env}
REPO=/workspace/AdaTooler-V/verltool
LOCK="$REPO/scripts/verl_tool_env.lock.txt"
FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl#sha256=75c51f34bb93c5a4438d6b767547ca6c22a9bfc6b29b5646dda98e943cd75d99"

# The HOI parquets store absolute image paths from the machine that built them
# (/media/shaun/workspace/hoi/dataset/...). The bridge symlink lives on the
# container overlay, so it is lost with the container too. Done first so it
# exists even if the package install fails.
LINK=/media/shaun/workspace/hoi/dataset
if [ -e "$LINK" ] && [ ! -L "$LINK" ]; then
    echo "FATAL: $LINK exists and is not a symlink; refusing to touch it" >&2
    exit 1
fi
mkdir -p "$(dirname "$LINK")"
ln -sfn /workspace/data "$LINK"

source /opt/conda/etc/profile.d/conda.sh
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    # conda-forge only: the default anaconda channels need a manual ToS accept.
    conda create -y -p "$ENV_PREFIX" --override-channels -c conda-forge python=3.10
fi
conda activate "$ENV_PREFIX"
python -m pip install pip==26.2.2 setuptools==84.2.1 wheel==0.48.0

cd "$REPO"
git submodule update --init --recursive verl
pip install -r "$LOCK"
pip install --no-deps "flash-attn @ $FA_WHEEL"
# No build isolation: build with the pinned setuptools/wheel above instead of
# letting pip fetch unpinned build deps.
pip install --no-deps --no-build-isolation -e verl -e .
# YOLOE detect tool. --no-deps: ultralytics' own requirements pull numpy 2 and
# opencv-python 5 (breaks verl); its other deps are pinned in the lockfile, and
# opencv-python-headless already provides cv2.
pip install --no-deps ultralytics==8.4.183
# The one expected complaint is ultralytics wanting opencv-python (see above).
PIP_STATUS=0
PIP_CHECK=$(pip check 2>&1) || PIP_STATUS=$?
# 0 = clean, 1 = broken requirements listed; anything else means pip itself failed.
if [ "$PIP_STATUS" -gt 1 ]; then
    echo "$PIP_CHECK" >&2; exit 1
fi
UNEXPECTED=$(grep -v -e "^ultralytics 8.4.183 requires opencv-python" \
    -e "^No broken requirements found" <<<"$PIP_CHECK" || true)
if [ -n "$UNEXPECTED" ]; then
    echo "$PIP_CHECK" >&2; exit 1
fi

python - <<'EOF'
import torch, vllm, transformers, flash_attn, numpy, verl, verl_tool, ultralytics
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("vllm", vllm.__version__, "| transformers", transformers.__version__,
      "| flash_attn", flash_attn.__version__, "| numpy", numpy.__version__)
EOF
echo "SETUP_DONE"
