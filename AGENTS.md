# Repository Guidelines

## Project Structure & Module Organization

AdaTooler-V is a Python research codebase for adaptive visual tool use and the current SAHA-CF training pipeline. Main implementation lives in `verltool/`: `verl_tool/` contains trainer, rollout, server, and reward-manager code; `eval_service/` contains the OpenAI-compatible tool API; `examples/` contains preprocessing, training, evaluation, and smoke-test scripts. Root docs include `README.md`, `reward_design.md`, `docs/`, `plan/`, and `CLAUDE.md`. `src/verl` and `src/verl-tool` are mirrored snapshots; prefer editing `verltool/` unless explicitly asked.

## Current SAHA-CF Rules

Before code changes, read `CLAUDE.md`, then `plan/saha-detect-tool/PLAN.md` (active direction since 2026-10-21: the YOLOE detector is a tool, not offline prompt-injected proposals). If a referenced path or required field is missing, stop and report it. Active tools are `detect` (no arguments; live YOLOE-26l prompt-free on the full original image in the tool server, no detect-on-crop in v1; boxes in the original image's 0-1000 grid), `zoom_in` and `zoom_out`; do not add `inspect_pose`, a text-prompted detector, detector-free variants, AXPO, or new default GRPO algorithms. Active reward is `R_format * (R_outcome + alpha * R_tool)` with `I_tool` covering all three tools; in detect-tool runs `s_ref` = same-uid no-tool sibling mean for both tasks (proposal-injected baseline runs keep the grounding anchor `s_ref`); do not reintroduce Rev-4 six-term rewards, SDS-v2 reward terms, proposal-action verbalization, or a per-call cost term. Keep `sds_grpo.py`, proposal-injected SAHA-CF, and `AdaTooler-V.py` runnable as baselines; write detect-tool data to new files. Never `pip install ultralytics` with deps into verl-tool-env (pulls numpy 2/opencv 5).

## Build, Test, and Development Commands

```bash
cd verltool
git submodule update --init --recursive
conda create --name verl-tool-env python=3.10
conda activate verl-tool-env
pip install -e verl
pip install -e ".[vllm,acecoder,torl,search_tool]"
pip install "flash-attn==2.8.3" --no-build-isolation
```

Prepare AdaTooler-V data with `python examples/data_preprocess/pixel_reasoner/prepare_train.py ...`. Run original RL with `bash examples/train/AdaTooler-V/train_qwen25vl.sh`; evaluate with `bash examples/train/AdaTooler-V/eval.sh`. For SAHA-CF, run `python examples/train/hoi/smoke_test_saha_cf.py` and inspect rollouts with `python examples/train/hoi/watch_tool_use.py <RUN>`. Start the API service from `verltool/eval_service` with `bash scripts/start_api_service.sh`.

## Coding Style & Naming Conventions

Use Python 3.10+, 4-space indentation, snake_case for functions/modules, PascalCase for classes, and lowercase YAML config keys. Keep Hydra config in `verltool/verl_tool/trainer/config/` and task scripts under `verltool/examples/`. The vendored `verltool/verl` package configures Ruff with a 120-character line length; use that baseline for shared trainer code.

## Testing Guidelines

Tests use `pytest` and `test_*.py` names. Run focused tests first:

```bash
cd verltool
pytest eval_service/test/test_api.py
pytest verl_tool/servers/tests
```

For service tests, update local server addresses as noted in `eval_service/README.md`. For SAHA-CF changes, also run drift checks: `rg -n "inspect_pose|detector-free" verltool/ --type py`, `rg -n "Delta S|Tool Benefit|AT-GRPO|AdaTooler" verltool/ --type py`, and `rg -rn "prepare_sft|trace_gen|generate_traces" --type py verltool/`.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit style, such as `fix(env): ...`, `feat(saha-cf): ...`, and `docs(saha-cf): ...`. Keep commits scoped. Pull requests should explain motivation, summarize changes, list verification commands, link related issues or runbooks, and include screenshots or metrics when training, evaluation, or visual outputs change.

## Security & Configuration Tips

Do not commit secrets, model tokens, or local credentials from `.env`. Large checkpoints, datasets, and generated outputs under `verltool/checkpoints/`, `verltool/data/`, and `verltool/outputs/` should stay out of normal code reviews unless intentionally versioned.

## Herdr

These instructions persist for future agents working in this repository.

- Use Herdr when explicitly requested; its presence alone does not authorize delegation or starting agents.
- Before operating it, check `test "${HERDR_ENV:-}" = 1`. If false, report that this agent is outside Herdr and do not control its sessions.
- Read the official operating instructions before first use: https://raw.githubusercontent.com/herdrdev/herdr/master/skills/herdr/SKILL.md. Setup guide: https://herdr.dev/agent-guide.md.
- Discover installed syntax with `herdr --help` and command groups such as `herdr pane` or `herdr agent`. Never run bare `herdr` inside a pane or probe mutating subcommands.
- Inspect with `herdr agent list` and `herdr pane list --workspace "$HERDR_WORKSPACE_ID"`. Use returned IDs, unique agent names, or `--current`; never assume UI focus identifies the intended target.
- For authorized background work, preserve cwd and focus: `herdr pane split --current --direction right --cwd "$PWD" --no-focus`. Choose `down` for narrow panes. Use pane commands for shell processes and agent commands for recognized agents.
- Read output after waits; settled status alone does not prove completion. Inspect timeouts before retrying input. Leave approval dialogs for the user.
- Do not close existing work, stop servers, change integrations, or install global skills without authorization. Preserve the repository's SAHA-CF rules in every pane.
