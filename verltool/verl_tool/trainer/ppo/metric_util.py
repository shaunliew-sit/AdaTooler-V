import torch
import numpy as np
from typing import Any
import torch
from verl.protocol import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics as verl_compute_data_metrics
from verl.trainer.ppo.metric_utils import bootstrap_metric, calc_maj_val
from functools import partial
from collections import defaultdict
def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    result = verl_compute_data_metrics(batch, use_critic)
    
    verl_tool_metrics = batch.non_tensor_batch.get("verl_tool_metrics", [])
    all_keys = []
    for x in verl_tool_metrics:
        all_keys.extend(list(x.keys()))
    all_keys = set(all_keys)
    for key in all_keys:
        values = np.array([float(m[key]) for m in verl_tool_metrics if key in m])
        result[f"verl_tool/{key}/mean"] = values.mean()
        result[f"verl_tool/{key}/max"] = values.max()
        result[f"verl_tool/{key}/min"] = values.min()

    # SAHA-CF per-step training dashboard (only when that reward manager ran).
    # Sequence-level GRPO advantage, split by tool use, is the most direct evidence
    # the tool-collapse is being counteracted: adv(tool) > adv(notool) on hard strata
    # means the policy gradient pushes toward tool use exactly where it helps.
    seq_adv = None
    try:
        adv = batch.batch["advantages"]
        resp_mask = batch.batch["response_mask"].bool()
        seq_adv = ((adv * resp_mask).sum(-1) / resp_mask.sum(-1).clamp(min=1)).detach().float().cpu().numpy()
    except Exception:
        seq_adv = None
    result.update(compute_saha_cf_metrics(batch.non_tensor_batch, seq_adv=seq_adv))
    return result


def compute_saha_cf_metrics(non_tensor_batch: dict, seq_adv=None) -> dict[str, float]:
    """Per-step training metrics for the SAHA-CF counterfactual tool-gain reward.

    Surfaces the tool-use-collapse signal and the runbook winner-selection criteria
    (SAHA-CF-RUNBOOK §sweep / §diagnostics) as ``saha_cf/*`` keys, so they plot per
    training step in WandB rather than only at eval cadence:
      * tool_rate (overall / grounding / referring / hard-low-s_ref / easy-high-s_ref)
        — COLLAPSE = trends to 0; healthy = stays up and concentrated on hard strata.
      * acc, acc_tool, acc_notool, acc_tool_minus_notool — outcome not degraded;
        tools should not underperform no-tool.
      * R_tool_given_tool, tool_gain_raw_(given_tool|hard) — positive tool-gain where
        proposals are weak (the fix working).
      * R_tool_share — tool contribution as a fraction of total reward; watch it stay
        FLAT (Peak-Then-Collapse guard).
      * reward_std — GRPO signal (collapse floor ~0.05); clip_frac — how often the clip binds.

    Reads the per-rollout diagnostics the reward manager wrote into
    ``batch.non_tensor_batch``. Returns {} (never raises) if SAHA-CF wasn't used.
    """
    if "i_tool" not in non_tensor_batch:
        return {}
    try:
        def arr(k):
            return np.asarray(non_tensor_batch[k], dtype=float) if k in non_tensor_batch else None

        i_tool = arr("i_tool")
        is_g, has_sref, s_ref = arr("is_grounding"), arr("has_sref"), arr("s_ref")
        acc, r_tool, r_out, r_fmt, score = arr("accuracy"), arr("r_tool"), arr("r_outcome"), arr("r_format"), arr("score")
        gain, gain_clip = arr("tool_gain_raw"), arr("tool_gain_clipped")
        z_in, z_out = arr("n_zoom_in"), arr("n_zoom_out")

        n = len(i_tool)
        false_mask = np.zeros(n, dtype=bool)
        tool = i_tool > 0.5
        g = (is_g > 0.5) if is_g is not None else false_mask
        gsref = (g & (has_sref > 0.5)) if has_sref is not None else g
        hard = (gsref & (s_ref < 0.5)) if s_ref is not None else false_mask   # weak proposal -> tool should help
        easy = (gsref & (s_ref >= 0.5)) if s_ref is not None else false_mask  # strong proposal -> tool wasteful

        def mean(a, mask=None):
            if a is None:
                return float("nan")
            a = a if mask is None else a[mask]
            return float(a.mean()) if a.size else float("nan")

        m = {
            "tool_rate": mean(i_tool),
            "tool_rate_grounding": mean(i_tool, g),
            "tool_rate_referring": mean(i_tool, ~g),
            "tool_rate_hard": mean(i_tool, hard),
            "tool_rate_easy": mean(i_tool, easy),
            "acc": mean(acc),
            "acc_tool": mean(acc, tool),
            "acc_notool": mean(acc, ~tool),
            "acc_tool_minus_notool": mean(acc, tool) - mean(acc, ~tool),
            "r_outcome": mean(r_out),
            "r_format": mean(r_fmt),
            "R_tool_mean": mean(r_tool),
            "R_tool_given_tool": mean(r_tool, tool),
            "tool_gain_raw_given_tool": mean(gain, tool),
            "tool_gain_raw_hard_tool": mean(gain, hard & tool),
            "s_ref_grounding": mean(s_ref, gsref),
            "n_zoom_in": mean(z_in),
            "n_zoom_out": mean(z_out),
            "reward_std": float(np.std(score)) if score is not None else float("nan"),
        }
        # GRPO advantage split by tool use — the anti-collapse signal. Positive
        # adv_tool_minus_notool (esp. on hard strata) = gradient favors useful tools.
        if seq_adv is not None and len(seq_adv) == n:
            sa = np.asarray(seq_adv, dtype=float)
            m["adv_tool"] = mean(sa, tool)
            m["adv_notool"] = mean(sa, ~tool)
            m["adv_tool_minus_notool"] = mean(sa, tool) - mean(sa, ~tool)
            m["adv_tool_hard"] = mean(sa, hard & tool)
            m["adv_notool_hard"] = mean(sa, hard & ~tool)

            # --- TRUE within-GRPO-group contrast (the group-sampling proof) ---
            # For each prompt's sampled group (shared uid, n rollouts), among groups
            # that contain BOTH tool and no-tool rollouts, does the tool side win the
            # zero-mean advantage tournament against its OWN siblings? This is the
            # direct GRPO tool-collapse measurement (collapse = tool side loses/ties).
            uid = non_tensor_batch.get("uid")
            if uid is not None and len(uid) == n:
                uids = np.asarray(uid)
                gaps, wins, gaps_h, wins_h = [], [], [], []
                for u in set(uids.tolist()):
                    gm = uids == u
                    gt, gn = gm & tool, gm & ~tool
                    if gt.any() and gn.any():
                        gap = float(sa[gt].mean() - sa[gn].mean())
                        gaps.append(gap); wins.append(1.0 if gap > 0 else 0.0)
                        if (gm & hard).any():  # grounding group with weak proposal
                            gaps_h.append(gap); wins_h.append(1.0 if gap > 0 else 0.0)
                if gaps:
                    m["group_adv_gap"] = float(np.mean(gaps))
                    m["group_tool_wins_frac"] = float(np.mean(wins))
                    m["n_mixed_groups"] = float(len(gaps))
                if gaps_h:
                    m["group_adv_gap_hard"] = float(np.mean(gaps_h))
                    m["group_tool_wins_frac_hard"] = float(np.mean(wins_h))
                    m["n_mixed_groups_hard"] = float(len(gaps_h))
        # R_tool share of total reward: tool contribution to R_total is
        # score - r_format*r_outcome = r_format*alpha*r_tool (captures alpha implicitly).
        if score is not None and r_out is not None and r_fmt is not None:
            tot = float(np.sum(score))
            m["R_tool_share"] = float(np.sum(score - r_fmt * r_out) / tot) if abs(tot) > 1e-8 else float("nan")
        # how often the asymmetric clip binds among tool rollouts
        if gain is not None and gain_clip is not None and tool.any():
            m["clip_frac"] = float((np.abs(gain[tool] - gain_clip[tool]) > 1e-9).mean())

        return {f"saha_cf/{k}": v for k, v in m.items()}
    except Exception as e:  # metrics must never break training
        print(f"[SAHA-CF metrics] skipped: {e}", flush=True)
        return {}

def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2uid2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                    
                var_vals = [x for x in var_vals if x is not None]
                if not var_vals:
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"], strict=True)
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2uid2var2metric[data_source][uid][var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)

    return data_src2var2metric2val
