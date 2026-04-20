#!/usr/bin/env python
"""Parse AWH pipeline logs and render a static progress report.

When matplotlib is available, figures are emitted as PNG. Otherwise the script
falls back to dependency-free SVG output.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


INFO_PREFIX = "\u250c Info: "
SOURCE_PREFIX = "\u2514 @ "
LEG_COLORS = {
    "solvent": "#c2552d",
    "vacuum": "#2d6db6",
}
METRIC_COLORS = {
    "df": "#3a7d44",
    "ess": "#26547c",
    "lin_neff": "#7a3e9d",
    "round_trips": "#d17b0f",
    "endpoint_band": "#006d77",
    "endpoint_low": "#8d99ae",
    "tail_sum": "#ff7f11",
    "tail_min": "#9c6644",
    "occ_min": "#457b9d",
    "split_gap": "#4a7c59",
    "parity_gap": "#bf4342",
    "supported_parity_gap": "#7b2cbf",
    "endpoint_parity_gap": "#ff7f11",
    "support_fraction": "#118ab2",
    "frames": "#5c677d",
    "probe_ns": "#9c89b8",
    "residual": "#bf4342",
    "accepted_residual": "#2a9d8f",
    "objective": "#bf4342",
    "accepted_objective": "#2a9d8f",
    "objective_threshold": "#6c757d",
    "prediction": "#3a86ff",
    "target": "#6c757d",
    "grad_norm": "#2d6a4f",
    "grad_max": "#d17b0f",
    "kl_est": "#7b2cbf",
    "kl_scaling": "#3a86ff",
    "line_search_alpha": "#ef476f",
    "max_phi_step": "#264653",
    "fim_cond": "#8d99ae",
    "truncated_eigs": "#ff7f11",
    "dG": "#4c956c",
    "ESS": "#1d3557",
    "N_active": "#9c6644",
}
TARGET_PALETTE = [
    "#3a86ff",
    "#ef476f",
    "#118ab2",
    "#6a4c93",
    "#2a9d8f",
    "#e76f51",
    "#ffb703",
    "#577590",
]
PHASE_COLORS = {
    "Initial Rewarm": "#b8b8c6",
    "Stage A Block": "#4c78a8",
    "Stage B Probe MD": "#f58518",
}
FAILURE_COLORS = {
    "passed": "#2a9d8f",
    "endpoint_parity": "#e63946",
    "supported_parity": "#f4a261",
    "raw_parity": "#f4a261",
    "split": "#7b2cbf",
    "low_support": "#577590",
    "passed_raw_only": "#e9c46a",
    "not_checked": "#9aa0a6",
}

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / "awhgrads_mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def to_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def to_float(value: str) -> float:
    text = value.strip().rstrip(".,;")
    lower = text.lower()
    if lower in {"nan", "+nan", "-nan"}:
        return float("nan")
    if lower in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if lower in {"-inf", "-infinity"}:
        return float("-inf")
    return float(text)


def to_int(value: str) -> int:
    return int(value.strip())


def split_message(line: str) -> Optional[str]:
    stripped = line.rstrip("\n")
    if not stripped:
        return None
    if stripped.startswith(INFO_PREFIX):
        return stripped[len(INFO_PREFIX) :]
    if stripped.startswith(SOURCE_PREFIX):
        return None
    return stripped


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    slug = slug.strip("._")
    return slug or "report"


def finite_values(values: Iterable[Optional[float]]) -> List[float]:
    result: List[float] = []
    for value in values:
        if value is None:
            continue
        if math.isfinite(value):
            result.append(value)
    return result


def parse_status_token(token: str) -> Tuple[str, str]:
    leg, status = token.split("=", 1)
    return leg.strip(), status.strip()


def parse_spent_token(token: str) -> Tuple[str, float, float]:
    leg, spent = token.split("=", 1)
    used, budget = spent.split("/", 1)
    return leg.strip(), to_float(used), to_float(budget)


def parse_probe_ess_map(text: str) -> Dict[str, float]:
    data: Dict[str, float] = {}
    for chunk in text.split("|"):
        item = chunk.strip()
        if not item or "=" not in item:
            continue
        leg, value = item.split("=", 1)
        data[leg.strip()] = to_float(value)
    return data


def parse_named_value_map(text: str) -> Dict[str, float]:
    data: Dict[str, float] = {}
    for chunk in text.split("|"):
        item = chunk.strip()
        if not item or "=" not in item:
            continue
        name, value = item.split("=", 1)
        data[name.strip()] = to_float(value)
    return data


def parse_pool_drift_map(text: str) -> Dict[str, Dict[str, float]]:
    data: Dict[str, Dict[str, float]] = {}
    for chunk in text.split("|"):
        item = chunk.strip()
        if not item or "=" not in item:
            continue
        pool, payload = item.split("=", 1)
        metrics: Dict[str, float] = {}
        for label, value in re.findall(r"([^\d\-+.eE]+)([-\deE+.]+)", payload):
            metrics[label.strip(" /")] = to_float(value)
        data[pool.strip()] = metrics
    return data


def parse_pipe_fields(text: str, *, lowercase_keys: bool = False) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for chunk in text.split("|"):
        item = chunk.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if lowercase_keys:
            key = key.lower()
        fields[key] = value.strip()
    return fields


def parse_gate_fields(text: str) -> Dict[str, object]:
    fields: Dict[str, object] = {}
    for chunk in text.split("|"):
        item = chunk.strip()
        if not item:
            continue
        if "<=" in item:
            lhs, rhs = item.split("<=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            if "=" in lhs:
                key, value = lhs.split("=", 1)
                key = key.strip()
                fields[key] = to_bool(value)
                fields[f"{key}_threshold"] = to_float(rhs)
            else:
                fields["threshold"] = to_float(rhs)
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key.strip()] = to_bool(value)
    return fields


def split_numeric_suffix(text: str) -> Tuple[Optional[float], str]:
    match = re.match(r"\s*([-\deE+.]+)(?:\s+(.*?))?\s*$", text)
    if not match:
        return None, ""
    value = to_float(match.group(1))
    suffix = (match.group(2) or "").strip()
    return value, suffix


def parse_index_list(raw: str) -> List[int]:
    text = raw.strip()
    if not text or text == "-":
        return []
    result: List[int] = []
    for part in text.split(","):
        part = part.strip()
        match = re.search(r"(\d+)", part)
        if match:
            result.append(int(match.group(1)))
    return result


def parse_optional_float(raw: str) -> Optional[float]:
    text = raw.strip()
    if not text or text == "-":
        return None
    try:
        return to_float(text)
    except ValueError:
        return None


@dataclass
class MacroStart:
    macro_index: int
    leg: str
    line_no: int
    warm_start: Optional[bool] = None
    bias_reused: Optional[bool] = None
    lambda_prev: Optional[int] = None
    lambda_new: Optional[int] = None
    idx_match: Optional[bool] = None
    restart_rmsd_nm: Optional[float] = None
    volume_ratio: Optional[float] = None
    raw: str = ""


@dataclass
class PhaseEvent:
    macro_index: int
    leg: str
    line_no: int
    kind: str
    phase: str
    md_steps: Optional[int] = None
    md_ns: Optional[float] = None
    wall_s: Optional[float] = None
    steps_per_s: Optional[float] = None
    global_ns_after: Optional[float] = None


@dataclass
class StageASnapshot:
    macro_index: int
    leg: str
    line_no: int
    global_ns: Optional[float]
    macro_spent_ns: Optional[float]
    budget_ns: Optional[float]
    leg_status: Optional[str]
    df: Optional[float] = None
    df_ok: Optional[bool] = None
    ess: Optional[float] = None
    ess_ok: Optional[bool] = None
    lin_neff: Optional[float] = None
    lin_neff_ok: Optional[bool] = None
    tau_int_est: Optional[float] = None
    switches: Optional[int] = None
    dwell_mean: Optional[float] = None
    dwell_median: Optional[float] = None
    round_trips: Optional[int] = None
    round_trips_ok: Optional[bool] = None
    endpoint_low: Optional[float] = None
    endpoint_band: Optional[float] = None
    endpoint_required: Optional[float] = None
    endpoint_ok: Optional[bool] = None
    tail_sum: Optional[float] = None
    tail_min: Optional[float] = None
    tail_ok: Optional[bool] = None
    tail_low_states: List[int] = field(default_factory=list)
    occ_min: Optional[float] = None
    low_occ_states: List[int] = field(default_factory=list)
    rollback: Optional[str] = None
    occ_floor: Optional[float] = None
    frac_above_floor: Optional[float] = None
    n_hist_recent: Optional[int] = None
    n_hist_total: Optional[int] = None
    streak: Optional[int] = None
    cooldown: Optional[int] = None
    failures: Optional[int] = None
    stage_ready: Optional[bool] = None


@dataclass
class StageBSnapshot:
    macro_index: int
    leg: str
    line_no: int
    global_ns: Optional[float]
    macro_spent_ns: Optional[float]
    budget_ns: Optional[float]
    leg_status: Optional[str]
    split_gap: Optional[float] = None
    split_ok: Optional[bool] = None
    parity_gap: Optional[float] = None
    parity_ok: Optional[bool] = None
    raw_parity_gap: Optional[float] = None
    supported_parity_gap: Optional[float] = None
    endpoint_parity_gap: Optional[float] = None
    endpoint_ok: Optional[bool] = None
    n_supported_states: Optional[int] = None
    n_states: Optional[int] = None
    required_supported_states: Optional[int] = None
    support_ok: Optional[bool] = None
    support_threshold: Optional[float] = None
    failure_mode: Optional[str] = None
    near_pass: Optional[bool] = None
    accumulation_mode: Optional[str] = None
    frames: Optional[int] = None
    accumulated_frames: Optional[int] = None
    probe_segments: Optional[int] = None
    diagnostics: str = ""
    fresh_probe_result: bool = False
    attempt_index: Optional[int] = None


@dataclass
class ControlEvent:
    macro_index: int
    leg: str
    line_no: int
    global_ns: Optional[float]
    event_type: str
    details: Dict[str, object] = field(default_factory=dict)


@dataclass
class ProductionArtifact:
    macro_index: int
    leg: str
    line_no: int
    frames: int
    lambda_states: int
    eval_threads: int
    lambda_tile: int
    eval_schedule: str


@dataclass
class OptimizationEpoch:
    macro_index: int
    global_epoch_index: int
    line_no: int
    block_name: str
    epoch_in_macro: Optional[int] = None
    step_clipped_scaling: Optional[float] = None
    prediction: Optional[float] = None
    target: Optional[float] = None
    residual: Optional[float] = None
    accepted_residual: Optional[float] = None
    objective_value: Optional[float] = None
    accepted_objective: Optional[float] = None
    objective_threshold: Optional[float] = None
    objective_in_band: Optional[bool] = None
    objective_step_accepted: Optional[bool] = None
    extra_residual_requirement: Optional[float] = None
    huber_dlde: Optional[float] = None
    grad_norm: Optional[float] = None
    grad_max: Optional[float] = None
    fim_raw_cond: Optional[float] = None
    truncated_eigs: Optional[int] = None
    fim_rank: Optional[int] = None
    health_score: Optional[float] = None
    health_scale: Optional[float] = None
    confidence_scale: Optional[float] = None
    confidence_endpoint_disagreement: Optional[float] = None
    confidence_cycle_disagreement: Optional[float] = None
    confidence_gradient_disagreement: Optional[float] = None
    confidence_eligible_legs: Optional[int] = None
    confidence_skipped_legs: List[str] = field(default_factory=list)
    confidence_prediction_disagreement: Optional[float] = None
    confidence_objective_disagreement: Optional[float] = None
    confidence_eligible_targets: Optional[int] = None
    confidence_skipped_targets: List[str] = field(default_factory=list)
    kl_est: Optional[float] = None
    kl_target: Optional[float] = None
    kl_scaling: Optional[float] = None
    line_search_alpha: Optional[float] = None
    actual_max_phi_step: Optional[float] = None
    max_solute_phi_step: Optional[float] = None
    params_min: Optional[float] = None
    params_max: Optional[float] = None
    line_search_iters: List[Dict[str, object]] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)
    leg_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    target_metrics: Dict[str, Dict[str, object]] = field(default_factory=dict)
    line_search_converged: bool = False


@dataclass
class RunRecord:
    log_path: str
    stage_a: List[StageASnapshot] = field(default_factory=list)
    stage_b: List[StageBSnapshot] = field(default_factory=list)
    phase_events: List[PhaseEvent] = field(default_factory=list)
    macro_starts: List[MacroStart] = field(default_factory=list)
    control_events: List[ControlEvent] = field(default_factory=list)
    production_artifacts: List[ProductionArtifact] = field(default_factory=list)
    optimization_epochs: List[OptimizationEpoch] = field(default_factory=list)
    pre_opt_params_per_macro: Dict[int, Dict[str, float]] = field(default_factory=dict)
    pre_opt_prediction_per_macro: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class LogParser:
    macro_start_re = re.compile(r"Macro Start \(([^)]+)\): (.*)")
    timing_re = re.compile(r"AWH Timing (Start|End): phase=([^|]+)\s+\|\s+leg=([^|]+)(.*)")
    block_status_re = re.compile(r"AWH Block Status:\s*(.*?)\s*\|\s*spent_ns:\s*(.*)")
    stage_a_re = re.compile(r"Stage A \(([^)]+)\): (.*)")
    stage_b_re = re.compile(r"Stage B \(([^)]+)\): (.*)")
    stage_b_cached_re = re.compile(r"Stage B Cached \(([^)]+)\): (.*)")
    stage_b_diag_re = re.compile(r"Stage B Diagnostics \(([^)]+)\): (.*)")
    probe_enter_re = re.compile(
        r"Stage A \(([^)]+)\) reached stable streak \((\d+)/(\d+)\); entering Stage B probe \| probe_steps=(\d+) \| probe_ns=([-\deE+.]+)"
    )
    probe_continue_re = re.compile(
        r"Stage B \(([^)]+)\) continuing frozen probe \| segment=(\d+)/(\d+) \| probe_steps=(\d+) \| probe_ns=([-\deE+.]+)"
    )
    split_continue_re = re.compile(
        r"Stage B \(([^)]+)\) split-only failure; continuing frozen probe in place \| next_segment_steps=(\d+) \| next_segment_ns=([-\deE+.]+) \| segments=(\d+)/(\d+)\."
    )
    cooldown_re = re.compile(r"Stage B \(([^)]+)\) cooldown active: remaining_checks=(\d+);")
    failure_keep_re = re.compile(
        r"Stage B \(([^)]+)\) failed \(([^)]+)\); keeping probe at (\d+) steps and scheduling cooldown=(\d+)"
    )
    failure_grow_re = re.compile(
        r"Stage B \(([^)]+)\) failed \(([^)]+)\); increasing next probe to (\d+) steps \(([-\deE+.]+) ns\) and scheduling cooldown=(\d+)"
    )
    near_pass_re = re.compile(
        r"Stage B \(([^)]+)\) near pass; retrying with probe=(\d+) steps \(([-\deE+.]+) ns\) after cooldown=(\d+)"
    )
    frozen_re = re.compile(
        r"([A-Za-z]+) leg frozen after passing Stage B \(split_gap=([-\deE+.]+) kT, parity_gap=([-\deE+.]+) kT\)\."
    )
    soften_re = re.compile(
        r"Stage B \(([^)]+)\) softening AWH bias: N_eff reduced from ([-\deE+.]+) to ([-\deE+.]+)"
    )
    strict_re = re.compile(r"Strict gating: Reverting Stage A \(([^)]+)\) to initial stage \(df=([-\deE+.]+), min_occ=([-\deE+.]+)\)")
    production_re = re.compile(
        r"Production artifact \(([^)]+)\): frames=(\d+) \| .*?states=(\d+) \| eval_threads=(\d+) \| lambda_tile=(\d+) \| eval_schedule=([A-Za-z0-9_]+)"
    )
    opt_epoch_re = re.compile(r">> Optimization Epoch: Active (?:Block|Pools) = (.+)")
    clip_re = re.compile(r"Step clipped by infinity-norm \(Scaling: ([-\deE+.]+)\)")
    ls_iter_new_re = re.compile(
        r"LS Iter (\d+) \(.*?=([-\deE+.]+)\): ESS\[(.*?)\](?: \| Drift\[(.*?)\])? \| Targets\[(.*?)\] \| Obj = ([-\deE+.]+) \| Gate\[(.*?)\]"
    )
    ls_iter_re = re.compile(r"LS Iter (\d+) \(.*?=([-\deE+.]+)\): ESS\[(.*?)\](?: \| Drift\[(.*?)\])? \| Res = ([-\deE+.]+)")
    opt_metrics_re = re.compile(r"--- Optimization Metrics \(Epoch (\d+) - (?:Block:\s*)?(.+)\) ---")
    objective_re = re.compile(
        r"Objective:\s+value\s*=\s*([-\deE+.]+)\s+\|\s+Accepted\s*=\s*([-\deE+.]+)\s+\|\s+Extra req\s*=\s*([-\deE+.]+)\s+\|\s+Threshold\s*=\s*([-\deE+.]+)(?:\s+\|\s+in_band\s*=\s*(true|false))?\s+\|\s+step_accepted\s*=\s*(true|false)"
    )
    prediction_re = re.compile(r"Prediction:\s+.*?=\s*([-\deE+.]+)\s*kT\s+\|\s+Target\s*=\s*([-\deE+.]+)\s*kT")
    error_re = re.compile(r"Error:\s+Residual\s*=\s*([-\deE+.]+)\s+\|\s+Huber dL/dE\s*=\s*([-\deE+.]+)")
    accepted_re = re.compile(r"Accepted:\s+Residual\s*=\s*([-\deE+.]+)\s+\|\s+Extra req\s*=\s*([-\deE+.]+)")
    gradients_re = re.compile(r"Gradients:\s+Norm\s*=\s*([-\deE+.]+)\s+\|\s+Max\s*=\s*([-\deE+.]+)")
    fim_re = re.compile(r"FIM .*?Raw Cond Number\s*=\s*([-\deE+.]+)\s+\|\s+Truncated Eigs\s*=\s*(\d+)\s*/\s*(\d+)")
    trust_region_health_re = re.compile(
        r"Trust Reg\.:.*?Health\s*=\s*([-\deE+.]+) \(scale=([-\deE+.]+)\) \| Confidence\s*=\s*([-\deE+.]+) \| KL target\s*=\s*([-\deE+.]+) \| Max solute .*?=\s*([-\deE+.]+)"
    )
    trust_region_re = re.compile(
        r"Trust Reg\.:.*?Confidence\s*=\s*([-\deE+.]+) \| KL target\s*=\s*([-\deE+.]+)(?: \| Max solute .*?=\s*([-\deE+.]+))?"
    )
    confidence_re = re.compile(
        r"Confidence:\s+endpoint_ΔG\s*=\s*([-\deE+.]+) \| cycle\s*=\s*([-\deE+.]+) \| gradient\s*=\s*([-\deE+.]+) \| eligible_legs\s*=\s*(\d+)"
    )
    confidence_skipped_re = re.compile(r"Confidence:\s+skipped_legs\s*=\s*(.*)")
    confidence_new_re = re.compile(
        r"Confidence:\s+prediction\s*=\s*([-\deE+.]+) \| objective\s*=\s*([-\deE+.]+) \| gradient\s*=\s*([-\deE+.]+) \| eligible_targets\s*=\s*(\d+)"
    )
    confidence_skipped_new_re = re.compile(r"Confidence:\s+skipped_targets\s*=\s*(.*)")
    kl_re = re.compile(r"KL Bound:\s+Est\. KL\s*=\s*([-\deE+.]+)\s+\|\s+Target\s*=\s*([-\deE+.]+)\s+\|\s+Scaling\s*=\s*([-\deE+.]+)")
    alpha_re = re.compile(r"Line Search:\s+Converged .*?=\s*([-\deE+.]+)")
    step_re = re.compile(r"Actual Step:\s+Max .*?=\s*([-\deE+.]+)\s+\(.*?=([-\deE+.]+)\)")
    params_re = re.compile(r"Params .*?Min\s*=\s*([-\deE+.]+)\s+\|\s+Max\s*=\s*([-\deE+.]+)")
    leg_metric_re = re.compile(
        r"Leg ([^:]+): coeff=([-\deE+.]+) \| .*?=([-\deE+.]+) kT \| ESS=([-\deE+.]+) / ([-\deE+.]+) \| N_active=([-\deE+.]+)"
    )
    restored_re = re.compile(r"Restored best inner-loop state from Epoch (\d+) \(Residual = ([-\deE+.]+)\)")
    solvent_invariant_re = re.compile(r"Solvent Invariant: max \|.*?\| = ([-\deE+.]+)")
    initial_prediction_re = re.compile(
        r"Initial Prediction:.*?=\s*([-\deE+.]+)\s*kT\s*\|\s*Target\s*=\s*([-\deE+.]+)\s*kT\s*\|\s*Residual\s*=\s*([-\deE+.]+)"
    )

    def __init__(self, path: Path):
        self.path = path
        self.run = RunRecord(log_path=str(path))
        self.current_macro = 0
        self.global_ns_by_leg: Dict[str, float] = {}
        self.current_block_status: Dict[str, Dict[str, object]] = {}
        self.pending_stage_b_snapshot_fresh: Dict[str, bool] = {}
        self.stage_b_attempt_counter: Dict[str, int] = {}
        self.current_opt_epoch: Optional[OptimizationEpoch] = None
        self.in_param_block = False
        self.in_initial_param_block = False

    def parse(self) -> RunRecord:
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                message = split_message(raw_line)
                if message is None:
                    continue
                self._parse_message(line_no, message.strip())
        self._finalize_current_opt_epoch()
        return self.run

    def _finalize_current_opt_epoch(self) -> None:
        if self.current_opt_epoch is None:
            return
        self.run.optimization_epochs.append(self.current_opt_epoch)
        self.current_opt_epoch = None
        self.in_param_block = False

    def _current_leg_state(self, leg: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
        leg_key = leg.lower()
        status = self.current_block_status.get(leg_key, {})
        return (
            self.global_ns_by_leg.get(leg_key),
            status.get("spent_ns"),
            status.get("budget_ns"),
            status.get("status"),
        )

    def _append_control(self, line_no: int, leg: str, event_type: str, details: Dict[str, object]) -> None:
        global_ns, _, _, _ = self._current_leg_state(leg)
        self.run.control_events.append(
            ControlEvent(
                macro_index=self.current_macro,
                leg=leg.lower(),
                line_no=line_no,
                global_ns=global_ns,
                event_type=event_type,
                details=details,
            )
        )

    def _parse_macro_flags(self, body: str) -> Dict[str, object]:
        flags: Dict[str, object] = {}
        for chunk in body.split("|"):
            item = chunk.strip().rstrip(".")
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in {"warm_start", "bias_reused", "idx_match"}:
                flags[key] = to_bool(value)
            elif key in {"\u03bb_prev", "\u03bb_new"}:
                match = re.search(r"(\d+)", value)
                if match:
                    flags[key] = int(match.group(1))
            elif key in {"restart_rmsd_nm", "volume_ratio"}:
                try:
                    flags[key] = to_float(value)
                except ValueError:
                    pass
            else:
                flags[key] = value
        return flags

    def _parse_timing_tail(self, tail: str) -> Dict[str, object]:
        data: Dict[str, object] = {}
        for chunk in tail.split("|"):
            item = chunk.strip()
            if not item or "=" not in item:
                continue
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "md_steps":
                data[key] = to_int(value)
            elif key in {"md_ns", "wall_s", "steps_per_s"}:
                data[key] = to_float(value)
            else:
                data[key] = value
        return data

    def _parse_stage_a_segments(self, body: str) -> Dict[str, object]:
        data: Dict[str, object] = {}
        segments = [segment.strip() for segment in body.split(" | ")]
        for segment in segments:
            if segment.startswith("df="):
                match = re.search(r"df=([-\deE+.]+) \(ok=(true|false)\)", segment)
                if match:
                    data["df"] = to_float(match.group(1))
                    data["df_ok"] = to_bool(match.group(2))
            elif segment.startswith("ess="):
                match = re.search(r"ess=([-\deE+.]+) \(ok=(true|false)\)", segment)
                if match:
                    data["ess"] = to_float(match.group(1))
                    data["ess_ok"] = to_bool(match.group(2))
            elif segment.startswith("lin_neff="):
                match = re.search(r"lin_neff=([-\deE+.]+) \(ok=(true|false)\)", segment)
                if match:
                    data["lin_neff"] = to_float(match.group(1))
                    data["lin_neff_ok"] = to_bool(match.group(2))
            elif segment.startswith("tau_int_est="):
                data["tau_int_est"] = to_float(segment.split("=", 1)[1])
            elif segment.startswith("switches="):
                data["switches"] = to_int(segment.split("=", 1)[1])
            elif segment.startswith("dwell_samples="):
                match = re.search(r"mean=([-\deE+.]+), med=([-\deE+.]+)", segment)
                if match:
                    data["dwell_mean"] = to_float(match.group(1))
                    data["dwell_median"] = to_float(match.group(2))
            elif segment.startswith("rt="):
                match = re.search(r"rt=(\d+) \(ok=(true|false)\)", segment)
                if match:
                    data["round_trips"] = to_int(match.group(1))
                    data["round_trips_ok"] = to_bool(match.group(2))
            elif segment.startswith("endpt_recent="):
                match = re.search(r"low=([-\deE+.]+), band=([-\deE+.]+); .*?req>=([-\deE+.]+)\) \(ok=(true|false)\)", segment)
                if match:
                    data["endpoint_low"] = to_float(match.group(1))
                    data["endpoint_band"] = to_float(match.group(2))
                    data["endpoint_required"] = to_float(match.group(3))
                    data["endpoint_ok"] = to_bool(match.group(4))
            elif segment.startswith("tail=("):
                match = re.search(r"\u03a3=([-\deE+.]+), min=([-\deE+.]+), ok=(true|false)\)", segment)
                if match:
                    data["tail_sum"] = to_float(match.group(1))
                    data["tail_min"] = to_float(match.group(2))
                    data["tail_ok"] = to_bool(match.group(3))
            elif segment.startswith("tail_low="):
                data["tail_low_states"] = parse_index_list(segment.split("=", 1)[1])
            elif segment.startswith("occ_min="):
                data["occ_min"] = to_float(segment.split("=", 1)[1])
            elif segment.startswith("low_occ="):
                data["low_occ_states"] = parse_index_list(segment.split("=", 1)[1])
            elif segment.startswith("rollback="):
                data["rollback"] = segment.split("=", 1)[1].strip()
            elif segment.startswith("occ_floor="):
                data["occ_floor"] = to_float(segment.split("=", 1)[1])
            elif segment.startswith("frac_above_floor="):
                data["frac_above_floor"] = to_float(segment.split("=", 1)[1])
            elif segment.startswith("n_hist_recent="):
                data["n_hist_recent"] = to_int(segment.split("=", 1)[1])
            elif segment.startswith("n_hist_total="):
                data["n_hist_total"] = to_int(segment.split("=", 1)[1])
            elif segment.startswith("streak="):
                data["streak"] = to_int(segment.split("=", 1)[1])
            elif segment.startswith("cooldown="):
                data["cooldown"] = to_int(segment.split("=", 1)[1])
            elif segment.startswith("failures="):
                data["failures"] = to_int(segment.split("=", 1)[1])
        data["stage_ready"] = data.get("streak", 0) > 0
        return data

    def _parse_stage_b_segments(self, body: str) -> Dict[str, object]:
        data: Dict[str, object] = {}
        segments = [segment.strip() for segment in body.split(" | ")]
        for segment in segments:
            if segment.startswith("split_gap="):
                match = re.search(r"split_gap=([-\deE+.]+) kT \(ok=(true|false)\)", segment)
                if match:
                    data["split_gap"] = to_float(match.group(1))
                    data["split_ok"] = to_bool(match.group(2))
            elif segment.startswith("parity_gap="):
                match = re.search(r"parity_gap=([-\deE+.]+) kT \(ok=(true|false)\)", segment)
                if match:
                    data["parity_gap"] = to_float(match.group(1))
                    data["parity_ok"] = to_bool(match.group(2))
            elif segment.startswith("raw_parity="):
                data["raw_parity_gap"] = to_float(segment.split("=", 1)[1])
            elif segment.startswith("supported_parity="):
                data["supported_parity_gap"] = to_float(segment.split("=", 1)[1])
            elif segment.startswith("endpoint_parity="):
                match = re.search(r"endpoint_parity=([-\deE+.]+) \(ok=(true|false)\)", segment)
                if match:
                    data["endpoint_parity_gap"] = to_float(match.group(1))
                    data["endpoint_ok"] = to_bool(match.group(2))
            elif segment.startswith("support_coverage="):
                match = re.search(r"support_coverage=(\d+)/(\d+) \(required>=(\d+), ok=(true|false)\) @ ess>=(.*)", segment)
                if match:
                    data["n_supported_states"] = to_int(match.group(1))
                    data["n_states"] = to_int(match.group(2))
                    data["required_supported_states"] = to_int(match.group(3))
                    data["support_ok"] = to_bool(match.group(4))
                    data["support_threshold"] = to_float(match.group(5))
            elif segment.startswith("failure="):
                data["failure_mode"] = segment.split("=", 1)[1]
            elif segment.startswith("near_pass="):
                data["near_pass"] = to_bool(segment.split("=", 1)[1])
            elif segment.startswith("mode="):
                data["accumulation_mode"] = segment.split("=", 1)[1]
            elif segment.startswith("frames="):
                data["frames"] = to_int(segment.split("=", 1)[1])
            elif segment.startswith("accumulated_frames="):
                data["accumulated_frames"] = to_int(segment.split("=", 1)[1])
            elif segment.startswith("probe_segments="):
                data["probe_segments"] = to_int(segment.split("=", 1)[1])
        return data

    def _parse_message(self, line_no: int, message: str) -> None:
        macro_match = self.macro_start_re.match(message)
        if macro_match:
            leg = macro_match.group(1).strip().lower()
            if leg == "solvent":
                self.current_macro += 1
            flags = self._parse_macro_flags(macro_match.group(2))
            self.run.macro_starts.append(
                MacroStart(
                    macro_index=self.current_macro,
                    leg=leg,
                    line_no=line_no,
                    warm_start=flags.get("warm_start"),
                    bias_reused=flags.get("bias_reused"),
                    lambda_prev=flags.get("\u03bb_prev"),
                    lambda_new=flags.get("\u03bb_new"),
                    idx_match=flags.get("idx_match"),
                    restart_rmsd_nm=flags.get("restart_rmsd_nm"),
                    volume_ratio=flags.get("volume_ratio"),
                    raw=message,
                )
            )
            return

        timing_match = self.timing_re.match(message)
        if timing_match:
            kind = timing_match.group(1).lower()
            phase = timing_match.group(2).strip()
            leg = timing_match.group(3).strip().lower()
            meta = self._parse_timing_tail(timing_match.group(4))
            event = PhaseEvent(
                macro_index=self.current_macro,
                leg=leg,
                line_no=line_no,
                kind=kind,
                phase=phase,
                md_steps=meta.get("md_steps"),
                md_ns=meta.get("md_ns"),
                wall_s=meta.get("wall_s"),
                steps_per_s=meta.get("steps_per_s"),
            )
            if kind == "end" and event.md_ns is not None:
                self.global_ns_by_leg[leg] = self.global_ns_by_leg.get(leg, 0.0) + event.md_ns
                event.global_ns_after = self.global_ns_by_leg[leg]
            else:
                event.global_ns_after = self.global_ns_by_leg.get(leg)
            self.run.phase_events.append(event)
            return

        block_match = self.block_status_re.match(message)
        if block_match:
            status_tokens = [token for token in block_match.group(1).split() if "=" in token]
            for token in status_tokens:
                leg, status = parse_status_token(token)
                entry = self.current_block_status.setdefault(leg.lower(), {})
                entry["status"] = status
            for token in block_match.group(2).split("|"):
                spent_token = token.strip()
                if not spent_token or "=" not in spent_token:
                    continue
                leg, spent_ns, budget_ns = parse_spent_token(spent_token)
                entry = self.current_block_status.setdefault(leg.lower(), {})
                entry["spent_ns"] = spent_ns
                entry["budget_ns"] = budget_ns
            return

        stage_a_match = self.stage_a_re.match(message)
        if stage_a_match:
            leg = stage_a_match.group(1).strip().lower()
            global_ns, spent_ns, budget_ns, status = self._current_leg_state(leg)
            snapshot = StageASnapshot(
                macro_index=self.current_macro,
                leg=leg,
                line_no=line_no,
                global_ns=global_ns,
                macro_spent_ns=spent_ns,
                budget_ns=budget_ns,
                leg_status=status,
                **self._parse_stage_a_segments(stage_a_match.group(2)),
            )
            self.run.stage_a.append(snapshot)
            return

        stage_b_match = self.stage_b_re.match(message)
        if stage_b_match:
            leg = stage_b_match.group(1).strip().lower()
            global_ns, spent_ns, budget_ns, status = self._current_leg_state(leg)
            fresh = self.pending_stage_b_snapshot_fresh.pop(leg, False)
            attempt_index = None
            if fresh:
                self.stage_b_attempt_counter[leg] = self.stage_b_attempt_counter.get(leg, 0) + 1
                attempt_index = self.stage_b_attempt_counter[leg]
            elif leg in self.stage_b_attempt_counter:
                attempt_index = self.stage_b_attempt_counter[leg]
            snapshot = StageBSnapshot(
                macro_index=self.current_macro,
                leg=leg,
                line_no=line_no,
                global_ns=global_ns,
                macro_spent_ns=spent_ns,
                budget_ns=budget_ns,
                leg_status=status,
                fresh_probe_result=fresh,
                attempt_index=attempt_index,
                **self._parse_stage_b_segments(stage_b_match.group(2)),
            )
            self.run.stage_b.append(snapshot)
            return

        stage_b_cached_match = self.stage_b_cached_re.match(message)
        if stage_b_cached_match:
            leg = stage_b_cached_match.group(1).strip().lower()
            details: Dict[str, object] = {}
            for chunk in stage_b_cached_match.group(2).split("|"):
                item = chunk.strip()
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                details[key.strip()] = value.strip()
            self._append_control(line_no, leg, "stage_b_cached", details)
            return

        diag_match = self.stage_b_diag_re.match(message)
        if diag_match:
            leg = diag_match.group(1).strip().lower()
            for snapshot in reversed(self.run.stage_b):
                if snapshot.leg == leg:
                    snapshot.diagnostics = diag_match.group(2).strip()
                    break
            return

        probe_enter_match = self.probe_enter_re.match(message)
        if probe_enter_match:
            leg = probe_enter_match.group(1).strip().lower()
            self.pending_stage_b_snapshot_fresh[leg] = True
            self._append_control(
                line_no,
                leg,
                "probe_enter",
                {
                    "stable_streak": to_int(probe_enter_match.group(2)),
                    "target_streak": to_int(probe_enter_match.group(3)),
                    "probe_steps": to_int(probe_enter_match.group(4)),
                    "probe_ns": to_float(probe_enter_match.group(5)),
                },
            )
            return

        probe_continue_match = self.probe_continue_re.match(message)
        if probe_continue_match:
            leg = probe_continue_match.group(1).strip().lower()
            self.pending_stage_b_snapshot_fresh[leg] = True
            self._append_control(
                line_no,
                leg,
                "probe_continue",
                {
                    "segment": to_int(probe_continue_match.group(2)),
                    "max_segments": to_int(probe_continue_match.group(3)),
                    "probe_steps": to_int(probe_continue_match.group(4)),
                    "probe_ns": to_float(probe_continue_match.group(5)),
                },
            )
            return

        split_continue_match = self.split_continue_re.match(message)
        if split_continue_match:
            self._append_control(
                line_no,
                split_continue_match.group(1).strip().lower(),
                "split_only_continue",
                {
                    "next_probe_steps": to_int(split_continue_match.group(2)),
                    "next_probe_ns": to_float(split_continue_match.group(3)),
                    "segment": to_int(split_continue_match.group(4)),
                    "max_segments": to_int(split_continue_match.group(5)),
                },
            )
            return

        cooldown_match = self.cooldown_re.match(message)
        if cooldown_match:
            self._append_control(
                line_no,
                cooldown_match.group(1).strip().lower(),
                "cooldown",
                {"remaining_checks": to_int(cooldown_match.group(2))},
            )
            return

        keep_match = self.failure_keep_re.match(message)
        if keep_match:
            self._append_control(
                line_no,
                keep_match.group(1).strip().lower(),
                "probe_retry_keep",
                {
                    "failure_mode": keep_match.group(2),
                    "next_probe_steps": to_int(keep_match.group(3)),
                    "cooldown": to_int(keep_match.group(4)),
                },
            )
            return

        grow_match = self.failure_grow_re.match(message)
        if grow_match:
            self._append_control(
                line_no,
                grow_match.group(1).strip().lower(),
                "probe_retry_grow",
                {
                    "failure_mode": grow_match.group(2),
                    "next_probe_steps": to_int(grow_match.group(3)),
                    "next_probe_ns": to_float(grow_match.group(4)),
                    "cooldown": to_int(grow_match.group(5)),
                },
            )
            return

        near_match = self.near_pass_re.match(message)
        if near_match:
            self._append_control(
                line_no,
                near_match.group(1).strip().lower(),
                "probe_retry_near_pass",
                {
                    "next_probe_steps": to_int(near_match.group(2)),
                    "next_probe_ns": to_float(near_match.group(3)),
                    "cooldown": to_int(near_match.group(4)),
                },
            )
            return

        frozen_match = self.frozen_re.match(message)
        if frozen_match:
            self._append_control(
                line_no,
                frozen_match.group(1).strip().lower(),
                "leg_frozen",
                {
                    "split_gap": to_float(frozen_match.group(2)),
                    "parity_gap": to_float(frozen_match.group(3)),
                },
            )
            return

        soften_match = self.soften_re.search(message)
        if soften_match:
            self._append_control(
                line_no,
                soften_match.group(1).strip().lower(),
                "bias_softened",
                {
                    "n_eff_before": to_float(soften_match.group(2)),
                    "n_eff_after": to_float(soften_match.group(3)),
                },
            )
            return

        strict_match = self.strict_re.search(message)
        if strict_match:
            self._append_control(
                line_no,
                strict_match.group(1).strip().lower(),
                "strict_reset",
                {
                    "df": to_float(strict_match.group(2)),
                    "min_occ": to_float(strict_match.group(3)),
                },
            )
            return

        production_match = self.production_re.match(message)
        if production_match:
            self.run.production_artifacts.append(
                ProductionArtifact(
                    macro_index=self.current_macro,
                    leg=production_match.group(1).strip().lower(),
                    line_no=line_no,
                    frames=to_int(production_match.group(2)),
                    lambda_states=to_int(production_match.group(3)),
                    eval_threads=to_int(production_match.group(4)),
                    lambda_tile=to_int(production_match.group(5)),
                    eval_schedule=production_match.group(6).strip(),
                )
            )
            return

        if message.startswith("--- Initial Parameter State"):
            self._finalize_current_opt_epoch()
            self.in_initial_param_block = True
            self.run.pre_opt_params_per_macro.setdefault(self.current_macro, {})
            return

        if self.in_initial_param_block:
            if message.startswith("---") or self.opt_epoch_re.search(message):
                self.in_initial_param_block = False
            elif ":" in message:
                name, value = message.split(":", 1)
                try:
                    self.run.pre_opt_params_per_macro[self.current_macro][name.strip()] = to_float(value)
                except ValueError:
                    pass
                return

        opt_epoch_match = self.opt_epoch_re.search(message)
        if opt_epoch_match:
            self._finalize_current_opt_epoch()
            self.in_initial_param_block = False
            self.current_opt_epoch = OptimizationEpoch(
                macro_index=self.current_macro,
                global_epoch_index=len(self.run.optimization_epochs) + 1,
                line_no=line_no,
                block_name=opt_epoch_match.group(1).strip(),
            )
            return

        if self.current_opt_epoch is not None:
            clip_match = self.clip_re.search(message)
            if clip_match:
                self.current_opt_epoch.step_clipped_scaling = to_float(clip_match.group(1))
                return

            ls_iter_new_match = self.ls_iter_new_re.search(message)
            if ls_iter_new_match:
                iter_data = {
                    "iter": to_int(ls_iter_new_match.group(1)),
                    "alpha": to_float(ls_iter_new_match.group(2)),
                    "ess": parse_probe_ess_map(ls_iter_new_match.group(3)),
                    "targets": parse_named_value_map(ls_iter_new_match.group(5)),
                    "objective": to_float(ls_iter_new_match.group(6)),
                    "gate": parse_gate_fields(ls_iter_new_match.group(7)),
                }
                if ls_iter_new_match.group(4):
                    iter_data["drift"] = parse_pool_drift_map(ls_iter_new_match.group(4))
                self.current_opt_epoch.line_search_iters.append(iter_data)
                return

            ls_iter_match = self.ls_iter_re.search(message)
            if ls_iter_match:
                iter_data = {
                    "iter": to_int(ls_iter_match.group(1)),
                    "alpha": to_float(ls_iter_match.group(2)),
                    "ess": parse_probe_ess_map(ls_iter_match.group(3)),
                    "residual": to_float(ls_iter_match.group(5)),
                }
                if ls_iter_match.group(4):
                    iter_data["drift"] = parse_pool_drift_map(ls_iter_match.group(4))
                self.current_opt_epoch.line_search_iters.append(iter_data)
                return

            if "Line search converged" in message:
                self.current_opt_epoch.line_search_converged = True
                if self.current_opt_epoch.line_search_iters:
                    last_iter = self.current_opt_epoch.line_search_iters[-1]
                    if "objective" in last_iter:
                        self.current_opt_epoch.accepted_objective = float(last_iter["objective"])
                    if "residual" in last_iter:
                        self.current_opt_epoch.accepted_residual = float(last_iter["residual"])
                return

            if message.startswith("--- Current Parameter State ---"):
                self.in_param_block = True
                return

            if self.in_param_block and message.startswith("--- Optimization Metrics"):
                self.in_param_block = False

            if self.in_param_block:
                if ":" in message:
                    name, value = message.split(":", 1)
                    try:
                        self.current_opt_epoch.parameters[name.strip()] = to_float(value)
                    except ValueError:
                        pass
                    return

            opt_metrics_match = self.opt_metrics_re.match(message)
            if opt_metrics_match:
                self.current_opt_epoch.epoch_in_macro = to_int(opt_metrics_match.group(1))
                self.current_opt_epoch.block_name = opt_metrics_match.group(2).strip()
                return

            init_pred_match = self.initial_prediction_re.search(message)
            if init_pred_match:
                macro = self.current_opt_epoch.macro_index
                self.run.pre_opt_prediction_per_macro[macro] = (
                    to_float(init_pred_match.group(1)),
                    to_float(init_pred_match.group(2)),
                    to_float(init_pred_match.group(3)),
                )
                return

            prediction_match = self.prediction_re.search(message)
            if prediction_match:
                self.current_opt_epoch.prediction = to_float(prediction_match.group(1))
                self.current_opt_epoch.target = to_float(prediction_match.group(2))
                return

            objective_match = self.objective_re.search(message)
            if objective_match:
                self.current_opt_epoch.objective_value = to_float(objective_match.group(1))
                self.current_opt_epoch.accepted_objective = to_float(objective_match.group(2))
                self.current_opt_epoch.extra_residual_requirement = to_float(objective_match.group(3))
                self.current_opt_epoch.objective_threshold = to_float(objective_match.group(4))
                self.current_opt_epoch.objective_in_band = to_bool(objective_match.group(5))
                self.current_opt_epoch.objective_step_accepted = to_bool(objective_match.group(6))
                return

            error_match = self.error_re.search(message)
            if error_match:
                self.current_opt_epoch.residual = to_float(error_match.group(1))
                self.current_opt_epoch.huber_dlde = to_float(error_match.group(2))
                return

            accepted_match = self.accepted_re.search(message)
            if accepted_match:
                self.current_opt_epoch.accepted_residual = to_float(accepted_match.group(1))
                self.current_opt_epoch.extra_residual_requirement = to_float(accepted_match.group(2))
                return

            gradients_match = self.gradients_re.search(message)
            if gradients_match:
                self.current_opt_epoch.grad_norm = to_float(gradients_match.group(1))
                self.current_opt_epoch.grad_max = to_float(gradients_match.group(2))
                return

            fim_match = self.fim_re.search(message)
            if fim_match:
                self.current_opt_epoch.fim_raw_cond = to_float(fim_match.group(1))
                self.current_opt_epoch.truncated_eigs = to_int(fim_match.group(2))
                self.current_opt_epoch.fim_rank = to_int(fim_match.group(3))
                return

            trust_region_health_match = self.trust_region_health_re.search(message)
            if trust_region_health_match:
                self.current_opt_epoch.health_score = to_float(trust_region_health_match.group(1))
                self.current_opt_epoch.health_scale = to_float(trust_region_health_match.group(2))
                self.current_opt_epoch.confidence_scale = to_float(trust_region_health_match.group(3))
                self.current_opt_epoch.kl_target = to_float(trust_region_health_match.group(4))
                self.current_opt_epoch.max_solute_phi_step = to_float(trust_region_health_match.group(5))
                return

            trust_region_match = self.trust_region_re.search(message)
            if trust_region_match:
                self.current_opt_epoch.confidence_scale = to_float(trust_region_match.group(1))
                self.current_opt_epoch.kl_target = to_float(trust_region_match.group(2))
                if trust_region_match.group(3):
                    self.current_opt_epoch.max_solute_phi_step = to_float(trust_region_match.group(3))
                return

            confidence_match = self.confidence_re.search(message)
            if confidence_match:
                self.current_opt_epoch.confidence_endpoint_disagreement = to_float(confidence_match.group(1))
                self.current_opt_epoch.confidence_cycle_disagreement = to_float(confidence_match.group(2))
                self.current_opt_epoch.confidence_gradient_disagreement = to_float(confidence_match.group(3))
                self.current_opt_epoch.confidence_eligible_legs = to_int(confidence_match.group(4))
                return

            confidence_new_match = self.confidence_new_re.search(message)
            if confidence_new_match:
                self.current_opt_epoch.confidence_prediction_disagreement = to_float(confidence_new_match.group(1))
                self.current_opt_epoch.confidence_objective_disagreement = to_float(confidence_new_match.group(2))
                self.current_opt_epoch.confidence_gradient_disagreement = to_float(confidence_new_match.group(3))
                self.current_opt_epoch.confidence_eligible_targets = to_int(confidence_new_match.group(4))
                return

            confidence_skipped_match = self.confidence_skipped_re.search(message)
            if confidence_skipped_match:
                skipped = confidence_skipped_match.group(1).strip()
                self.current_opt_epoch.confidence_skipped_legs = [item for item in skipped.split(",") if item]
                return

            confidence_skipped_new_match = self.confidence_skipped_new_re.search(message)
            if confidence_skipped_new_match:
                skipped = confidence_skipped_new_match.group(1).strip()
                self.current_opt_epoch.confidence_skipped_targets = [item.strip() for item in skipped.split(",") if item.strip()]
                return

            kl_match = self.kl_re.search(message)
            if kl_match:
                self.current_opt_epoch.kl_est = to_float(kl_match.group(1))
                self.current_opt_epoch.kl_target = to_float(kl_match.group(2))
                self.current_opt_epoch.kl_scaling = to_float(kl_match.group(3))
                return

            alpha_match = self.alpha_re.search(message)
            if alpha_match:
                self.current_opt_epoch.line_search_alpha = to_float(alpha_match.group(1))
                self.current_opt_epoch.line_search_converged = True
                return

            step_match = self.step_re.search(message)
            if step_match:
                self.current_opt_epoch.actual_max_phi_step = to_float(step_match.group(1))
                return

            params_match = self.params_re.search(message)
            if params_match:
                self.current_opt_epoch.params_min = to_float(params_match.group(1))
                self.current_opt_epoch.params_max = to_float(params_match.group(2))
                return

            if message.startswith("Target "):
                target_body = message[len("Target ") :]
                if ":" in target_body:
                    target_name, payload = target_body.split(":", 1)
                    fields = parse_pipe_fields(payload)
                    pred_value, pred_unit = split_numeric_suffix(fields.get("pred", ""))
                    target_metric: Dict[str, object] = {
                        "kind": fields.get("kind", ""),
                        "prediction": pred_value,
                        "unit": pred_unit,
                        "reference": parse_optional_float(fields.get("ref", "")),
                        "residual": parse_optional_float(fields.get("residual", "")),
                        "weight": parse_optional_float(fields.get("weight", "")),
                        "loss": parse_optional_float(fields.get("loss", "")),
                        "ESS": parse_optional_float(fields.get("ESS", "")),
                    }
                    if "leg" in fields:
                        target_metric["leg"] = fields["leg"]
                    if "state" in fields:
                        target_metric["state"] = fields["state"]
                    target_name = target_name.strip()
                    self.current_opt_epoch.target_metrics[target_name] = target_metric
                    if target_metric.get("kind") == "cycle_free_energy" and self.current_opt_epoch.prediction is None:
                        self.current_opt_epoch.prediction = pred_value
                        self.current_opt_epoch.target = target_metric.get("reference")
                        self.current_opt_epoch.residual = target_metric.get("residual")
                    return

            leg_metric_match = self.leg_metric_re.search(message)
            if leg_metric_match:
                leg = leg_metric_match.group(1).strip().lower()
                self.current_opt_epoch.leg_metrics[leg] = {
                    "coeff": to_float(leg_metric_match.group(2)),
                    "dG": to_float(leg_metric_match.group(3)),
                    "ESS": to_float(leg_metric_match.group(4)),
                    "ESS_threshold": to_float(leg_metric_match.group(5)),
                    "N_active": to_float(leg_metric_match.group(6)),
                }
                return

        restored_match = self.restored_re.search(message)
        if restored_match:
            self.run.warnings.append(
                "restored_best_inner_loop_epoch=%s residual=%s"
                % (restored_match.group(1), restored_match.group(2))
            )
            return

        solvent_match = self.solvent_invariant_re.search(message)
        if solvent_match:
            self.run.warnings.append("solvent_invariant=%s" % solvent_match.group(1))
            return



@dataclass
class SeriesSpec:
    name: str
    points: List[Tuple[float, float]]
    color: str
    stroke_width: float = 2.2
    dashed: bool = False
    draw_markers: bool = False
    alpha: float = 1.0


@dataclass
class HLineSpec:
    value: float
    color: str
    label: str = ""
    dashed: bool = True
    alpha: float = 0.9


@dataclass
class VLineSpec:
    value: float
    color: str
    label: str = ""
    dashed: bool = True
    alpha: float = 0.8


@dataclass
class MarkerSpec:
    x: float
    y: float
    color: str
    label: str = ""
    radius: float = 4.0
    marker: str = "o"


@dataclass
class PanelSpec:
    title: str
    x_label: str
    y_label: str
    series: List[SeriesSpec] = field(default_factory=list)
    hlines: List[HLineSpec] = field(default_factory=list)
    vlines: List[VLineSpec] = field(default_factory=list)
    markers: List[MarkerSpec] = field(default_factory=list)
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    legend: bool = True
    notes: List[str] = field(default_factory=list)


import numpy as np
from bisect import bisect_right
from collections import defaultdict
from matplotlib.patches import Rectangle
from matplotlib import colors as mcolors


STATUS_COLORS = {
    "pass": "#2a9d8f",
    "fail": "#d62828",
    "warn": "#f4a261",
    "muted": "#adb5bd",
    "info": "#457b9d",
    "vacuum": LEG_COLORS["vacuum"],
    "solvent": LEG_COLORS["solvent"],
}

PHASE_MARKERS = {
    "Initial Rewarm": "s",
    "Stage A Block": "o",
    "Stage B Probe MD": "^",
    "Stage B Neighbor Precompute": "D",
    "Stage B Ensemble Eval": "P",
    "Stage B Split-Parity": "X",
}

STAGE_B_STATE_ORDER = [
    "none",
    "cached_pass",
    "cached_fail",
    "fresh_pass",
    "fresh_fail",
    "cooldown",
]

STAGE_B_STATE_COLORS = {
    "none": "#e9ecef",
    "cached_pass": "#b7e4c7",
    "cached_fail": "#ffd6a5",
    "fresh_pass": "#2a9d8f",
    "fresh_fail": "#d62828",
    "cooldown": "#4d96ff",
}

LAMBDA_FLAG_COLORS = {
    0: "#f8f9fa",
    1: "#ffd166",
    2: "#ef476f",
    3: "#7b2cbf",
}

GATE_PASS_COLORS = {
    -1: "#f1f3f5",
    0: "#f28482",
    1: "#84a98c",
}


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def sanitize_json(data: object) -> object:
    if isinstance(data, dict):
        return {str(key): sanitize_json(value) for key, value in data.items()}
    if isinstance(data, list):
        return [sanitize_json(value) for value in data]
    if isinstance(data, float):
        return data if math.isfinite(data) else None
    return data


def write_summary_json(outdir: Path, run: RunRecord) -> None:
    payload = sanitize_json(asdict(run))
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def apply_axis_style(ax: plt.Axes) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, which="major", color="#e9ecef", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#d0d7de")


def leg_title(leg: str) -> str:
    return leg.capitalize()


def get_leg_stage_a(run: RunRecord, leg: str) -> List[StageASnapshot]:
    return [snap for snap in run.stage_a if snap.leg == leg]


def get_leg_stage_b(run: RunRecord, leg: str) -> List[StageBSnapshot]:
    return [snap for snap in run.stage_b if snap.leg == leg]


def block_line_numbers(run: RunRecord, leg: str) -> List[int]:
    return [snap.line_no for snap in get_leg_stage_a(run, leg)]


def block_index_for_line(run: RunRecord, leg: str, line_no: int) -> Optional[int]:
    lines = block_line_numbers(run, leg)
    if not lines:
        return None
    idx = bisect_right(lines, line_no) - 1
    if idx < 0:
        return None
    return idx


def macro_boundaries_from_stage_a(run: RunRecord, leg: str) -> List[Tuple[int, int]]:
    snaps = get_leg_stage_a(run, leg)
    boundaries: List[Tuple[int, int]] = []
    last_macro: Optional[int] = None
    for idx, snap in enumerate(snaps):
        if last_macro is None:
            last_macro = snap.macro_index
            continue
        if snap.macro_index != last_macro:
            boundaries.append((idx, snap.macro_index))
            last_macro = snap.macro_index
    return boundaries


def macro_boundaries_from_epochs(run: RunRecord) -> List[VLineSpec]:
    seen: set = set()
    boundaries: List[VLineSpec] = []
    for epoch in run.optimization_epochs:
        macro = epoch.macro_index
        if macro not in seen:
            seen.add(macro)
            if macro > 1:
                boundaries.append(VLineSpec(epoch.global_epoch_index, "#999999", label=f"M{macro}"))
    return boundaries


def latest_stage_a_snapshot(run: RunRecord, leg: str) -> Optional[StageASnapshot]:
    snaps = get_leg_stage_a(run, leg)
    return snaps[-1] if snaps else None


def latest_stage_b_snapshot(run: RunRecord, leg: str) -> Optional[StageBSnapshot]:
    snaps = get_leg_stage_b(run, leg)
    return snaps[-1] if snaps else None


def latest_cached_stage_b_event(run: RunRecord, leg: str) -> Optional[ControlEvent]:
    events = [
        ev for ev in run.control_events
        if ev.leg == leg and ev.event_type == "stage_b_cached"
    ]
    return events[-1] if events else None


def epoch_objective_value(epoch: OptimizationEpoch) -> Optional[float]:
    return epoch.objective_value if epoch.objective_value is not None else epoch.residual


def epoch_accepted_objective(epoch: OptimizationEpoch) -> Optional[float]:
    return epoch.accepted_objective if epoch.accepted_objective is not None else epoch.accepted_residual


def epoch_confidence_prediction(epoch: OptimizationEpoch) -> Optional[float]:
    return (
        epoch.confidence_prediction_disagreement
        if epoch.confidence_prediction_disagreement is not None
        else epoch.confidence_endpoint_disagreement
    )


def epoch_confidence_objective(epoch: OptimizationEpoch) -> Optional[float]:
    return (
        epoch.confidence_objective_disagreement
        if epoch.confidence_objective_disagreement is not None
        else epoch.confidence_cycle_disagreement
    )


def collect_target_names(epochs: Sequence[OptimizationEpoch]) -> List[str]:
    names: List[str] = []
    for epoch in epochs:
        for name in epoch.target_metrics:
            if name not in names:
                names.append(name)
    return names


def stage_a_gate_value(snapshot: StageASnapshot, gate: str) -> int:
    if gate == "df":
        return -1 if snapshot.df_ok is None else int(bool(snapshot.df_ok))
    if gate == "ESS":
        return -1 if snapshot.ess_ok is None else int(bool(snapshot.ess_ok))
    if gate == "linN":
        return -1 if snapshot.lin_neff_ok is None else int(bool(snapshot.lin_neff_ok))
    if gate == "RT":
        return -1 if snapshot.round_trips_ok is None else int(bool(snapshot.round_trips_ok))
    if gate == "endpoint":
        return -1 if snapshot.endpoint_ok is None else int(bool(snapshot.endpoint_ok))
    if gate == "tail":
        return -1 if snapshot.tail_ok is None else int(bool(snapshot.tail_ok))
    if gate == "occ":
        if snapshot.occ_min is None:
            return -1
        return 0 if snapshot.low_occ_states else 1
    if gate == "ready":
        return int(bool(snapshot.stage_ready))
    raise KeyError(gate)


def stage_a_primary_blocker(snapshot: Optional[StageASnapshot]) -> str:
    if snapshot is None:
        return "no Stage A data"
    checks = [
        (snapshot.df_ok is False, "df"),
        (snapshot.ess_ok is False, "ESS"),
        (snapshot.lin_neff_ok is False, "linear N_eff"),
        (snapshot.round_trips_ok is False, "round trips"),
        (snapshot.endpoint_ok is False, "endpoint occupancy"),
        (snapshot.tail_ok is False, "tail occupancy"),
        (bool(snapshot.low_occ_states), "low-occ λ"),
    ]
    failed = [name for cond, name in checks if cond]
    if failed:
        return ", ".join(failed[:3])
    return "none"


def stage_b_status_text(run: RunRecord, leg: str) -> str:
    snap = latest_stage_b_snapshot(run, leg)
    cached = latest_cached_stage_b_event(run, leg)
    if snap is not None:
        mode = snap.failure_mode or "unknown"
        if snap.fresh_probe_result:
            return f"fresh {mode}"
        return mode
    if cached is not None:
        return f"cached {cached.details.get('failure', 'unknown')}"
    return "not checked"


def status_color_from_text(text: str) -> str:
    lower = text.lower()
    if "pass" in lower or "frozen" in lower:
        return STATUS_COLORS["pass"]
    if "cached" in lower or "cooldown" in lower or "near" in lower:
        return STATUS_COLORS["warn"]
    if "not checked" in lower:
        return STATUS_COLORS["muted"]
    return STATUS_COLORS["fail"]


def max_lambda_state(run: RunRecord) -> int:
    values: List[int] = []
    for snap in run.stage_a:
        values.extend(snap.tail_low_states)
        values.extend(snap.low_occ_states)
    for snap in run.stage_b:
        if snap.n_states:
            values.append(snap.n_states)
    for artifact in run.production_artifacts:
        values.append(artifact.lambda_states)
    return max(values) if values else 1


def safe_log10(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    return math.log10(value)


def finite_xy(points: Sequence[Tuple[Optional[float], Optional[float]]]) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for x, y in points:
        if x is None or y is None:
            continue
        if math.isfinite(x) and math.isfinite(y):
            result.append((float(x), float(y)))
    return result


def write_dashboard(path: Path, run: RunRecord) -> None:
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    fig.patch.set_facecolor("#fbfbfd")

    ax.text(0.02, 0.96, "AWH Progress Dashboard", fontsize=22, fontweight="bold", va="top")
    ax.text(0.02, 0.92, Path(run.log_path).name, fontsize=11, color="#666666", va="top")

    card_positions = {
        "solvent": (0.02, 0.53, 0.29, 0.31),
        "vacuum": (0.35, 0.53, 0.29, 0.31),
        "totals": (0.68, 0.53, 0.30, 0.31),
        "notes": (0.02, 0.10, 0.96, 0.32),
    }

    def card(x: float, y: float, w: float, h: float, title: str) -> None:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#ffffff", edgecolor="#d9dce3", linewidth=1.0))
        ax.text(x + 0.02, y + h - 0.04, title, fontsize=14, fontweight="bold", va="top")

    for title, (x, y, w, h) in card_positions.items():
        label = title.capitalize() if title in {"solvent", "vacuum"} else ("Run Totals" if title == "totals" else "Current Interpretation")
        card(x, y, w, h, label)

    for leg in ("solvent", "vacuum"):
        x, y, w, h = card_positions[leg]
        snap = latest_stage_a_snapshot(run, leg)
        stage_b = latest_stage_b_snapshot(run, leg)
        cached = latest_cached_stage_b_event(run, leg)
        stage_b_text = stage_b_status_text(run, leg)
        stage_b_color = status_color_from_text(stage_b_text)
        spent = snap.macro_spent_ns if snap and snap.macro_spent_ns is not None else 0.0
        budget = snap.budget_ns if snap and snap.budget_ns is not None else 1.0
        frac = 0.0 if budget <= 0 else max(0.0, min(1.0, spent / budget))
        ax.text(x + 0.02, y + 0.21, f"Stage A blocks: {len(get_leg_stage_a(run, leg))}", fontsize=11)
        ax.text(x + 0.02, y + 0.17, f"Fresh Stage B probes: {sum(1 for s in get_leg_stage_b(run, leg) if s.fresh_probe_result)}", fontsize=11)
        ax.text(x + 0.02, y + 0.13, f"Latest Stage B: {stage_b_text}", fontsize=11, color=stage_b_color)
        ax.text(x + 0.02, y + 0.09, f"Primary Stage A blocker: {stage_a_primary_blocker(snap)}", fontsize=11)
        if snap is not None:
            low_states = snap.low_occ_states or snap.tail_low_states
            if low_states:
                ax.text(x + 0.02, y + 0.05, f"Weak λ states: {', '.join('λ'+str(v) for v in low_states[:5])}", fontsize=10, color="#666666")
            elif snap.rollback:
                ax.text(x + 0.02, y + 0.05, f"Rollback state: {snap.rollback}", fontsize=10, color="#666666")
        ax.add_patch(Rectangle((x + 0.02, y + 0.025), w - 0.04, 0.022, facecolor="#edf2f7", edgecolor="#d9dce3", linewidth=0.6))
        ax.add_patch(Rectangle((x + 0.02, y + 0.025), (w - 0.04) * frac, 0.022, facecolor=LEG_COLORS[leg], edgecolor="none"))
        ax.text(x + w - 0.02, y + 0.052, f"{spent:.1f}/{budget:.1f} ns", fontsize=10, ha="right", color="#555555")

    x, y, w, h = card_positions["totals"]
    macros = max((m.macro_index for m in run.macro_starts), default=0)
    ax.text(x + 0.02, y + 0.22, f"Macros detected: {macros}", fontsize=11)
    ax.text(x + 0.02, y + 0.18, f"Optimization epochs: {len(run.optimization_epochs)}", fontsize=11)
    ax.text(x + 0.02, y + 0.14, f"Production artifacts: {len(run.production_artifacts)}", fontsize=11)
    ax.text(x + 0.02, y + 0.10, f"Fresh Stage B probes: {sum(1 for s in run.stage_b if s.fresh_probe_result)}", fontsize=11)
    ax.text(x + 0.02, y + 0.06, f"Warnings: {len(run.warnings)}", fontsize=11)
    if run.optimization_epochs:
        last = run.optimization_epochs[-1]
        last_metric = epoch_objective_value(last)
        ax.text(
            x + 0.02,
            y + 0.02,
            f"Last objective: {last_metric:.3f}" if last_metric is not None else "Last objective: -",
            fontsize=10,
            color="#555555",
        )

    x, y, w, h = card_positions["notes"]
    lines: List[str] = []
    solvent_last = latest_stage_a_snapshot(run, "solvent")
    vacuum_last = latest_stage_a_snapshot(run, "vacuum")
    if solvent_last is not None:
        lines.append(
            f"Solvent is currently limited by {stage_a_primary_blocker(solvent_last)}; latest rollback state: {solvent_last.rollback or '-'}"
        )
    if vacuum_last is not None:
        lines.append(
            f"Vacuum status is {stage_b_status_text(run, 'vacuum')} with Stage A blocker {stage_a_primary_blocker(vacuum_last)}."
        )
    if run.optimization_epochs:
        last = run.optimization_epochs[-1]
        accepted_metric = epoch_accepted_objective(last)
        lines.append(
            f"The most recent optimization epoch targeted pools '{last.block_name}' with accepted objective {accepted_metric}."
        )
    if not lines:
        lines.append("No summary interpretation available.")
    y_text = y + h - 0.07
    for line in lines:
        ax.text(x + 0.02, y_text, line, fontsize=11, va="top")
        y_text -= 0.08

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_timeline_and_performance(path: Path, run: RunRecord) -> None:
    fig, (ax_timeline, ax_perf) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2.2, 1.4]})

    legs = ["solvent", "vacuum"]
    y_map = {"solvent": 1.0, "vacuum": 0.0}
    cumulative_by_leg = {leg: 0.0 for leg in legs}
    perf_points: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    phase_segments: List[Tuple[str, str, float, float]] = []
    for event in run.phase_events:
        if event.leg not in y_map:
            continue
        if event.kind == "end" and event.md_ns is not None:
            start = cumulative_by_leg[event.leg]
            end = start + event.md_ns
            phase_segments.append((event.leg, event.phase, start, end))
            cumulative_by_leg[event.leg] = end
            if event.steps_per_s is not None:
                perf_points[event.leg].append((end, event.steps_per_s, event.phase))

    apply_axis_style(ax_timeline)
    for leg, phase, start, end in phase_segments:
        ax_timeline.broken_barh(
            [(start, max(end - start, 1e-6))],
            (y_map[leg] - 0.18, 0.36),
            facecolors=PHASE_COLORS.get(phase, "#888888"),
            edgecolors="none",
            alpha=0.95,
        )
    event_styles = {
        "probe_enter": ("^", "#f77f00", "probe"),
        "probe_continue": ("^", "#f77f00", "probe cont."),
        "cooldown": ("x", "#4d96ff", "cooldown"),
        "leg_frozen": ("|", "#2a9d8f", "frozen"),
        "probe_retry_keep": ("s", "#e76f51", "retry keep"),
        "probe_retry_grow": ("s", "#e76f51", "retry grow"),
        "probe_retry_near_pass": ("D", "#f4a261", "near pass"),
        "split_only_continue": ("P", "#7b2cbf", "split cont."),
    }
    used_labels: set = set()
    for event in run.control_events:
        if event.leg not in y_map or event.global_ns is None or event.event_type not in event_styles:
            continue
        marker, color, label = event_styles[event.event_type]
        ax_timeline.scatter(
            [event.global_ns],
            [y_map[event.leg] + 0.28],
            marker=marker,
            color=color,
            s=60,
            label=label if label not in used_labels else "",
            zorder=4,
        )
        used_labels.add(label)

    ax_timeline.set_yticks([1.0, 0.0], ["Solvent", "Vacuum"])
    max_x = max((end for _, _, _, end in phase_segments), default=1.0)
    ax_timeline.set_xlim(0.0, max_x * 1.02)
    ax_timeline.set_xlabel("Cumulative AWH MD time (ns)")
    ax_timeline.set_title("Phase Timeline And Control Events", loc="left", fontsize=15, fontweight="bold")
    phase_handles = [Patch(facecolor=color, edgecolor="none", label=phase) for phase, color in PHASE_COLORS.items()]
    event_handles = [
        Line2D([0], [0], marker=event_styles[name][0], color="w", markerfacecolor=event_styles[name][1], markeredgecolor=event_styles[name][1], linestyle="None", label=event_styles[name][2])
        for name in event_styles if event_styles[name][2] in used_labels
    ]
    ax_timeline.legend(handles=phase_handles + event_handles, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4, frameon=False, fontsize=9)

    apply_axis_style(ax_perf)
    for leg in legs:
        pts = perf_points.get(leg, [])
        if not pts:
            continue
        xs = [x for x, _, _ in pts]
        ys = [y for _, y, _ in pts]
        ax_perf.plot(xs, ys, color=LEG_COLORS[leg], linewidth=2.2, label=leg_title(leg))
        for x, y, phase in pts:
            ax_perf.scatter(x, y, color=LEG_COLORS[leg], marker=PHASE_MARKERS.get(phase, "o"), s=38, alpha=0.95)
    ax_perf.set_title("Throughput By Completed Phase", loc="left", fontsize=14, fontweight="bold")
    ax_perf.set_xlabel("Cumulative AWH MD time (ns)")
    ax_perf.set_ylabel("steps / s")
    ax_perf.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_stage_a_gate_heatmap(path: Path, run: RunRecord) -> None:
    gates = ["df", "ESS", "linN", "RT", "endpoint", "tail", "occ", "ready"]
    gate_labels = ["df", "ESS", "lin N_eff", "round trips", "endpoint", "tail", "low-occ", "stable streak"]
    cmap = mcolors.ListedColormap([GATE_PASS_COLORS[-1], GATE_PASS_COLORS[0], GATE_PASS_COLORS[1]])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    for ax, leg in zip(axes, ("solvent", "vacuum")):
        snaps = get_leg_stage_a(run, leg)
        if not snaps:
            ax.set_axis_off()
            continue
        mat = np.full((len(gates), len(snaps)), -1, dtype=float)
        for j, snap in enumerate(snaps):
            for i, gate in enumerate(gates):
                mat[i, j] = stage_a_gate_value(snap, gate)
        ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_yticks(range(len(gates)), gate_labels)
        ax.set_xticks(range(len(snaps)))
        ax.set_xticklabels([str(i + 1) for i in range(len(snaps))], fontsize=8)
        ax.set_title(f"Stage A Gate Matrix — {leg_title(leg)}", loc="left", fontsize=14, fontweight="bold")
        ax.set_ylabel("gate")
        for xpos, macro in macro_boundaries_from_stage_a(run, leg):
            ax.axvline(xpos - 0.5, color="#495057", linewidth=1.0, linestyle="--", alpha=0.8)
            ax.text(xpos - 0.35, -0.85, f"M{macro}", fontsize=8, color="#495057")
        ax.set_xlim(-0.5, len(snaps) - 0.5)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color("#d0d7de")
    axes[-1].set_xlabel("Stage A block index")
    fig.suptitle("Stage A Readiness Gates", x=0.05, ha="left", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_stage_a_metric_traces(path: Path, run: RunRecord) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    configs = [
        ("df", "df", lambda s: s.df, lambda s: s.df_ok),
        ("ESS", "ESS", lambda s: s.ess, lambda s: s.ess_ok),
        ("Endpoint band / requirement", "occupancy", None, None),
        ("Tail min / occ min", "occupancy", None, None),
    ]
    for ax, (title, ylabel, getter, ok_getter) in zip(axes, configs):
        apply_axis_style(ax)
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
        for leg in ("solvent", "vacuum"):
            snaps = get_leg_stage_a(run, leg)
            xs = list(range(1, len(snaps) + 1))
            if getter is not None:
                ys = [getter(s) for s in snaps]
                pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
                if pts:
                    ax.plot([x for x, _ in pts], [y for _, y in pts], color=LEG_COLORS[leg], linewidth=2.2, label=leg_title(leg))
                if ok_getter is not None:
                    good = [(x, getter(s)) for x, s in zip(xs, snaps) if getter(s) is not None and ok_getter(s) is True]
                    bad = [(x, getter(s)) for x, s in zip(xs, snaps) if getter(s) is not None and ok_getter(s) is False]
                    if good:
                        ax.scatter([x for x, _ in good], [y for _, y in good], color="#2a9d8f", s=24, zorder=4)
                    if bad:
                        ax.scatter([x for x, _ in bad], [y for _, y in bad], color="#d62828", s=24, zorder=4)
            elif title.startswith("Endpoint"):
                band = [(x, s.endpoint_band) for x, s in zip(xs, snaps) if s.endpoint_band is not None]
                low = [(x, s.endpoint_low) for x, s in zip(xs, snaps) if s.endpoint_low is not None]
                req = [(x, s.endpoint_required) for x, s in zip(xs, snaps) if s.endpoint_required is not None]
                if band:
                    ax.plot([x for x, _ in band], [y for _, y in band], color=LEG_COLORS[leg], linewidth=2.0, label=f"{leg_title(leg)} band")
                if low:
                    ax.plot([x for x, _ in low], [y for _, y in low], color=LEG_COLORS[leg], linewidth=1.5, linestyle="--", alpha=0.9, label=f"{leg_title(leg)} endpoint low")
                if req:
                    ax.plot([x for x, _ in req], [y for _, y in req], color=LEG_COLORS[leg], linewidth=1.0, linestyle=":", alpha=0.85, label=f"{leg_title(leg)} requirement")
            else:
                tail = [(x, s.tail_min) for x, s in zip(xs, snaps) if s.tail_min is not None]
                occ = [(x, s.occ_min) for x, s in zip(xs, snaps) if s.occ_min is not None]
                if tail:
                    ax.plot([x for x, _ in tail], [y for _, y in tail], color=LEG_COLORS[leg], linewidth=2.0, label=f"{leg_title(leg)} tail min")
                if occ:
                    ax.plot([x for x, _ in occ], [y for _, y in occ], color=LEG_COLORS[leg], linewidth=1.5, linestyle="--", alpha=0.95, label=f"{leg_title(leg)} occ min")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", ncol=3, fontsize=8, frameon=False)
    axes[-1].set_xlabel("Stage A block index")
    fig.suptitle("Stage A Metric Traces", x=0.05, ha="left", fontsize=18, fontweight="bold")
    fig.subplots_adjust(top=0.88, hspace=0.45, wspace=0.2)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_lambda_issue_heatmap(path: Path, run: RunRecord) -> None:
    n_lambda = max_lambda_state(run)
    cmap = mcolors.ListedColormap([LAMBDA_FLAG_COLORS[i] for i in range(4)])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    for ax, leg in zip(axes, ("solvent", "vacuum")):
        snaps = get_leg_stage_a(run, leg)
        mat = np.zeros((n_lambda, max(1, len(snaps))), dtype=int)
        for block_idx, snap in enumerate(snaps):
            for lam in snap.low_occ_states:
                if 1 <= lam <= n_lambda:
                    mat[n_lambda - lam, block_idx] |= 1
            for lam in snap.tail_low_states:
                if 1 <= lam <= n_lambda:
                    mat[n_lambda - lam, block_idx] |= 2
        ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_title(f"Problematic λ-State Incidence — {leg_title(leg)}", loc="left", fontsize=14, fontweight="bold")
        ax.set_ylabel("λ state")
        ax.set_yticks(np.linspace(0, n_lambda - 1, min(n_lambda, 8)).astype(int))
        ax.set_yticklabels([str(n_lambda - int(v)) for v in np.linspace(0, n_lambda - 1, min(n_lambda, 8)).astype(int)])
        ax.set_xticks(range(len(snaps)))
        ax.set_xticklabels([str(i + 1) for i in range(len(snaps))], fontsize=8)
        for xpos, macro in macro_boundaries_from_stage_a(run, leg):
            ax.axvline(xpos - 0.5, color="#495057", linewidth=1.0, linestyle="--", alpha=0.8)
            ax.text(xpos - 0.35, -0.85, f"M{macro}", fontsize=8, color="#495057")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color("#d0d7de")
    axes[-1].set_xlabel("Stage A block index")
    handles = [
        Patch(facecolor=LAMBDA_FLAG_COLORS[0], edgecolor="none", label="none"),
        Patch(facecolor=LAMBDA_FLAG_COLORS[1], edgecolor="none", label="low_occ"),
        Patch(facecolor=LAMBDA_FLAG_COLORS[2], edgecolor="none", label="tail_low"),
        Patch(facecolor=LAMBDA_FLAG_COLORS[3], edgecolor="none", label="both"),
    ]
    axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=False)
    fig.suptitle("Where Stage A Is Struggling", x=0.05, ha="left", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def stage_b_state_strip(run: RunRecord, leg: str) -> np.ndarray:
    blocks = len(get_leg_stage_a(run, leg))
    state = np.zeros(max(1, blocks), dtype=int)
    lines = block_line_numbers(run, leg)
    if not lines:
        return state
    cached_events = [ev for ev in run.control_events if ev.leg == leg and ev.event_type == "stage_b_cached"]
    cooldown_events = [ev for ev in run.control_events if ev.leg == leg and ev.event_type == "cooldown"]
    for ev in cached_events:
        idx = block_index_for_line(run, leg, ev.line_no)
        if idx is None:
            continue
        failure = str(ev.details.get("failure", ""))
        state[idx] = STAGE_B_STATE_ORDER.index("cached_pass" if failure == "passed" else "cached_fail")
    for ev in cooldown_events:
        idx = block_index_for_line(run, leg, ev.line_no)
        if idx is None:
            continue
        state[idx] = STAGE_B_STATE_ORDER.index("cooldown")
    for snap in get_leg_stage_b(run, leg):
        idx = block_index_for_line(run, leg, snap.line_no)
        if idx is None:
            continue
        state[idx] = STAGE_B_STATE_ORDER.index("fresh_pass" if (snap.failure_mode == "passed") else "fresh_fail")
    return state.reshape(1, -1)


def write_stage_b_decisions(path: Path, run: RunRecord) -> None:
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.35, 1.75, 0.35, 1.75], hspace=0.45, wspace=0.2)
    cmap = mcolors.ListedColormap([STAGE_B_STATE_COLORS[name] for name in STAGE_B_STATE_ORDER])
    bounds = np.arange(len(STAGE_B_STATE_ORDER) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for col, leg in enumerate(("solvent", "vacuum")):
        ax_strip = fig.add_subplot(gs[0, col])
        strip = stage_b_state_strip(run, leg)
        ax_strip.imshow(strip, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax_strip.set_title(f"Stage B decision strip — {leg_title(leg)}", loc="left", fontsize=13, fontweight="bold")
        ax_strip.set_yticks([])
        ax_strip.set_xticks(range(strip.shape[1]))
        ax_strip.set_xticklabels([str(i + 1) for i in range(strip.shape[1])], fontsize=8)
        ax_strip.set_xlabel("Stage A block index")
        for xpos, macro in macro_boundaries_from_stage_a(run, leg):
            ax_strip.axvline(xpos - 0.5, color="#495057", linewidth=1.0, linestyle="--", alpha=0.8)
            ax_strip.text(xpos - 0.35, -0.8, f"M{macro}", fontsize=8, color="#495057")
        for spine in ax_strip.spines.values():
            spine.set_color("#d0d7de")

        ax = fig.add_subplot(gs[1, col])
        apply_axis_style(ax)
        fresh = [snap for snap in get_leg_stage_b(run, leg) if snap.fresh_probe_result and snap.attempt_index is not None]
        if fresh:
            split = finite_xy([(snap.attempt_index, snap.split_gap) for snap in fresh])
            parity = finite_xy([(snap.attempt_index, snap.parity_gap) for snap in fresh])
            endpoint = finite_xy([(snap.attempt_index, snap.endpoint_parity_gap) for snap in fresh])
            support = finite_xy([
                (snap.attempt_index, (snap.n_supported_states / snap.n_states) if snap.n_supported_states is not None and snap.n_states else None)
                for snap in fresh
            ])
            if split:
                ax.plot([x for x, _ in split], [y for _, y in split], color=METRIC_COLORS["split_gap"], linewidth=2.1, label="split gap")
            if parity:
                ax.plot([x for x, _ in parity], [y for _, y in parity], color=METRIC_COLORS["parity_gap"], linewidth=2.1, label="parity gap")
            if endpoint:
                ax.plot([x for x, _ in endpoint], [y for _, y in endpoint], color=METRIC_COLORS["endpoint_parity_gap"], linewidth=1.8, linestyle="--", label="endpoint parity")
            for snap in fresh:
                if snap.attempt_index is None or snap.parity_gap is None:
                    continue
                color = FAILURE_COLORS.get(snap.failure_mode or "not_checked", "#888888")
                ax.scatter([snap.attempt_index], [snap.parity_gap], color=color, s=45, zorder=4)
            ax2 = ax.twinx()
            ax2.grid(False)
            if support:
                ax2.plot([x for x, _ in support], [y for _, y in support], color=METRIC_COLORS["support_fraction"], linewidth=1.6, alpha=0.9, label="support frac")
                ax2.set_ylim(0.0, 1.05)
                ax2.set_ylabel("support fraction")
            else:
                ax2.set_yticks([])
            ax.set_xlabel("Fresh Stage B probe attempt")
            ax.set_ylabel("gap (kT)")
            ax.set_title(f"Fresh Stage B probes — {leg_title(leg)}", loc="left", fontsize=13, fontweight="bold")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=False, fontsize=8)
        else:
            ax.text(0.5, 0.5, "No fresh Stage B probes", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

        ax3 = fig.add_subplot(gs[3, col])
        apply_axis_style(ax3)
        fresh_frames = finite_xy([(snap.attempt_index, float(snap.frames)) for snap in fresh if snap.attempt_index is not None and snap.frames is not None])
        probe_size_points: List[Tuple[float, float]] = []
        c = 0
        for ev in run.control_events:
            if ev.leg != leg or ev.event_type not in {"probe_enter", "probe_continue", "probe_retry_keep", "probe_retry_grow", "probe_retry_near_pass", "split_only_continue"}:
                continue
            probe_steps = ev.details.get("probe_steps") or ev.details.get("next_probe_steps")
            if isinstance(probe_steps, (int, float)):
                c += 1
                probe_size_points.append((float(c), float(probe_steps)))
        if fresh_frames:
            ax3.plot([x for x, _ in fresh_frames], [y for _, y in fresh_frames], color=METRIC_COLORS["frames"], linewidth=2.1, label="retained frames")
        if probe_size_points:
            ax3_t = ax3.twinx()
            ax3_t.grid(False)
            ax3_t.plot([x for x, _ in probe_size_points], [y for _, y in probe_size_points], color=METRIC_COLORS["probe_ns"], linewidth=1.8, linestyle="--", label="probe steps")
            ax3_t.set_ylabel("probe steps")
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_t.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=False, fontsize=8)
        ax3.set_title(f"Frame retention and probe growth — {leg_title(leg)}", loc="left", fontsize=13, fontweight="bold")
        ax3.set_xlabel("Attempt / control-event index")
        ax3.set_ylabel("retained frames")

    handles = [Patch(facecolor=STAGE_B_STATE_COLORS[name], edgecolor="none", label=name.replace("_", " ")) for name in STAGE_B_STATE_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=9)
    fig.suptitle("Stage B Decision History", x=0.05, ha="left", fontsize=18, fontweight="bold")
    fig.subplots_adjust(top=0.88, hspace=0.45, wspace=0.2)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_optimization_panels(path: Path, run: RunRecord) -> None:
    epochs = run.optimization_epochs
    if not epochs:
        return
    boundaries = macro_boundaries_from_epochs(run)
    target_names = collect_target_names(epochs)
    has_target_panels = bool(target_names)
    has_objectives = any(e.objective_value is not None for e in epochs)
    nrows = 4 if has_target_panels else 3
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4 * nrows), sharex=True)
    axes = axes.ravel()

    def add_boundaries(ax: plt.Axes) -> None:
        for v in boundaries:
            ax.axvline(v.value, color=v.color, linestyle="--", linewidth=1.0, alpha=v.alpha)
            if v.label:
                ax.text(v.value, 0.98, v.label, transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=8, color=v.color)

    # Map macro -> x-position of the pre-optimization snapshot for that macro.
    macro_first_epoch: Dict[int, int] = {}
    for e in epochs:
        if e.macro_index not in macro_first_epoch:
            macro_first_epoch[e.macro_index] = e.global_epoch_index
    init_x = {macro: idx - 1 for macro, idx in macro_first_epoch.items()}

    x = [e.global_epoch_index for e in epochs]
    objective_panel = [
        (
            "Objective" if has_objectives else "Residual",
            [epoch_objective_value(e) for e in epochs] if has_objectives else [e.residual for e in epochs],
            METRIC_COLORS["objective"] if has_objectives else METRIC_COLORS["residual"],
        ),
        (
            "Accepted objective" if has_objectives else "Accepted",
            [epoch_accepted_objective(e) for e in epochs] if has_objectives else [e.accepted_residual for e in epochs],
            METRIC_COLORS["accepted_objective"] if has_objectives else METRIC_COLORS["accepted_residual"],
        ),
    ]
    if has_objectives:
        objective_panel.extend(
            [
                ("Threshold", [e.objective_threshold for e in epochs], METRIC_COLORS["objective_threshold"]),
            ]
        )
    else:
        objective_panel.extend(
            [
                ("Prediction", [e.prediction for e in epochs], METRIC_COLORS["prediction"]),
                ("Target", [e.target for e in epochs], METRIC_COLORS["target"]),
            ]
        )

    plots = [
        (axes[0], "Objective" if has_objectives else "Residuals", "loss" if has_objectives else "kT", objective_panel),
        (axes[1], "Trust region", "KL / alpha / step", [
            ("Est. KL", [e.kl_est for e in epochs], METRIC_COLORS["kl_est"]),
            ("KL scaling", [e.kl_scaling for e in epochs], METRIC_COLORS["kl_scaling"]),
            ("Alpha", [e.line_search_alpha for e in epochs], METRIC_COLORS["line_search_alpha"]),
            ("Max φ step", [e.actual_max_phi_step for e in epochs], METRIC_COLORS["max_phi_step"]),
        ]),
        (axes[2], "Gradients", "norm", [
            ("Grad norm", [e.grad_norm for e in epochs], METRIC_COLORS["grad_norm"]),
            ("Grad max", [e.grad_max for e in epochs], METRIC_COLORS["grad_max"]),
        ]),
        (axes[3], "Fisher conditioning", "log10(cond) / trunc", [
            ("log10(cond)", [safe_log10(e.fim_raw_cond) for e in epochs], METRIC_COLORS["fim_cond"]),
            ("Truncated eigs", [float(e.truncated_eigs) if e.truncated_eigs is not None else None for e in epochs], METRIC_COLORS["truncated_eigs"]),
        ]),
        (axes[4], "Per-leg dG", "kT", [
            ("Solvent dG", [e.leg_metrics.get("solvent", {}).get("dG") for e in epochs], LEG_COLORS["solvent"]),
            ("Vacuum dG", [e.leg_metrics.get("vacuum", {}).get("dG") for e in epochs], LEG_COLORS["vacuum"]),
        ]),
        (axes[5], "Per-leg support", "ESS / log10(N_active)", [
            ("Solvent ESS", [e.leg_metrics.get("solvent", {}).get("ESS") for e in epochs], LEG_COLORS["solvent"]),
            ("Vacuum ESS", [e.leg_metrics.get("vacuum", {}).get("ESS") for e in epochs], LEG_COLORS["vacuum"]),
            ("Solvent log10(N_active)", [safe_log10(e.leg_metrics.get("solvent", {}).get("N_active")) for e in epochs], "#b56576"),
            ("Vacuum log10(N_active)", [safe_log10(e.leg_metrics.get("vacuum", {}).get("N_active")) for e in epochs], "#6d597a"),
        ]),
    ]
    if has_target_panels:
        target_colors = {name: TARGET_PALETTE[idx % len(TARGET_PALETTE)] for idx, name in enumerate(target_names)}
        plots.extend(
            [
                (axes[6], "Per-target loss", "loss", [
                    (f"{name} loss", [e.target_metrics.get(name, {}).get("loss") for e in epochs], target_colors[name])
                    for name in target_names
                ]),
                (axes[7], "Per-target support", "ESS", [
                    (f"{name} ESS", [e.target_metrics.get(name, {}).get("ESS") for e in epochs], target_colors[name])
                    for name in target_names
                ]),
            ]
        )

    for ax, title, ylabel, series_list in plots:
        apply_axis_style(ax)
        for name, values, color in series_list:
            pts = finite_xy(list(zip(x, values)))
            if pts:
                linestyle = "--" if name in {"Prediction", "Target", "Threshold", "Solvent log10(N_active)", "Vacuum log10(N_active)"} else "-"
                ax.plot([a for a, _ in pts], [b for _, b in pts], color=color, linewidth=2.0, linestyle=linestyle, label=name)
        add_boundaries(ax)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left", fontsize=8, frameon=False)

    # Overlay each macro's pre-optimization prediction and residual as star markers.
    labelled_init = False
    for macro, (pred, target, residual) in run.pre_opt_prediction_per_macro.items():
        xi = init_x.get(macro)
        if xi is None:
            continue
        label_pred = "Pre-opt prediction" if not labelled_init else None
        label_res = "Pre-opt residual" if not labelled_init else None
        labelled_init = True
        axes[0].scatter([xi], [pred], color=METRIC_COLORS["prediction"], marker="*", s=120, zorder=5, label=label_pred)
        axes[0].scatter([xi], [residual], color=METRIC_COLORS["residual"], marker="*", s=120, zorder=5, label=label_res)
    if labelled_init:
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, loc="upper left", fontsize=8, frameon=False)

    axes[-2].set_xlabel("Global optimization epoch")
    axes[-1].set_xlabel("Global optimization epoch")
    fig.suptitle("Optimization Diagnostics", x=0.05, ha="left", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _pool_name_of(param_name: str) -> str:
    raw = param_name[len("derived_charge:"):] if param_name.startswith("derived_charge:") else param_name
    if raw.startswith("pool_"):
        rest = raw[len("pool_"):]
        for sep in ("_atom_", "_charge_"):
            idx = rest.find(sep)
            if idx != -1:
                return rest[:idx]
    return ""


def _parameter_category_of(param_name: str) -> str:
    if param_name.startswith("derived_charge:"):
        return "charge"
    if param_name.endswith("_σ"):
        return "sigma"
    if param_name.endswith("_ϵ"):
        return "epsilon"
    if param_name.endswith("_χ"):
        return "chi"
    if param_name.endswith("_η"):
        return "eta"
    return "other"


def _parameter_target_of(param_name: str) -> str:
    raw = param_name[len("derived_charge:"):] if param_name.startswith("derived_charge:") else param_name
    if raw.startswith("pool_"):
        rest = raw[len("pool_"):]
        for sep in ("_atom_", "_charge_"):
            idx = rest.find(sep)
            if idx != -1:
                tail = rest[idx + len(sep):]
                for suffix in ("_σ", "_ϵ", "_χ", "_η"):
                    if tail.endswith(suffix):
                        tail = tail[: -len(suffix)]
                        break
                return tail
    return raw


def _pretty_parameter_label(param_name: str, derived_charges: Dict[str, Tuple[str, str, str]]) -> str:
    if param_name in derived_charges:
        raw_label = derived_charges[param_name][0]
        return raw_label
    target = _parameter_target_of(param_name)
    category = _parameter_category_of(param_name)
    suffix_map = {
        "sigma": "σ",
        "epsilon": "ϵ",
        "chi": "χ",
        "eta": "η",
        "charge": "q",
        "other": "",
    }
    suffix = suffix_map.get(category, "")
    return f"{target} ({suffix})" if suffix else target


def _parameter_category_title(category: str) -> str:
    return {
        "sigma": "LJ σ",
        "epsilon": "LJ ϵ",
        "charge": "Derived charges",
        "chi": "Charge χ",
        "eta": "Charge η",
        "other": "Other parameters",
    }.get(category, category)


def _parameter_value_label(category: str) -> str:
    return {
        "sigma": "σ",
        "epsilon": "ϵ",
        "charge": "q",
        "chi": "χ",
        "eta": "η",
        "other": "value",
    }.get(category, "value")


def _sort_series_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return sorted(points, key=lambda item: (item[0], item[1]))


def parameter_panels(run: RunRecord) -> List[Tuple[str, str, List[PanelSpec]]]:
    epochs = run.optimization_epochs
    if not epochs:
        return []

    # Place each pre-optimization snapshot one tick before that macro's first optimization epoch.
    macro_first_epoch: Dict[int, int] = {}
    for e in epochs:
        if e.macro_index not in macro_first_epoch:
            macro_first_epoch[e.macro_index] = e.global_epoch_index
    pre_opt_x: Dict[int, int] = {macro: idx - 1 for macro, idx in macro_first_epoch.items()}

    param_names: List[str] = []
    for epoch in epochs:
        for name in epoch.parameters:
            if name not in param_names:
                param_names.append(name)
    for params in run.pre_opt_params_per_macro.values():
        for name in params:
            if name not in param_names:
                param_names.append(name)

    derived_charges: Dict[str, Tuple[str, str, str]] = {}
    for name in param_names:
        if name.endswith("_χ"):
            eta_key = name[:-1] + "η"
            if eta_key in param_names:
                label = name
                if label.startswith("pool_"):
                    label = label[len("pool_"):]
                if label.endswith("_χ"):
                    label = label[:-2]
                label = label.replace("_charge_", "/") + " (q)"
                derived_charges[f"derived_charge:{name}"] = (label, name, eta_key)

    all_names = param_names + list(derived_charges.keys())
    pools_ordered: List[str] = []
    by_pool: Dict[str, List[str]] = {}
    for name in all_names:
        pool = _pool_name_of(name)
        if pool not in by_pool:
            pools_ordered.append(pool)
            by_pool[pool] = []
        by_pool[pool].append(name)

    palette = [
        "#c2552d", "#2d6db6", "#3a7d44", "#7b2cbf", "#d17b0f", "#ef476f", "#118ab2", "#6d597a", "#4c956c", "#264653",
    ]
    category_order = ["sigma", "epsilon", "charge", "chi", "eta", "other"]
    macro_boundaries = macro_boundaries_from_epochs(run)
    panels_by_file: List[Tuple[str, str, List[PanelSpec]]] = []

    for pool in pools_ordered:
        pool_names = by_pool[pool]
        by_category: Dict[str, List[str]] = {category: [] for category in category_order}
        for name in pool_names:
            by_category.setdefault(_parameter_category_of(name), []).append(name)

        panels: List[PanelSpec] = []
        for category in category_order:
            names = by_category.get(category, [])
            if not names:
                continue

            absolute_series: List[SeriesSpec] = []
            drift_series: List[SeriesSpec] = []
            zero_baseline_labels: List[str] = []
            for idx, name in enumerate(sorted(names, key=lambda item: (_parameter_target_of(item), item))):
                color = palette[idx % len(palette)]
                if name in derived_charges:
                    label, chi_key, eta_key = derived_charges[name]
                    raw_points: List[Tuple[float, float]] = []
                    for macro, params in sorted(run.pre_opt_params_per_macro.items()):
                        chi = params.get(chi_key)
                        eta = params.get(eta_key)
                        xi = pre_opt_x.get(macro)
                        if chi is not None and eta is not None and eta != 0.0 and xi is not None:
                            value = -chi / eta
                            raw_points.append((xi, value))
                    for epoch in epochs:
                        chi = epoch.parameters.get(chi_key)
                        eta = epoch.parameters.get(eta_key)
                        if chi is not None and eta is not None and eta != 0.0:
                            raw_points.append((epoch.global_epoch_index, -chi / eta))
                else:
                    raw_points = []
                    for macro, params in sorted(run.pre_opt_params_per_macro.items()):
                        val = params.get(name)
                        xi = pre_opt_x.get(macro)
                        if val is not None and xi is not None:
                            raw_points.append((xi, val))
                    raw_points += [
                        (epoch.global_epoch_index, epoch.parameters[name])
                        for epoch in epochs
                        if name in epoch.parameters and epoch.parameters[name] is not None
                    ]
                    label = _pretty_parameter_label(name, derived_charges)
                if not raw_points:
                    continue
                raw_points = _sort_series_points(raw_points)
                absolute_series.append(SeriesSpec(label, raw_points, color, stroke_width=2.0))

                baseline = raw_points[0][1]
                if abs(baseline) > 1e-12:
                    drift_points = [(x, 100.0 * (y - baseline) / baseline) for x, y in raw_points]
                    drift_label = label
                else:
                    drift_points = [(x, y - baseline) for x, y in raw_points]
                    drift_label = label + " (Δ)"
                    zero_baseline_labels.append(label)
                drift_series.append(SeriesSpec(drift_label, drift_points, color, stroke_width=2.0))

            if not absolute_series:
                continue

            category_title = _parameter_category_title(category)
            value_label = _parameter_value_label(category)
            abs_notes = [
                f"Absolute parameter values for {category_title.lower()}.",
                "Pre-optimization snapshots are shown once per macro, immediately before that macro's optimization epochs.",
            ]
            drift_notes = ["Relative drift with respect to the original pre-optimization snapshot."]
            if zero_baseline_labels:
                joined = ", ".join(zero_baseline_labels[:4])
                if len(zero_baseline_labels) > 4:
                    joined += ", …"
                drift_notes.append(f"Zero-baseline series shown as absolute Δ: {joined}.")

            panels.append(
                PanelSpec(
                    title=f"{category_title} — actual values",
                    x_label="Global optimization epoch",
                    y_label=value_label,
                    series=absolute_series,
                    vlines=macro_boundaries,
                    notes=abs_notes,
                )
            )
            panels.append(
                PanelSpec(
                    title=f"{category_title} — relative drift",
                    x_label="Global optimization epoch",
                    y_label="drift (%) / Δ",
                    series=drift_series,
                    vlines=macro_boundaries,
                    notes=drift_notes,
                )
            )

        if not panels:
            continue
        pool_slug = pool if pool else "other"
        file_title = f"Parameter trajectories — {pool}" if pool else "Parameter trajectories"
        panels_by_file.append((f"parameter_trajectories_{pool_slug}.png", file_title, panels))
    return panels_by_file


def compute_range(panel: PanelSpec) -> Tuple[float, float]:
    all_values: List[float] = []
    for series in panel.series:
        all_values.extend(y for _, y in series.points if math.isfinite(y))
    all_values.extend(line.value for line in panel.hlines if math.isfinite(line.value))
    all_values.extend(marker.y for marker in panel.markers if math.isfinite(marker.y))
    if panel.y_min is not None:
        all_values.append(panel.y_min)
    if panel.y_max is not None:
        all_values.append(panel.y_max)
    if not all_values:
        return 0.0, 1.0
    y_min = min(all_values) if panel.y_min is None else panel.y_min
    y_max = max(all_values) if panel.y_max is None else panel.y_max
    if y_min == y_max:
        pad = abs(y_min) * 0.1 or 1.0
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad
    return y_min, y_max


def compute_x_range(panel: PanelSpec) -> Tuple[float, float]:
    all_values: List[float] = []
    for series in panel.series:
        all_values.extend(x for x, _ in series.points if math.isfinite(x))
    all_values.extend(line.value for line in panel.vlines if math.isfinite(line.value))
    all_values.extend(marker.x for marker in panel.markers if math.isfinite(marker.x))
    if not all_values:
        return 0.0, 1.0
    x_min = min(all_values)
    x_max = max(all_values)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    return x_min, x_max


def write_multiplot(path: Path, title: str, panels: Sequence[PanelSpec], width: int = 1400, panel_height: int = 260) -> None:
    fig_height = max(3.0, len(panels) * (panel_height / 100.0) + 0.8)
    fig, axes = plt.subplots(len(panels), 1, figsize=(width / 100.0, fig_height), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]
    fig.suptitle(title, fontsize=18, fontweight="bold", x=0.05, ha="left")
    for ax, panel in zip(axes_flat, panels):
        apply_axis_style(ax)
        x_min, x_max = compute_x_range(panel)
        y_min, y_max = compute_range(panel)
        ax.set_title(panel.title, fontsize=13, fontweight="bold", loc="left")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        for hline in panel.hlines:
            ax.axhline(hline.value, color=hline.color, linewidth=1.3, linestyle="--" if hline.dashed else "-", alpha=hline.alpha)
            if hline.label:
                ax.text(0.995, hline.value, hline.label, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color=hline.color)
        for vline in panel.vlines:
            ax.axvline(vline.value, color=vline.color, linewidth=1.2, linestyle="--" if vline.dashed else "-", alpha=vline.alpha)
            if vline.label:
                ax.text(vline.value, 0.98, vline.label, transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=8, color=vline.color)
        for series in panel.series:
            pts = finite_xy(series.points)
            if not pts:
                continue
            ax.plot(
                [x for x, _ in pts],
                [y for _, y in pts],
                label=series.name,
                color=series.color,
                linewidth=series.stroke_width,
                linestyle="--" if series.dashed else "-",
                marker="o" if series.draw_markers else None,
                markersize=3.0 if series.draw_markers else None,
                alpha=series.alpha,
            )
        for marker in panel.markers:
            if math.isfinite(marker.x) and math.isfinite(marker.y):
                ax.scatter([marker.x], [marker.y], s=max(18.0, marker.radius * marker.radius * 2.4), color=marker.color, marker=marker.marker, edgecolors="#ffffff", linewidths=0.8, zorder=4)
                if marker.label:
                    ax.annotate(marker.label, (marker.x, marker.y), textcoords="offset points", xytext=(5, 6), fontsize=8, color=marker.color)
        ax.set_xlabel(panel.x_label)
        ax.set_ylabel(panel.y_label)
        if panel.legend and panel.series:
            ax.legend(loc="upper left", fontsize=8, ncol=min(4, max(1, len(panel.series))), frameon=False)
        if panel.notes:
            ax.text(
                0.995,
                0.02,
                "\n".join(panel.notes),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#666666",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#dddddd", alpha=0.8),
            )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report_text(run: RunRecord) -> str:
    lines = [
        f"log_path: {run.log_path}",
        f"macros_detected: {max((macro.macro_index for macro in run.macro_starts), default=0)}",
        f"stage_a_snapshots: {len(run.stage_a)}",
        f"stage_b_snapshots: {len(run.stage_b)}",
        f"fresh_stage_b_probes: {sum(1 for snap in run.stage_b if snap.fresh_probe_result)}",
        f"optimization_epochs: {len(run.optimization_epochs)}",
        f"production_artifacts: {len(run.production_artifacts)}",
    ]
    for leg in ("solvent", "vacuum"):
        stage_a = get_leg_stage_a(run, leg)
        snap = latest_stage_a_snapshot(run, leg)
        lines.append(f"{leg}: stage_a_blocks={len(stage_a)} stage_b_status={stage_b_status_text(run, leg)}")
        if snap is not None:
            lines.append(
                f"  spent={snap.macro_spent_ns}/{snap.budget_ns} ns blocker={stage_a_primary_blocker(snap)} rollback={snap.rollback} streak={snap.streak} cooldown={snap.cooldown}"
            )
        sb = latest_stage_b_snapshot(run, leg)
        if sb is not None:
            lines.append(
                f"  latest_stage_b: failure={sb.failure_mode} split_gap={sb.split_gap} parity_gap={sb.parity_gap} support={sb.n_supported_states}/{sb.n_states} mode={sb.accumulation_mode}"
            )
        cached = latest_cached_stage_b_event(run, leg)
        if cached is not None:
            lines.append(
                f"  latest_cached_stage_b: failure={cached.details.get('failure')} age_blocks={cached.details.get('age_blocks')} cooldown={cached.details.get('cooldown')} streak={cached.details.get('streak')}"
            )
    if run.optimization_epochs:
        last_epoch = run.optimization_epochs[-1]
        final_metric = epoch_objective_value(last_epoch)
        final_accepted = epoch_accepted_objective(last_epoch)
        lines.append(
            f"final_optimization_epoch: global={last_epoch.global_epoch_index} macro={last_epoch.macro_index} objective={final_metric} accepted_objective={final_accepted}"
        )
        lines.append(f"tracked_parameters: {len(last_epoch.parameters)}")
    if run.warnings:
        lines.append("notes:")
        lines.extend(f"  - {warning}" for warning in run.warnings)
    return "\n".join(lines)


def write_index_html(outdir: Path, stem: str, figure_names: Sequence[str], report_txt: str, run: Optional[RunRecord] = None) -> None:
    figure_labels = {
        "dashboard.png": "Dashboard",
        "timeline_and_performance.png": "Timeline and throughput",
        "stage_a_gates.png": "Stage A gate heatmaps",
        "stage_a_metrics.png": "Stage A raw metric traces",
        "stage_a_lambda_issues.png": "Problematic λ-state incidence",
        "stage_b_decisions.png": "Stage B decisions",
        "optimization_metrics.png": "Optimization diagnostics",
    }
    blocks = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8' />",
        f"<title>{svg_escape(stem)} progress report</title>",
        "<style>body{font-family:Inter,Segoe UI,Arial,sans-serif;margin:0;background:#f6f8fb;color:#1f2937}main{max-width:1200px;margin:0 auto;padding:28px}h1{margin:0 0 6px 0}p.meta{color:#667085;margin-top:0}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:18px 0 26px 0}.card{background:#fff;border:1px solid #d9dee8;border-radius:12px;padding:16px}.card h3{margin:0 0 8px 0;font-size:15px}.small{font-size:13px;color:#667085}.figure{background:#fff;border:1px solid #d9dee8;border-radius:12px;padding:16px;margin:18px 0}.figure img{max-width:100%;display:block;border:1px solid #e5e7eb;background:#fff}.summary{white-space:pre-wrap;background:#fff;border:1px solid #d9dee8;border-radius:12px;padding:16px;overflow:auto}</style>",
        "</head><body><main>",
        f"<h1>{svg_escape(stem)} progress report</h1>",
        f"<p class='meta'>{svg_escape(Path(stem).name)}</p>",
        "<div class='cards'>",
        f"<div class='card'><h3>Artifacts</h3><div class='small'>{len(figure_names)} figures<br>{svg_escape(Path(outdir).name)} directory</div></div>",
        f"<div class='card'><h3>Stage A snapshots</h3><div class='small'>{len(run.stage_a) if run is not None else '-'} total<br>{len(run.stage_b) if run is not None else '-'} Stage B snapshots</div></div>",
        "<div class='card'><h3>Summary files</h3><div class='small'>summary.json<br>report.txt</div></div>",
        "</div>",
        "<h2>Summary</h2>",
        f"<div class='summary'>{svg_escape(report_txt)}</div>",
        "<h2>Figures</h2>",
    ]
    for name in figure_names:
        label = figure_labels.get(name, name)
        if name.startswith("parameter_trajectories_"):
            pool_name = name[len("parameter_trajectories_"):-len(".png")].replace("_", " ")
            label = f"Parameter trajectories — {pool_name}"
        blocks.append("<section class='figure'>")
        blocks.append(f"<h3>{svg_escape(label)}</h3>")
        blocks.append(f"<img src='{svg_escape(name)}' alt='{svg_escape(name)}' />")
        blocks.append("</section>")
    blocks.append("</main></body></html>")
    (outdir / "index.html").write_text("\n".join(blocks), encoding="utf-8")


def write_report(outdir: Path, run: RunRecord) -> List[str]:
    figure_names: List[str] = []
    write_summary_json(outdir, run)
    report_txt = build_report_text(run)
    (outdir / "report.txt").write_text(report_txt + "\n", encoding="utf-8")

    dashboard_name = "dashboard.png"
    write_dashboard(outdir / dashboard_name, run)
    figure_names.append(dashboard_name)

    timeline_name = "timeline_and_performance.png"
    write_timeline_and_performance(outdir / timeline_name, run)
    figure_names.append(timeline_name)

    stage_a_gates_name = "stage_a_gates.png"
    write_stage_a_gate_heatmap(outdir / stage_a_gates_name, run)
    figure_names.append(stage_a_gates_name)

    stage_a_metrics_name = "stage_a_metrics.png"
    write_stage_a_metric_traces(outdir / stage_a_metrics_name, run)
    figure_names.append(stage_a_metrics_name)

    lambda_issue_name = "stage_a_lambda_issues.png"
    write_lambda_issue_heatmap(outdir / lambda_issue_name, run)
    figure_names.append(lambda_issue_name)

    stage_b_name = "stage_b_decisions.png"
    write_stage_b_decisions(outdir / stage_b_name, run)
    figure_names.append(stage_b_name)

    if run.optimization_epochs:
        opt_name = "optimization_metrics.png"
        write_optimization_panels(outdir / opt_name, run)
        figure_names.append(opt_name)

    for name, figure_title, panels in parameter_panels(run):
        write_multiplot(outdir / name, figure_title, panels, panel_height=245)
        figure_names.append(name)

    write_index_html(outdir, Path(run.log_path).stem, figure_names, report_txt, run)
    return figure_names


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="One or more log files to parse.")
    parser.add_argument("--output-dir", default="plots", help="Base output directory. Default: plots")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing per-log output directory.")
    parser.add_argument("--summary-only", action="store_true", help="Parse logs and write summary files without rendering figures.")
    return parser.parse_args(argv)


def ensure_outdir(base_dir: Path, log_path: Path, overwrite: bool) -> Path:
    outdir = base_dir / safe_slug(log_path.stem)
    if outdir.exists() and any(outdir.iterdir()) and not overwrite:
        raise SystemExit(
            f"Output directory {outdir} already exists and is not empty. Pass --overwrite to reuse it."
        )
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    base_dir = Path(args.output_dir)
    wrote_any = False
    for raw_path in args.logs:
        log_path = Path(raw_path)
        if not log_path.is_file():
            print(f"[warn] missing log file: {log_path}", file=sys.stderr)
            continue
        parser = LogParser(log_path)
        run = parser.parse()
        outdir = ensure_outdir(base_dir, log_path, args.overwrite)
        write_summary_json(outdir, run)
        report_txt = build_report_text(run)
        (outdir / "report.txt").write_text(report_txt + "\n", encoding="utf-8")
        figure_names: List[str] = []
        if not args.summary_only:
            figure_names = write_report(outdir, run)
        else:
            write_index_html(outdir, log_path.stem, figure_names, report_txt, run)
        wrote_any = True
        print(f"[ok] parsed {log_path} -> {outdir}")
        print(f"     stage_a={len(run.stage_a)} stage_b={len(run.stage_b)} optimization_epochs={len(run.optimization_epochs)}")
        print(f"     figure_backend=matplotlib/png")
        if figure_names:
            print(f"     figures={len(figure_names)} index={outdir / 'index.html'}")
        if run.warnings:
            print(f"     notes={len(run.warnings)}")
    if not wrote_any:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
