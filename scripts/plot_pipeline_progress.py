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

HAS_MATPLOTLIB = False
MATPLOTLIB_IMPORT_ERROR: Optional[Exception] = None
plt = None
Line2D = None
Patch = None

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / "awhgrads_mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from matplotlib.lines import Line2D as _Line2D
    from matplotlib.patches import Patch as _Patch

    plt = _plt
    Line2D = _Line2D
    Patch = _Patch
    HAS_MATPLOTLIB = True
except Exception as exc:  # pragma: no cover - exercised only in fallback envs.
    MATPLOTLIB_IMPORT_ERROR = exc


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
    ls_iter_re = re.compile(r"LS Iter (\d+) \(.*?=([-\deE+.]+)\): ESS\[(.*?)\](?: \| Drift\[(.*?)\])? \| Res = ([-\deE+.]+)")
    opt_metrics_re = re.compile(r"--- Optimization Metrics \(Epoch (\d+) - (?:Block:\s*)?(.+)\) ---")
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
    kl_re = re.compile(r"KL Bound:\s+Est\. KL\s*=\s*([-\deE+.]+)\s+\|\s+Target\s*=\s*([-\deE+.]+)\s+\|\s+Scaling\s*=\s*([-\deE+.]+)")
    alpha_re = re.compile(r"Line Search:\s+Converged .*?=\s*([-\deE+.]+)")
    step_re = re.compile(r"Actual Step:\s+Max .*?=\s*([-\deE+.]+)\s+\(.*?=([-\deE+.]+)\)")
    params_re = re.compile(r"Params .*?Min\s*=\s*([-\deE+.]+)\s+\|\s+Max\s*=\s*([-\deE+.]+)")
    leg_metric_re = re.compile(
        r"Leg ([^:]+): coeff=([-\deE+.]+) \| .*?=([-\deE+.]+) kT \| ESS=([-\deE+.]+) / ([-\deE+.]+) \| N_active=([-\deE+.]+)"
    )
    restored_re = re.compile(r"Restored best inner-loop state from Epoch (\d+) \(Residual = ([-\deE+.]+)\)")
    solvent_invariant_re = re.compile(r"Solvent Invariant: max \|.*?\| = ([-\deE+.]+)")

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

        opt_epoch_match = self.opt_epoch_re.search(message)
        if opt_epoch_match:
            self._finalize_current_opt_epoch()
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
                    self.current_opt_epoch.accepted_residual = self.current_opt_epoch.line_search_iters[-1]["residual"]
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

            prediction_match = self.prediction_re.search(message)
            if prediction_match:
                self.current_opt_epoch.prediction = to_float(prediction_match.group(1))
                self.current_opt_epoch.target = to_float(prediction_match.group(2))
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

            confidence_skipped_match = self.confidence_skipped_re.search(message)
            if confidence_skipped_match:
                skipped = confidence_skipped_match.group(1).strip()
                self.current_opt_epoch.confidence_skipped_legs = [item for item in skipped.split(",") if item]
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
    stroke_width: float = 2.0
    dashed: bool = False
    draw_markers: bool = False


@dataclass
class HLineSpec:
    value: float
    color: str
    label: str = ""
    dashed: bool = True


@dataclass
class VLineSpec:
    value: float
    color: str
    label: str = ""
    dashed: bool = True


@dataclass
class MarkerSpec:
    x: float
    y: float
    color: str
    label: str = ""
    radius: float = 4.0


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


def figure_extension() -> str:
    return "png" if HAS_MATPLOTLIB else "svg"


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt_tick(value: float) -> str:
    if not math.isfinite(value):
        return ""
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:.0f}"
    if abs_value >= 100:
        return f"{value:.1f}"
    if abs_value >= 10:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if abs_value >= 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.4f}".rstrip("0").rstrip(".")


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


def map_x(value: float, x_min: float, x_max: float, x0: float, width: float) -> float:
    return x0 + (value - x_min) / (x_max - x_min) * width


def map_y(value: float, y_min: float, y_max: float, y0: float, height: float) -> float:
    return y0 + height - (value - y_min) / (y_max - y_min) * height


def add_line(parts: List[str], x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.0, dashed: bool = False, opacity: float = 1.0) -> None:
    dash_attr = ' stroke-dasharray="6,4"' if dashed else ""
    parts.append(
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width:.2f}" opacity="{opacity:.3f}"{dash_attr} />'
    )


def add_rect(parts: List[str], x: float, y: float, width: float, height: float, fill: str, stroke: str = "", opacity: float = 1.0) -> None:
    stroke_attr = f' stroke="{stroke}"' if stroke else ""
    parts.append(
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{fill}" opacity="{opacity:.3f}"{stroke_attr} />'
    )


def add_text(parts: List[str], x: float, y: float, text: str, size: int = 13, anchor: str = "start", fill: str = "#222222", weight: str = "normal") -> None:
    parts.append(
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{svg_escape(text)}</text>'
    )


def add_polyline(parts: List[str], points: Sequence[Tuple[float, float]], color: str, width: float = 2.0, dashed: bool = False, opacity: float = 1.0) -> None:
    if len(points) < 2:
        return
    dash_attr = ' stroke-dasharray="6,4"' if dashed else ""
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    parts.append(
        f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="{width:.2f}" stroke-linejoin="round" stroke-linecap="round" opacity="{opacity:.3f}"{dash_attr} />'
    )


def add_circle(parts: List[str], x: float, y: float, radius: float, fill: str, stroke: str = "#ffffff") -> None:
    parts.append(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="1.0" />'
    )


def draw_line_panel(parts: List[str], panel: PanelSpec, x0: float, y0: float, width: float, height: float, is_last: bool) -> None:
    plot_left = x0 + 70
    plot_top = y0 + 28
    plot_width = width - 120
    plot_height = height - 72
    x_min, x_max = compute_x_range(panel)
    y_min, y_max = compute_range(panel)

    add_text(parts, x0 + 8, y0 + 20, panel.title, size=16, weight="bold")
    add_rect(parts, plot_left, plot_top, plot_width, plot_height, "#ffffff", stroke="#d8d8d8")

    for i in range(6):
        frac = i / 5.0
        y_val = y_min + (y_max - y_min) * frac
        y_px = map_y(y_val, y_min, y_max, plot_top, plot_height)
        add_line(parts, plot_left, y_px, plot_left + plot_width, y_px, "#eeeeee", width=1.0)
        add_text(parts, plot_left - 8, y_px + 4, fmt_tick(y_val), size=11, anchor="end", fill="#666666")

    for i in range(6):
        frac = i / 5.0
        x_val = x_min + (x_max - x_min) * frac
        x_px = map_x(x_val, x_min, x_max, plot_left, plot_width)
        add_line(parts, x_px, plot_top, x_px, plot_top + plot_height, "#f2f2f2", width=1.0)
        add_text(parts, x_px, plot_top + plot_height + 18, fmt_tick(x_val), size=11, anchor="middle", fill="#666666")

    add_line(parts, plot_left, plot_top + plot_height, plot_left + plot_width, plot_top + plot_height, "#888888", width=1.2)
    add_line(parts, plot_left, plot_top, plot_left, plot_top + plot_height, "#888888", width=1.2)

    for hline in panel.hlines:
        y_px = map_y(hline.value, y_min, y_max, plot_top, plot_height)
        add_line(parts, plot_left, y_px, plot_left + plot_width, y_px, hline.color, width=1.4, dashed=hline.dashed, opacity=0.85)
        if hline.label:
            add_text(parts, plot_left + plot_width - 4, y_px - 4, hline.label, size=11, anchor="end", fill=hline.color)

    for vline in panel.vlines:
        x_px = map_x(vline.value, x_min, x_max, plot_left, plot_width)
        add_line(parts, x_px, plot_top, x_px, plot_top + plot_height, vline.color, width=1.4, dashed=vline.dashed, opacity=0.7)
        if vline.label:
            add_text(parts, x_px + 4, plot_top + 12, vline.label, size=11, fill=vline.color)

    legend_y = plot_top + 16
    legend_x = plot_left + 10
    if panel.legend:
        for idx, series in enumerate(panel.series):
            row = idx // 4
            col = idx % 4
            item_x = legend_x + col * 190
            item_y = legend_y + row * 16
            add_line(parts, item_x, item_y, item_x + 20, item_y, series.color, width=2.5, dashed=series.dashed)
            add_text(parts, item_x + 26, item_y + 4, series.name, size=11)

    for series in panel.series:
        mapped_points = [
            (map_x(x, x_min, x_max, plot_left, plot_width), map_y(y, y_min, y_max, plot_top, plot_height))
            for x, y in series.points
            if math.isfinite(x) and math.isfinite(y)
        ]
        add_polyline(parts, mapped_points, series.color, width=series.stroke_width, dashed=series.dashed)
        if series.draw_markers:
            for x_px, y_px in mapped_points:
                add_circle(parts, x_px, y_px, 2.8, series.color)

    for marker in panel.markers:
        if not (math.isfinite(marker.x) and math.isfinite(marker.y)):
            continue
        x_px = map_x(marker.x, x_min, x_max, plot_left, plot_width)
        y_px = map_y(marker.y, y_min, y_max, plot_top, plot_height)
        add_circle(parts, x_px, y_px, marker.radius, marker.color)
        if marker.label:
            add_text(parts, x_px + 6, y_px - 6, marker.label, size=11, fill=marker.color)

    add_text(parts, plot_left + plot_width / 2.0, y0 + height - 16, panel.x_label, size=12, anchor="middle", fill="#444444")
    add_text(parts, x0 + 18, plot_top + plot_height / 2.0, panel.y_label, size=12, fill="#444444")

    if panel.notes:
        note_y = y0 + height - 34
        for idx, note in enumerate(panel.notes):
            add_text(parts, plot_left + plot_width - 4, note_y - idx * 14, note, size=10, anchor="end", fill="#666666")

    if not is_last:
        add_line(parts, x0, y0 + height + 8, x0 + width, y0 + height + 8, "#e6e6e6", width=1.0)


def write_multiplot_svg(path: Path, title: str, panels: Sequence[PanelSpec], width: int = 1400, panel_height: int = 250) -> None:
    outer_margin = 18
    title_height = 34
    gap = 24
    height = outer_margin * 2 + title_height + len(panels) * panel_height + max(0, len(panels) - 1) * gap
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfd" />',
    ]
    add_text(parts, outer_margin, outer_margin + 20, title, size=22, weight="bold")
    y_cursor = outer_margin + title_height
    for idx, panel in enumerate(panels):
        draw_line_panel(parts, panel, outer_margin, y_cursor, width - outer_margin * 2, panel_height, idx == len(panels) - 1)
        y_cursor += panel_height + gap
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_multiplot_matplotlib(path: Path, title: str, panels: Sequence[PanelSpec], width: int = 1400, panel_height: int = 250) -> None:
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib backend unavailable")

    fig_height = max(3.0, len(panels) * (panel_height / 100.0) + 0.8)
    fig, axes = plt.subplots(len(panels), 1, figsize=(width / 100.0, fig_height), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]
    fig.suptitle(title, fontsize=18, fontweight="bold", x=0.05, ha="left")

    for ax, panel in zip(axes_flat, panels):
        x_min, x_max = compute_x_range(panel)
        y_min, y_max = compute_range(panel)
        ax.set_title(panel.title, fontsize=13, fontweight="bold", loc="left")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, which="major", color="#e9e9ef", linewidth=0.8)
        ax.set_facecolor("#ffffff")

        for hline in panel.hlines:
            ax.axhline(hline.value, color=hline.color, linewidth=1.3, linestyle="--" if hline.dashed else "-")
            if hline.label:
                ax.text(0.995, hline.value, hline.label, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color=hline.color)

        for vline in panel.vlines:
            ax.axvline(vline.value, color=vline.color, linewidth=1.2, linestyle="--" if vline.dashed else "-", alpha=0.75)
            if vline.label:
                ax.text(vline.value, 0.98, vline.label, transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=8, color=vline.color)

        maxd = -math.inf
        minn = math.inf
        maxx = -math.inf

        for series in panel.series:
            xs = [x for x, y in series.points if math.isfinite(x) and math.isfinite(y)]
            if "Parameter Trajectories" in panel.title:
                ys = [100*(y - series.points[0][1])/series.points[0][1] for x, y in series.points if math.isfinite(x) and math.isfinite(y)]
                d = abs(max(ys)-min(ys))
                if d > maxd:
                    maxd = d
                if min(ys) < minn:
                    minn = min(ys)
                if max(ys) > maxx:
                    maxx = max(ys)
            else:
                ys = [y for x, y in series.points if math.isfinite(x) and math.isfinite(y)]
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                label=series.name,
                color=series.color,
                linewidth=series.stroke_width,
                linestyle="--" if series.dashed else "-",
                marker="o" if series.draw_markers else None,
                markersize=3.0 if series.draw_markers else None,
            )

        for marker in panel.markers:
            if not (math.isfinite(marker.x) and math.isfinite(marker.y)):
                continue
            ax.scatter([marker.x], [marker.y], s=max(18.0, marker.radius * marker.radius * 2.4), color=marker.color, edgecolors="#ffffff", linewidths=0.8, zorder=4)
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

    if "Parameter Trajectories" in panel.title:
        ax.set_ylim(minn - maxd*0.05, maxx + maxd*0.05)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_timeline_svg(path: Path, run: RunRecord) -> None:
    width = 1400
    height = 360
    margin = 18
    title_height = 34
    plot_left = 140
    plot_top = 90
    plot_width = width - plot_left - 28
    row_height = 56
    legs = ["solvent", "vacuum"]
    cumulative_segments: List[Tuple[str, str, float, float]] = []
    cumulative_by_leg = {leg: 0.0 for leg in legs}
    for event in run.phase_events:
        if event.kind != "end" or event.md_ns is None or event.leg not in cumulative_by_leg:
            continue
        start = cumulative_by_leg[event.leg]
        end = start + event.md_ns
        cumulative_segments.append((event.leg, event.phase, start, end))
        cumulative_by_leg[event.leg] = end
    max_x = max((end for _, _, _, end in cumulative_segments), default=1.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfd" />',
    ]
    add_text(parts, margin, margin + 20, "AWH Timeline", size=22, weight="bold")
    add_text(parts, margin, margin + 42, "Global cumulative MD ns per leg. Colors encode the sampled phase.", size=12, fill="#555555")

    for i in range(6):
        frac = i / 5.0
        x_val = max_x * frac
        x_px = plot_left + plot_width * frac
        add_line(parts, x_px, plot_top - 8, x_px, plot_top + row_height * len(legs) + 12, "#ececec", width=1.0)
        add_text(parts, x_px, plot_top + row_height * len(legs) + 32, fmt_tick(x_val), size=11, anchor="middle", fill="#666666")

    for row, leg in enumerate(legs):
        y = plot_top + row * row_height
        add_text(parts, margin + 8, y + 22, leg.capitalize(), size=15, weight="bold")
        add_rect(parts, plot_left, y, plot_width, 24, "#ffffff", stroke="#d8d8d8")
        for seg_leg, phase, start, end in cumulative_segments:
            if seg_leg != leg:
                continue
            x1 = plot_left + (start / max_x) * plot_width
            x2 = plot_left + (end / max_x) * plot_width
            add_rect(parts, x1, y + 1, max(1.0, x2 - x1), 22, PHASE_COLORS.get(phase, "#888888"), opacity=0.85)
        freeze_events = [event for event in run.control_events if event.leg == leg and event.event_type == "leg_frozen" and event.global_ns is not None]
        for freeze in freeze_events:
            x_px = plot_left + (freeze.global_ns / max_x) * plot_width
            add_line(parts, x_px, y - 6, x_px, y + 30, "#2a9d8f", width=2.0, dashed=True)
            add_text(parts, x_px + 4, y - 10, f"freeze m{freeze.macro_index}", size=10, fill="#2a9d8f")

    legend_x = plot_left
    legend_y = plot_top - 34
    for idx, (phase, color) in enumerate(PHASE_COLORS.items()):
        x = legend_x + idx * 220
        add_rect(parts, x, legend_y, 18, 12, color)
        add_text(parts, x + 24, legend_y + 11, phase, size=11)

    add_text(parts, plot_left + plot_width / 2.0, height - 18, "Global cumulative AWH MD ns", size=12, anchor="middle", fill="#444444")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_timeline_matplotlib(path: Path, run: RunRecord) -> None:
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib backend unavailable")

    fig, ax = plt.subplots(figsize=(14, 4.2))
    legs = ["solvent", "vacuum"]
    cumulative_segments: List[Tuple[str, str, float, float]] = []
    cumulative_by_leg = {leg: 0.0 for leg in legs}
    for event in run.phase_events:
        if event.kind != "end" or event.md_ns is None or event.leg not in cumulative_by_leg:
            continue
        start = cumulative_by_leg[event.leg]
        end = start + event.md_ns
        cumulative_segments.append((event.leg, event.phase, start, end))
        cumulative_by_leg[event.leg] = end

    y_pos = {"solvent": 1.0, "vacuum": 0.0}
    for leg, phase, start, end in cumulative_segments:
        ax.broken_barh([(start, end - start)], (y_pos[leg] - 0.18, 0.36), facecolors=PHASE_COLORS.get(phase, "#888888"), edgecolors="none", alpha=0.9)

    for event in run.control_events:
        if event.event_type != "leg_frozen" or event.global_ns is None or event.leg not in y_pos:
            continue
        ax.axvline(event.global_ns, color="#2a9d8f", linestyle="--", linewidth=1.4, alpha=0.9)
        ax.annotate(f"{event.leg} freeze m{event.macro_index}", (event.global_ns, y_pos[event.leg] + 0.2), textcoords="offset points", xytext=(4, 2), fontsize=8, color="#2a9d8f")

    max_x = max((end for _, _, _, end in cumulative_segments), default=1.0)
    ax.set_xlim(0.0, max_x * 1.02)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([1.0, 0.0], ["Solvent", "Vacuum"])
    ax.set_xlabel("Global cumulative AWH MD ns")
    ax.set_title("AWH Timeline", fontsize=18, fontweight="bold", loc="left")
    ax.grid(True, axis="x", color="#ececec", linewidth=0.9)
    ax.set_facecolor("#ffffff")
    legend_handles = [Patch(facecolor=color, edgecolor="none", label=phase) for phase, color in PHASE_COLORS.items()]
    legend_handles.append(Line2D([0], [0], color="#2a9d8f", linestyle="--", label="Leg frozen"))
    ax.legend(handles=legend_handles, loc="upper center", ncol=min(4, len(legend_handles)), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_multiplot(path: Path, title: str, panels: Sequence[PanelSpec], width: int = 1400, panel_height: int = 250) -> None:
    if path.suffix.lower() == ".png" and HAS_MATPLOTLIB:
        write_multiplot_matplotlib(path, title, panels, width=width, panel_height=panel_height)
    else:
        write_multiplot_svg(path, title, panels, width=width, panel_height=panel_height)


def write_timeline(path: Path, run: RunRecord) -> None:
    if path.suffix.lower() == ".png" and HAS_MATPLOTLIB:
        write_timeline_matplotlib(path, run)
    else:
        write_timeline_svg(path, run)


def by_leg(records: Sequence[object], leg_name: str) -> List[object]:
    return [record for record in records if getattr(record, "leg", None) == leg_name]


def make_stage_a_panels(run: RunRecord) -> List[PanelSpec]:
    panels: List[PanelSpec] = []
    for metric_key, title, y_label, color in [
        ("df", "Stage A Bias-Change", "df_mean", METRIC_COLORS["df"]),
        ("ess", "Stage A Lambda ESS", "ESS", METRIC_COLORS["ess"]),
        ("lin_neff", "Stage A Linear N_eff", "linear N_eff", METRIC_COLORS["lin_neff"]),
        ("round_trips", "Stage A Round Trips", "round trips", METRIC_COLORS["round_trips"]),
    ]:
        series: List[SeriesSpec] = []
        markers: List[MarkerSpec] = []
        for leg in ("solvent", "vacuum"):
            points = [
                (snapshot.global_ns, getattr(snapshot, metric_key))
                for snapshot in by_leg(run.stage_a, leg)
                if snapshot.global_ns is not None and getattr(snapshot, metric_key) is not None
            ]
            if points:
                series.append(SeriesSpec(leg.capitalize(), points, LEG_COLORS[leg], draw_markers=False))
            ready_points = [
                (snapshot.global_ns, getattr(snapshot, metric_key))
                for snapshot in by_leg(run.stage_a, leg)
                if snapshot.global_ns is not None and getattr(snapshot, metric_key) is not None and snapshot.stage_ready
            ]
            for x_val, y_val in ready_points:
                markers.append(MarkerSpec(x_val, y_val, "#2a9d8f", radius=3.4))
        panels.append(
            PanelSpec(
                title=title,
                x_label="Global cumulative AWH MD ns",
                y_label=y_label,
                series=series,
                markers=markers,
                notes=["Green dots indicate Stage A ready blocks."],
            )
        )
    return panels


def make_stage_a_occupancy_panels(run: RunRecord) -> List[PanelSpec]:
    panels: List[PanelSpec] = []
    solvent = by_leg(run.stage_a, "solvent")
    vacuum = by_leg(run.stage_a, "vacuum")
    solvent_series = [
        SeriesSpec(
            "Endpoint band",
            [(snap.global_ns, snap.endpoint_band) for snap in solvent if snap.global_ns is not None and snap.endpoint_band is not None],
            METRIC_COLORS["endpoint_band"],
        ),
        SeriesSpec(
            "Endpoint low",
            [(snap.global_ns, snap.endpoint_low) for snap in solvent if snap.global_ns is not None and snap.endpoint_low is not None],
            METRIC_COLORS["endpoint_low"],
            dashed=True,
        ),
        SeriesSpec(
            "Tail sum",
            [(snap.global_ns, snap.tail_sum) for snap in solvent if snap.global_ns is not None and snap.tail_sum is not None],
            METRIC_COLORS["tail_sum"],
        ),
        SeriesSpec(
            "Tail min",
            [(snap.global_ns, snap.tail_min) for snap in solvent if snap.global_ns is not None and snap.tail_min is not None],
            METRIC_COLORS["tail_min"],
            dashed=True,
        ),
        SeriesSpec(
            "Occ min",
            [(snap.global_ns, snap.occ_min) for snap in solvent if snap.global_ns is not None and snap.occ_min is not None],
            METRIC_COLORS["occ_min"],
        ),
    ]
    panels.append(
        PanelSpec(
            title="Stage A Occupancy: Solvent",
            x_label="Global cumulative AWH MD ns",
            y_label="occupancy",
            series=solvent_series,
        )
    )
    vacuum_series = [
        SeriesSpec(
            "Endpoint band",
            [(snap.global_ns, snap.endpoint_band) for snap in vacuum if snap.global_ns is not None and snap.endpoint_band is not None],
            METRIC_COLORS["endpoint_band"],
        ),
        SeriesSpec(
            "Endpoint low",
            [(snap.global_ns, snap.endpoint_low) for snap in vacuum if snap.global_ns is not None and snap.endpoint_low is not None],
            METRIC_COLORS["endpoint_low"],
            dashed=True,
        ),
        SeriesSpec(
            "Occ min",
            [(snap.global_ns, snap.occ_min) for snap in vacuum if snap.global_ns is not None and snap.occ_min is not None],
            METRIC_COLORS["occ_min"],
        ),
    ]
    panels.append(
        PanelSpec(
            title="Stage A Occupancy: Vacuum",
            x_label="Global cumulative AWH MD ns",
            y_label="occupancy",
            series=vacuum_series,
        )
    )
    return panels


def make_stage_b_panels(run: RunRecord) -> List[PanelSpec]:
    panels: List[PanelSpec] = []
    for leg in ("solvent", "vacuum"):
        fresh = [snap for snap in by_leg(run.stage_b, leg) if snap.fresh_probe_result and snap.attempt_index is not None]
        if not fresh:
            continue
        series = [
            SeriesSpec("Split gap", [(snap.attempt_index, snap.split_gap) for snap in fresh if snap.split_gap is not None], METRIC_COLORS["split_gap"]),
            SeriesSpec("Parity gap", [(snap.attempt_index, snap.parity_gap) for snap in fresh if snap.parity_gap is not None], METRIC_COLORS["parity_gap"]),
            SeriesSpec(
                "Supported parity",
                [(snap.attempt_index, snap.supported_parity_gap) for snap in fresh if snap.supported_parity_gap is not None],
                METRIC_COLORS["supported_parity_gap"],
                dashed=True,
            ),
            SeriesSpec(
                "Endpoint parity",
                [(snap.attempt_index, snap.endpoint_parity_gap) for snap in fresh if snap.endpoint_parity_gap is not None],
                METRIC_COLORS["endpoint_parity_gap"],
                dashed=True,
            ),
        ]
        markers = [
            MarkerSpec(
                snap.attempt_index,
                snap.parity_gap if snap.parity_gap is not None else 0.0,
                FAILURE_COLORS.get(snap.failure_mode or "not_checked", "#777777"),
                label="pass" if (snap.failure_mode or "") == "passed" else "",
                radius=4.2,
            )
            for snap in fresh
            if snap.parity_gap is not None
        ]
        panels.append(
            PanelSpec(
                title=f"Stage B Gaps: {leg.capitalize()}",
                x_label="Fresh Stage B probe attempt",
                y_label="gap (kT)",
                series=series,
                markers=markers,
                notes=["Marker color encodes the Stage B failure mode."],
            )
        )
        support_series = [
            SeriesSpec(
                "Support fraction",
                [
                    (snap.attempt_index, snap.n_supported_states / snap.n_states)
                    for snap in fresh
                    if snap.n_supported_states is not None and snap.n_states
                ],
                METRIC_COLORS["support_fraction"],
            )
        ]
        panels.append(
            PanelSpec(
                title=f"Stage B Support Coverage: {leg.capitalize()}",
                x_label="Fresh Stage B probe attempt",
                y_label="fraction",
                series=support_series,
                y_min=0.0,
                y_max=1.05,
            )
        )
        probe_series = [
            SeriesSpec(
                "Retained frames",
                [(snap.attempt_index, snap.frames) for snap in fresh if snap.frames is not None],
                METRIC_COLORS["frames"],
            )
        ]
        control_ns = []
        for event in run.control_events:
            if event.leg != leg:
                continue
            if event.event_type in {"probe_retry_keep", "probe_retry_grow", "probe_retry_near_pass"}:
                probe_ns = event.details.get("next_probe_ns")
                if isinstance(probe_ns, (int, float)):
                    idx = len(control_ns) + 1
                    control_ns.append((idx, float(probe_ns)))
        if control_ns:
            probe_series.append(SeriesSpec("Next probe ns", control_ns, METRIC_COLORS["probe_ns"], dashed=True))
        panels.append(
            PanelSpec(
                title=f"Stage B Frames And Probe Size: {leg.capitalize()}",
                x_label="Fresh Stage B probe attempt",
                y_label="frames / ns",
                series=probe_series,
            )
        )
    return panels


def make_optimization_panels(run: RunRecord) -> List[PanelSpec]:
    epochs = run.optimization_epochs
    if not epochs:
        return []
    boundaries = sorted({epoch.global_epoch_index for epoch in epochs if epoch.epoch_in_macro == 1})
    panels: List[PanelSpec] = []
    panels.append(
        PanelSpec(
            title="Optimization Residuals",
            x_label="Global optimization epoch",
            y_label="kT",
            series=[
                SeriesSpec(
                    "Residual",
                    [(epoch.global_epoch_index, epoch.residual) for epoch in epochs if epoch.residual is not None],
                    METRIC_COLORS["residual"],
                ),
                SeriesSpec(
                    "Accepted residual",
                    [(epoch.global_epoch_index, epoch.accepted_residual) for epoch in epochs if epoch.accepted_residual is not None],
                    METRIC_COLORS["accepted_residual"],
                ),
                SeriesSpec(
                    "Prediction",
                    [(epoch.global_epoch_index, epoch.prediction) for epoch in epochs if epoch.prediction is not None],
                    METRIC_COLORS["prediction"],
                    dashed=True,
                ),
                SeriesSpec(
                    "Target",
                    [(epoch.global_epoch_index, epoch.target) for epoch in epochs if epoch.target is not None],
                    METRIC_COLORS["target"],
                    dashed=True,
                ),
            ],
            vlines=[VLineSpec(value, "#bbbbbb", label=f"m{epochs[value - 1].macro_index}" if 0 < value <= len(epochs) else "", dashed=True) for value in boundaries],
        )
    )
    panels.append(
        PanelSpec(
            title="Optimization Trust Region",
            x_label="Global optimization epoch",
            y_label="KL / scaling / alpha / step",
            series=[
                SeriesSpec("Est. KL", [(epoch.global_epoch_index, epoch.kl_est) for epoch in epochs if epoch.kl_est is not None], METRIC_COLORS["kl_est"]),
                SeriesSpec("KL scaling", [(epoch.global_epoch_index, epoch.kl_scaling) for epoch in epochs if epoch.kl_scaling is not None], METRIC_COLORS["kl_scaling"]),
                SeriesSpec(
                    "Line-search alpha",
                    [(epoch.global_epoch_index, epoch.line_search_alpha) for epoch in epochs if epoch.line_search_alpha is not None],
                    METRIC_COLORS["line_search_alpha"],
                ),
                SeriesSpec(
                    "Max phi step",
                    [(epoch.global_epoch_index, epoch.actual_max_phi_step) for epoch in epochs if epoch.actual_max_phi_step is not None],
                    METRIC_COLORS["max_phi_step"],
                ),
            ],
        )
    )
    panels.append(
        PanelSpec(
            title="Optimization Gradients And Fisher Conditioning",
            x_label="Global optimization epoch",
            y_label="gradient / condition / truncation",
            series=[
                SeriesSpec("Grad norm", [(epoch.global_epoch_index, epoch.grad_norm) for epoch in epochs if epoch.grad_norm is not None], METRIC_COLORS["grad_norm"]),
                SeriesSpec("Grad max", [(epoch.global_epoch_index, epoch.grad_max) for epoch in epochs if epoch.grad_max is not None], METRIC_COLORS["grad_max"]),
                SeriesSpec(
                    "Truncated eigs",
                    [(epoch.global_epoch_index, float(epoch.truncated_eigs)) for epoch in epochs if epoch.truncated_eigs is not None],
                    METRIC_COLORS["truncated_eigs"],
                ),
                SeriesSpec(
                    "log10(cond)",
                    [
                        (epoch.global_epoch_index, math.log10(epoch.fim_raw_cond))
                        for epoch in epochs
                        if epoch.fim_raw_cond is not None and epoch.fim_raw_cond > 0 and math.isfinite(epoch.fim_raw_cond)
                    ],
                    METRIC_COLORS["fim_cond"],
                ),
            ],
            notes=["Fisher condition number is plotted as log10(cond)."],
        )
    )
    for leg in ("solvent", "vacuum"):
        leg_dg = []
        leg_ess = []
        leg_active_log = []
        for epoch in epochs:
            metric = epoch.leg_metrics.get(leg)
            if not metric:
                continue
            leg_dg.append((epoch.global_epoch_index, metric.get("dG")))
            leg_ess.append((epoch.global_epoch_index, metric.get("ESS")))
            n_active = metric.get("N_active")
            if n_active is not None and n_active > 0:
                leg_active_log.append((epoch.global_epoch_index, math.log10(n_active)))
        if leg_dg:
            panels.append(
                PanelSpec(
                    title=f"Optimization dG: {leg.capitalize()}",
                    x_label="Global optimization epoch",
                    y_label="dG (kT)",
                    series=[
                        SeriesSpec("dG", [(x, y) for x, y in leg_dg if y is not None], METRIC_COLORS["dG"]),
                    ],
                )
            )
        if leg_ess or leg_active_log:
            panels.append(
                PanelSpec(
                    title=f"Optimization Support: {leg.capitalize()}",
                    x_label="Global optimization epoch",
                    y_label="ESS / log10(N_active)",
                    series=[
                        SeriesSpec("ESS", [(x, y) for x, y in leg_ess if y is not None], METRIC_COLORS["ESS"]),
                        SeriesSpec("log10(N_active)", [(x, y) for x, y in leg_active_log if y is not None], METRIC_COLORS["N_active"]),
                    ],
                    notes=["N_active is plotted as log10(N_active)."],
                )
            )
    return panels


def chunked(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def parameter_panels(run: RunRecord, chunk_size: int) -> List[Tuple[str, List[PanelSpec]]]:
    epochs = run.optimization_epochs
    if not epochs:
        return []
    param_names = []
    for epoch in epochs:
        for name in epoch.parameters:
            if name not in param_names:
                param_names.append(name)
    panels_by_file: List[Tuple[str, List[PanelSpec]]] = []
    palette = [
        "#c2552d",
        "#2d6db6",
        "#3a7d44",
        "#7b2cbf",
        "#d17b0f",
        "#ef476f",
        "#118ab2",
        "#6d597a",
    ]
    for file_idx, chunk in enumerate(chunked(param_names, max(1, chunk_size)), start=1):
        series = []
        for idx, name in enumerate(chunk):
            points = [
                (epoch.global_epoch_index, epoch.parameters.get(name))
                for epoch in epochs
                if name in epoch.parameters
            ]
            filtered = [(x, y) for x, y in points if y is not None]
            if filtered:
                series.append(SeriesSpec(name, filtered, palette[idx % len(palette)]))
        if not series:
            continue
        panels_by_file.append(
            (
                f"parameter_trajectories_{file_idx:02d}.{figure_extension()}",
                [
                    PanelSpec(
                        title=f"Parameter Trajectories ({file_idx})",
                        x_label="Global optimization epoch",
                        y_label="parameter drift (%)",
                        series=series,
                    )
                ],
            )
        )
    return panels_by_file


def write_index_html(outdir: Path, stem: str, figure_names: Sequence[str], report_txt: str) -> None:
    blocks = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8' />",
        f"<title>{svg_escape(stem)} progress report</title>",
        "<style>body{font-family:DejaVu Sans,Arial,sans-serif;margin:24px;background:#fbfbfd;color:#222} img{max-width:100%;border:1px solid #ddd;background:#fff;margin:18px 0} pre{background:#fff;border:1px solid #ddd;padding:16px;overflow:auto}</style>",
        "</head><body>",
        f"<h1>{svg_escape(stem)} progress report</h1>",
        "<h2>Summary</h2>",
        f"<pre>{svg_escape(report_txt)}</pre>",
        "<h2>Figures</h2>",
    ]
    for name in figure_names:
        blocks.append(f"<h3>{svg_escape(name)}</h3>")
        blocks.append(f"<img src='{svg_escape(name)}' alt='{svg_escape(name)}' />")
    blocks.append("</body></html>")
    (outdir / "index.html").write_text("\n".join(blocks), encoding="utf-8")


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
        leg_stage_b = [snap for snap in run.stage_b if snap.leg == leg]
        fresh = [snap for snap in leg_stage_b if snap.fresh_probe_result]
        last_snapshot = leg_stage_b[-1] if leg_stage_b else None
        lines.append(
            f"{leg}: stage_a={sum(1 for snap in run.stage_a if snap.leg == leg)} fresh_stage_b={len(fresh)}"
        )
        if last_snapshot is not None:
            lines.append(
                f"  final_stage_b: failure={last_snapshot.failure_mode} split_gap={last_snapshot.split_gap} parity_gap={last_snapshot.parity_gap} support={last_snapshot.n_supported_states}/{last_snapshot.n_states}"
            )
    if run.optimization_epochs:
        last_epoch = run.optimization_epochs[-1]
        lines.append(
            f"final_optimization_epoch: global={last_epoch.global_epoch_index} macro={last_epoch.macro_index} residual={last_epoch.residual} accepted_residual={last_epoch.accepted_residual}"
        )
        lines.append(f"tracked_parameters: {len(last_epoch.parameters)}")
    if run.warnings:
        lines.append("notes:")
        lines.extend(f"  - {warning}" for warning in run.warnings)
    return "\n".join(lines)


def write_report(outdir: Path, run: RunRecord, max_params_per_figure: int) -> List[str]:
    figure_names: List[str] = []
    write_summary_json(outdir, run)
    ext = figure_extension()

    timeline_name = f"awh_timeline.{ext}"
    write_timeline(outdir / timeline_name, run)
    figure_names.append(timeline_name)

    stage_a_panels = make_stage_a_panels(run)
    if stage_a_panels:
        name = f"stage_a_convergence.{ext}"
        write_multiplot(outdir / name, "Stage A Convergence", stage_a_panels)
        figure_names.append(name)

    occupancy_panels = make_stage_a_occupancy_panels(run)
    if occupancy_panels:
        name = f"stage_a_occupancy.{ext}"
        write_multiplot(outdir / name, "Stage A Occupancy", occupancy_panels)
        figure_names.append(name)

    stage_b_panels = make_stage_b_panels(run)
    if stage_b_panels:
        name = f"stage_b_convergence.{ext}"
        write_multiplot(outdir / name, "Stage B Convergence", stage_b_panels)
        figure_names.append(name)

    opt_panels = make_optimization_panels(run)
    if opt_panels:
        name = f"optimization_metrics.{ext}"
        write_multiplot(outdir / name, "Optimization Metrics", opt_panels, panel_height=240)
        figure_names.append(name)

    for name, panels in parameter_panels(run, max_params_per_figure):
        write_multiplot(outdir / name, "Parameter Trajectories", panels, panel_height=260)
        figure_names.append(name)

    report_txt = build_report_text(run)
    (outdir / "report.txt").write_text(report_txt + "\n", encoding="utf-8")
    write_index_html(outdir, Path(run.log_path).stem, figure_names, report_txt)
    return figure_names


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="One or more log files to parse.")
    parser.add_argument("--output-dir", default="plots", help="Base output directory. Default: plots")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing per-log output directory.")
    parser.add_argument("--summary-only", action="store_true", help="Parse logs and write summary files without rendering figures.")
    parser.add_argument("--max-params-per-figure", type=int, default=8, help="Maximum parameter traces per figure. Default: 8")
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
            figure_names = write_report(outdir, run, args.max_params_per_figure)
        else:
            write_index_html(outdir, log_path.stem, figure_names, report_txt)
        wrote_any = True
        print(f"[ok] parsed {log_path} -> {outdir}")
        print(f"     stage_a={len(run.stage_a)} stage_b={len(run.stage_b)} optimization_epochs={len(run.optimization_epochs)}")
        print(f"     figure_backend={'matplotlib/png' if HAS_MATPLOTLIB else 'svg-fallback'}")
        if figure_names:
            print(f"     figures={len(figure_names)} index={outdir / 'index.html'}")
        if run.warnings:
            print(f"     notes={len(run.warnings)}")
    if not wrote_any:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
