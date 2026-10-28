#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from case_builder.builder import CaseBuilder
from case_builder.models import DEFAULT_PLANNER_BASE_URL, DEFAULT_PLANNER_MODEL, BuildRequest


def _log_cli(message: str) -> None:
    print(f"[build_case] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one executable scene-based case.")
    parser.add_argument(
        "--catalog",
        default="topology/Town10HD_Opt/walkable_cells.json",
        help="Path to walkable_cells.json.",
    )
    parser.add_argument(
        "--output-root",
        default="data",
        help="Root directory for generated cases. Cases will be written under data/<map>/<case_id>/.",
    )
    parser.add_argument(
        "--planner",
        choices=("heuristic", "llm"),
        default="heuristic",
        help="Semantic planner mode.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for deterministic case generation.")
    parser.add_argument(
        "--scene-kind",
        choices=("crossing", "sidewalk"),
        default=None,
        help="Optional scene kind filter.",
    )
    parser.add_argument(
        "--no-sampling",
        action="store_true",
        help="Disable sampling block and ego declaration.",
    )
    parser.add_argument("--execute", action="store_true", help="Run run_simulation.py after building the case.")
    parser.add_argument("--duration", type=float, default=None, help="Optional simulation duration override.")
    parser.add_argument(
        "--wall-time-limit",
        type=float,
        default=None,
        help="Optional wall-clock timeout in seconds for one case execution.",
    )
    parser.add_argument("--conda-env", default="carla", help="Conda environment used for execution.")
    parser.add_argument("--carla-host", default="127.0.0.1", help="CARLA host.")
    parser.add_argument("--carla-port", type=int, default=2000, help="CARLA port.")
    parser.add_argument("--scenario-prefix", default="case", help="Prefix used in scenario_name.")
    parser.add_argument("--model", default=DEFAULT_PLANNER_MODEL, help="Planner model when --planner llm is used.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_PLANNER_BASE_URL,
        help="OpenAI-compatible base URL when --planner llm is used.",
    )
    parser.add_argument("--api-key", default=None, help="Optional API key override for llm planner.")
    parser.add_argument(
        "--llm-strict",
        action="store_true",
        help="Fail instead of falling back to heuristic planning when the LLM planner is unavailable or invalid.",
    )
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=2,
        help="Total LLM planning attempts before fallback or failure.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=90.0,
        help="HTTP timeout in seconds for one LLM planning attempt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _log_cli(
        "start "
        f"catalog={args.catalog} planner={args.planner} seed={args.seed} "
        f"execute={bool(args.execute)} sampling={not args.no_sampling} output_root={args.output_root}"
    )
    request = BuildRequest(
        catalog_path=Path(args.catalog),
        output_root=Path(args.output_root),
        planner=args.planner,
        seed=args.seed,
        scene_kind=args.scene_kind,
        sampling_enabled=not args.no_sampling,
        execute=bool(args.execute),
        duration=args.duration,
        wall_time_limit_s=args.wall_time_limit,
        conda_env=args.conda_env,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        scenario_prefix=args.scenario_prefix,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        llm_strict=bool(args.llm_strict),
        llm_retries=args.llm_retries,
        llm_timeout_s=args.llm_timeout,
    )
    builder = CaseBuilder(request)
    built_case = builder.build()
    payload = {
        "status": "built",
        "case_id": built_case.case_id,
        "case_index": built_case.case_index,
        "case_dir": str(built_case.case_dir),
        "config_dir": str(built_case.case_dir / "config"),
        "simulation_dir": str(built_case.case_dir / "simulation"),
        "sampling_dir": str(built_case.case_dir / "sampling"),
        "visualization_dir": str(built_case.case_dir / "visualization"),
        "logs_dir": str(built_case.case_dir / "logs"),
        "scenario_path": str(built_case.scenario_path),
        "description_path": str(built_case.description_path),
        "metadata_path": str(built_case.metadata_path),
        "manifest_path": str(built_case.manifest_path),
        "sampling_policy_path": str(built_case.sampling_policy_path),
        "scenario_name": built_case.plan.scenario_name,
        "description": built_case.plan.description,
    }
    if args.execute:
        result = builder.execute(built_case)
        payload["run_result"] = result.to_dict()
        payload["status"] = "executed" if result.returncode == 0 else "run_failed"
    _log_cli(
        "summary "
        f"status={payload['status']} case_dir={built_case.case_dir} "
        f"scenario={built_case.plan.scenario_name}"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
