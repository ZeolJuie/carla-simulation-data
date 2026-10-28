#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scenario_director import ScenarioDirector
from scenario_director.models import (
    DEFAULT_PLANNER_BASE_URL,
    DEFAULT_PLANNER_MODEL,
    CatalogSpec,
    DirectorRequest,
)


def _log_cli(message: str) -> None:
    print(f"[scenario_batch] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scenario batch generation with a lightweight ScenarioDirector.")
    parser.add_argument("--output-root", required=True, help="Final case root, e.g. ./data")
    parser.add_argument("--target-count", type=int, required=True, help="Target accepted case count.")
    parser.add_argument(
        "--catalog",
        action="append",
        required=True,
        help="Path to a walkable_cells.json catalog. Can be passed multiple times.",
    )
    parser.add_argument("--planner", choices=("heuristic", "llm"), default="heuristic")
    parser.add_argument("--model", default=DEFAULT_PLANNER_MODEL)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-sampling", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--state-path", default=None, help="Optional director state file path.")
    parser.add_argument("--conda-env", default="carla")
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--carla-root", default=None, help="Optional CARLA installation root for managed restarts.")
    parser.add_argument(
        "--wall-time-limit",
        type=float,
        default=None,
        help="Optional wall-clock timeout in seconds for each case execution.",
    )
    parser.add_argument(
        "--restart-min-cases",
        type=int,
        default=5,
        help="Minimum successful attempts between proactive CARLA restarts.",
    )
    parser.add_argument(
        "--restart-max-cases",
        type=int,
        default=8,
        help="Maximum successful attempts between proactive CARLA restarts.",
    )
    parser.add_argument(
        "--restart-wait",
        type=float,
        default=12.0,
        help="Seconds to wait after restarting CARLA before continuing.",
    )
    parser.add_argument(
        "--carla-ready-timeout",
        type=float,
        default=90.0,
        help="Seconds to wait for CARLA to become responsive after restart.",
    )
    parser.add_argument("--base-url", default=DEFAULT_PLANNER_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--llm-strict",
        action="store_true",
        help="Fail a case instead of falling back to heuristic planning when the LLM planner is unavailable or invalid.",
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
    parser.add_argument("--scenario-prefix", default="case")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _log_cli(
        "start "
        f"target_count={args.target_count} catalogs={len(args.catalog)} planner={args.planner} "
        f"execute={bool(args.execute)} sampling={not args.no_sampling} output_root={args.output_root}"
    )
    output_root = Path(args.output_root).resolve()
    state_path = Path(args.state_path).resolve() if args.state_path else output_root / "scenario_director_state.json"

    catalogs = []
    for catalog_path in args.catalog:
        path = Path(catalog_path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        catalogs.append(CatalogSpec(map_name=str(data["map_name"]), catalog_path=path))

    request = DirectorRequest(
        output_root=output_root,
        state_path=state_path,
        target_count=args.target_count,
        catalogs=catalogs,
        planner=args.planner,
        model=args.model,
        execute=bool(args.execute),
        sampling_enabled=not args.no_sampling,
        seed=args.seed,
        resume=bool(args.resume),
        conda_env=args.conda_env,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        carla_root=Path(args.carla_root).resolve() if args.carla_root else None,
        wall_time_limit_s=args.wall_time_limit,
        restart_min_cases=args.restart_min_cases,
        restart_max_cases=args.restart_max_cases,
        restart_wait_s=args.restart_wait,
        carla_ready_timeout_s=args.carla_ready_timeout,
        base_url=args.base_url,
        api_key=args.api_key,
        llm_strict=bool(args.llm_strict),
        llm_retries=args.llm_retries,
        llm_timeout_s=args.llm_timeout,
        scenario_prefix=args.scenario_prefix,
        dry_run=bool(args.dry_run),
    )
    director = ScenarioDirector(request)
    result = director.run()
    _log_cli(
        "summary "
        f"status={result.status} accepted={result.accepted_cases} attempted={result.attempted_cases} "
        f"state_path={result.state_path}"
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
