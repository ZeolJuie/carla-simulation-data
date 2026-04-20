from __future__ import annotations

import json
import math
import os
import random
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

from case_builder.builder import CaseBuilder
from case_builder.models import BuildRequest
from scenario_director.models import (
    CaseTask,
    CatalogSpec,
    CellRecord,
    DirectorRequest,
    DirectorResult,
    RouteRecord,
)

REVERSE_ROUTE_SUFFIX = "__reverse"


def _log_director(message: str) -> None:
    print(f"[director] {message}", flush=True)


def _log_coverage(message: str) -> None:
    print(f"[coverage] {message}", flush=True)


def _normalize_angle_deg(angle: float) -> float:
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle


def _heading_deg(p1, p2) -> float:
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    return math.degrees(math.atan2(dy, dx))


def ego_action_bucket(route: Dict[str, object]) -> str:
    route_type = str(route.get("route_type", ""))
    if route_type == "sidewalk_through":
        return "sidewalk_straight"
    if route_type == "crosswalk_cross":
        return "crosswalk_cross"
    if route_type == "unmarked_junction_cross":
        return "unmarked_cross"
    if route_type == "junction_through":
        return "junction_straight"
    if route_type == "junction_u_turn":
        return "u_turn"

    if route_type == "junction_turn":
        points = route.get("path_waypoints") or []
        if len(points) >= 2:
            start_heading = _heading_deg(points[0], points[min(1, len(points) - 1)])
            end_heading = _heading_deg(points[max(0, len(points) - 2)], points[-1])
            delta = _normalize_angle_deg(end_heading - start_heading)
            if delta > 35.0:
                return "left_turn"
            if delta < -35.0:
                return "right_turn"
        return "junction_turn"

    return route_type or "unknown"


def _synthetic_reverse_route(route: Dict[str, object]) -> Dict[str, object]:
    reversed_route = dict(route)
    reversed_route["route_id"] = f"{route['route_id']}{REVERSE_ROUTE_SUFFIX}"
    path_waypoints = route.get("path_waypoints")
    if isinstance(path_waypoints, list):
        reversed_route["path_waypoints"] = list(reversed(path_waypoints))
    return reversed_route


def _cell_target_case_count(scene_kind: str, route_count: int, action_bucket_count: int) -> int:
    score = 1
    if route_count >= 8:
        score += 1
    if action_bucket_count >= 2:
        score += 1
    if scene_kind == "crossing":
        score += 1
    return min(score, 3)


class ScenarioDirector:
    def __init__(self, request: DirectorRequest):
        self.request = request
        self.rng = random.Random(request.seed)
        self.state = self._load_or_init_state()
        self.cells = self._load_cells(request.catalogs)
        self.carla_restarts = 0
        self._attempts_since_restart = 0
        self._proactive_restart_interval = self._sample_restart_interval()

    def run(self) -> DirectorResult:
        accepted_case_dirs: List[str] = []
        _log_director(
            "start "
            f"target={self.request.target_count} catalogs={len(self.request.catalogs)} "
            f"planner={self.request.planner} execute={self.request.execute} "
            f"sampling={self.request.sampling_enabled} output_root={self.request.output_root}"
        )

        while self.state["accepted_cases"] < self.request.target_count:
            self._maybe_restart_carla_proactively()
            task = self._select_next_task()
            if task is None:
                _log_director("stop no eligible cell/task remains")
                break

            self.state["attempted_cases"] += 1
            self.state["last_task"] = task.to_dict()
            self._save_state()
            _log_director(
                "attempt "
                f"{self.state['attempted_cases']} accepted={self.state['accepted_cases']}/{self.request.target_count} "
                f"map={task.map_name} scene={task.target_scene_id} ego_action={task.target_ego_action} "
                f"preferred_routes={len(task.preferred_main_route_ids)} excluded_routes={len(task.excluded_main_route_ids)} "
                f"seed={task.seed}"
            )

            if self.request.dry_run:
                _log_director("dry-run selected one task and stopped before case build")
                break

            build_request = BuildRequest(
                catalog_path=task.catalog_path,
                output_root=self.request.output_root,
                planner=self.request.planner,
                seed=task.seed,
                map_name=task.map_name,
                sampling_enabled=task.sampling_enabled,
                execute=task.execute,
                conda_env=self.request.conda_env,
                carla_host=self.request.carla_host,
                carla_port=self.request.carla_port,
                wall_time_limit_s=self.request.wall_time_limit_s,
                scenario_prefix=self.request.scenario_prefix,
                model=self.request.model,
                base_url=self.request.base_url,
                api_key=self.request.api_key,
                llm_strict=self.request.llm_strict,
                llm_retries=self.request.llm_retries,
                llm_timeout_s=self.request.llm_timeout_s,
                target_scene_id=task.target_scene_id,
                preferred_main_route_ids=list(task.preferred_main_route_ids),
                excluded_main_route_ids=list(task.excluded_main_route_ids),
            )
            builder = CaseBuilder(build_request)
            built_case = builder.build()
            execution_result = builder.execute(built_case) if task.execute else None

            accepted = (not task.execute) or (execution_result is not None and execution_result.returncode == 0)
            self._update_state_after_case(task, built_case, accepted=accepted)
            self._save_state()
            if accepted:
                self._attempts_since_restart += 1
                accepted_case_dirs.append(str(built_case.case_dir))
                cell_state = self.state["cells"][self._cell_key(task.map_name, task.target_scene_id)]
                target_cases = self.cells[self._cell_key(task.map_name, task.target_scene_id)].target_case_count
                _log_coverage(
                    "accepted "
                    f"map={task.map_name} scene={task.target_scene_id} main_route={built_case.skeleton.main_route.route_id} "
                    f"ego_action={task.target_ego_action} scene_progress={cell_state['accepted_cases']}/{target_cases} "
                    f"map_cases={self.state['maps'][task.map_name]['accepted_cases']}"
                )
            else:
                reason = "run_failed"
                if execution_result is not None and execution_result.timed_out:
                    reason = "timeout"
                unhealthy = self._is_server_unhealthy(execution_result)
                if unhealthy:
                    reason = "server_unhealthy"
                _log_director(
                    "reject "
                    f"map={task.map_name} scene={task.target_scene_id} ego_action={task.target_ego_action} "
                    f"reason={reason}"
                )
                if unhealthy:
                    self._restart_carla(reason="detected unhealthy CARLA server from run log")

        status = "completed" if self.state["accepted_cases"] >= self.request.target_count else "stopped"
        _log_director(
            "summary "
            f"status={status} accepted={self.state['accepted_cases']} "
            f"attempted={self.state['attempted_cases']} state={self.request.state_path}"
        )
        return DirectorResult(
            status=status,
            accepted_cases=int(self.state["accepted_cases"]),
            attempted_cases=int(self.state["attempted_cases"]),
            state_path=self.request.state_path,
            carla_restarts=self.carla_restarts,
            accepted_case_dirs=accepted_case_dirs,
        )

    def _sample_restart_interval(self) -> int:
        low = max(1, int(self.request.restart_min_cases))
        high = max(low, int(self.request.restart_max_cases))
        return self.rng.randint(low, high)

    def _maybe_restart_carla_proactively(self) -> None:
        if not self.request.execute:
            return
        if self.request.carla_root is None:
            return
        if self._attempts_since_restart < self._proactive_restart_interval:
            return
        self._restart_carla(
            reason=(
                f"proactive restart after {self._attempts_since_restart} cases "
                f"(target interval={self._proactive_restart_interval})"
            )
        )

    def _restart_carla(self, *, reason: str) -> None:
        carla_root = self.request.carla_root
        if carla_root is None:
            _log_director(f"skip restart reason={reason} because carla_root is not configured")
            return

        _log_director(f"restart start reason={reason}")
        self._stop_carla_processes()
        launcher = carla_root / "CarlaUE4.sh"
        env = os.environ.copy()
        log_path = Path("/tmp/carla_scenario_director.log")
        log_handle = log_path.open("a", encoding="utf-8")
        try:
            subprocess.Popen(
                [str(launcher), "-RenderOffScreen", "-quality-level=Epic"],
                cwd=str(carla_root),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        finally:
            log_handle.close()

        self._wait_for_carla_ready()
        self.carla_restarts += 1
        self._attempts_since_restart = 0
        self._proactive_restart_interval = self._sample_restart_interval()
        _log_director(
            "restart done "
            f"count={self.carla_restarts} next_interval={self._proactive_restart_interval}"
        )

    def _stop_carla_processes(self) -> None:
        for pattern in ("CarlaUE4-Linux-Shipping", "CarlaUE4.sh"):
            subprocess.run(
                ["pkill", "-f", pattern],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        time.sleep(2.0)

    def _wait_for_carla_ready(self) -> None:
        deadline = time.time() + float(self.request.carla_ready_timeout_s)
        while time.time() < deadline:
            if self._tcp_ready():
                time.sleep(float(self.request.restart_wait_s))
                return
            time.sleep(1.0)
        raise RuntimeError(
            f"CARLA did not become ready on {self.request.carla_host}:{self.request.carla_port} "
            f"within {self.request.carla_ready_timeout_s:.1f}s after restart."
        )

    def _tcp_ready(self) -> bool:
        try:
            with socket.create_connection(
                (self.request.carla_host, int(self.request.carla_port)),
                timeout=1.0,
            ):
                return True
        except OSError:
            return False

    @staticmethod
    def _is_server_unhealthy(execution_result) -> bool:
        if execution_result is None:
            return False
        try:
            log_text = Path(execution_result.log_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False

        unhealthy_markers = (
            "time-out of 30000ms while waiting for the simulator",
            "world.apply_settings",
            "self.world.apply_settings(settings)",
            "self.world.tick()",
            "while waiting for the simulator, make sure the simulator is ready and connected",
        )
        return any(marker in log_text for marker in unhealthy_markers)

    def _load_cells(self, catalogs: List[CatalogSpec]) -> Dict[str, CellRecord]:
        cells: Dict[str, CellRecord] = {}
        for catalog_spec in catalogs:
            data = json.loads(Path(catalog_spec.catalog_path).read_text(encoding="utf-8"))
            for raw_cell in data.get("cells", []):
                if not raw_cell.get("attributes", {}).get("usable", False):
                    continue
                raw_routes = raw_cell.get("routes") or []
                if not raw_routes:
                    continue
                routes = []
                for route in raw_routes:
                    for candidate_route in (route, _synthetic_reverse_route(route)):
                        route_id = str(candidate_route["route_id"])
                        if route_id.endswith(REVERSE_ROUTE_SUFFIX) and str(route["route_id"]).endswith(REVERSE_ROUTE_SUFFIX):
                            continue
                        routes.append(
                            RouteRecord(
                                route_id=route_id,
                                route_type=str(candidate_route.get("route_type", "")),
                                ego_action=ego_action_bucket(candidate_route),
                                route_length_m=float(candidate_route.get("route_length_m", 0.0)),
                                supported_behaviors=list(
                                    candidate_route.get("attributes", {}).get("supported_behaviors", [])
                                ),
                            )
                        )
                action_bucket_count = len({route.ego_action for route in routes})
                target_case_count = _cell_target_case_count(
                    str(raw_cell.get("kind", "unknown")),
                    len(routes),
                    action_bucket_count,
                )
                cell = CellRecord(
                    map_name=str(data["map_name"]),
                    scene_id=str(raw_cell["id"]),
                    scene_kind=str(raw_cell.get("kind", "unknown")),
                    target_case_count=target_case_count,
                    routes=routes,
                )
                cells[self._cell_key(cell.map_name, cell.scene_id)] = cell
        return cells

    def _select_next_task(self) -> CaseTask | None:
        candidates: List[Tuple[float, CellRecord, str, List[str], List[str]]] = []

        for cell_key, cell in self.cells.items():
            cell_state = self.state["cells"].setdefault(
                cell_key,
                {
                    "accepted_cases": 0,
                    "failed_cases": 0,
                    "used_main_route_ids": [],
                    "ego_action_counts": {},
                },
            )
            accepted_cases = int(cell_state["accepted_cases"])
            if accepted_cases >= cell.target_case_count:
                continue

            unused_actions = sorted({
                route.ego_action
                for route in cell.routes
                if int(cell_state["ego_action_counts"].get(route.ego_action, 0)) == 0
            })
            if unused_actions:
                target_action = self.rng.choice(unused_actions)
            else:
                target_action = min(
                    {route.ego_action for route in cell.routes},
                    key=lambda action: int(cell_state["ego_action_counts"].get(action, 0)),
                )

            preferred_routes = [
                route.route_id
                for route in cell.routes
                if route.ego_action == target_action and route.route_id not in set(cell_state["used_main_route_ids"])
            ]
            excluded_routes = list(cell_state["used_main_route_ids"])

            map_state = self.state["maps"].setdefault(cell.map_name, {"accepted_cases": 0})
            map_gap = max(self.request.target_count - int(map_state["accepted_cases"]), 0)
            cell_gap = max(cell.target_case_count - accepted_cases, 0)
            action_gap = len(unused_actions)
            route_gap = len(preferred_routes)
            failure_penalty = int(cell_state["failed_cases"])
            score = 10 * cell_gap + 4 * action_gap + 2 * route_gap + 0.01 * map_gap - 3 * failure_penalty
            candidates.append((score, cell, target_action, preferred_routes, excluded_routes))

        if not candidates:
            return None

        positive_weights = [max(score, 1.0) for score, *_ in candidates]
        _, selected_cell, selected_action, selected_preferred_routes, selected_excluded_routes = self.rng.choices(
            population=candidates,
            weights=positive_weights,
            k=1,
        )[0]

        seed = self.rng.randint(0, 2**31 - 1)
        return CaseTask(
            map_name=selected_cell.map_name,
            catalog_path=self._catalog_path_for_map(selected_cell.map_name),
            target_scene_id=selected_cell.scene_id,
            target_ego_action=selected_action,
            preferred_main_route_ids=selected_preferred_routes,
            excluded_main_route_ids=selected_excluded_routes,
            sampling_enabled=self.request.sampling_enabled,
            execute=self.request.execute,
            seed=seed,
        )

    def _update_state_after_case(self, task: CaseTask, built_case, *, accepted: bool) -> None:
        cell_key = self._cell_key(task.map_name, task.target_scene_id)
        cell_state = self.state["cells"].setdefault(
            cell_key,
            {
                "accepted_cases": 0,
                "failed_cases": 0,
                "used_main_route_ids": [],
                "ego_action_counts": {},
            },
        )
        map_state = self.state["maps"].setdefault(task.map_name, {"accepted_cases": 0})

        if accepted:
            cell_state["accepted_cases"] += 1
            map_state["accepted_cases"] += 1
            main_route_id = built_case.skeleton.main_route.route_id
            if main_route_id not in cell_state["used_main_route_ids"]:
                cell_state["used_main_route_ids"].append(main_route_id)
            action_counts = cell_state["ego_action_counts"]
            action_counts[task.target_ego_action] = int(action_counts.get(task.target_ego_action, 0)) + 1
            self.state["accepted_cases"] += 1
        else:
            cell_state["failed_cases"] += 1

    def _load_or_init_state(self) -> Dict[str, object]:
        if self.request.resume and self.request.state_path.exists():
            return json.loads(self.request.state_path.read_text(encoding="utf-8"))
        return {
            "accepted_cases": 0,
            "attempted_cases": 0,
            "maps": {},
            "cells": {},
            "last_task": None,
        }

    def _save_state(self) -> None:
        self.request.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.request.state_path.write_text(
            json.dumps(self.state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _catalog_path_for_map(self, map_name: str) -> Path:
        for catalog_spec in self.request.catalogs:
            if catalog_spec.map_name == map_name:
                return catalog_spec.catalog_path
        raise KeyError(map_name)

    @staticmethod
    def _cell_key(map_name: str, scene_id: str) -> str:
        return f"{map_name}::{scene_id}"
