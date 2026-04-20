from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BuildRequest:
    catalog_path: Path
    output_root: Path
    planner: str = "heuristic"
    seed: int = 2026
    map_name: Optional[str] = None
    scene_kind: Optional[str] = None
    sampling_enabled: bool = True
    execute: bool = False
    duration: Optional[float] = None
    conda_env: str = "carla"
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    scenario_prefix: str = "case"
    model: str = "qwen-plus"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: Optional[str] = None
    llm_strict: bool = False
    llm_retries: int = 2
    llm_timeout_s: float = 90.0
    target_scene_id: Optional[str] = None
    preferred_main_route_ids: List[str] = field(default_factory=list)
    excluded_main_route_ids: List[str] = field(default_factory=list)
    wall_time_limit_s: Optional[float] = None


@dataclass
class RouteSummary:
    route_id: str
    route_type: str
    route_length_m: float
    supported_behaviors: List[str]
    crossing_required: bool
    spawn_point_xy: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneSummary:
    scene_id: str
    scene_kind: str
    bounds: Dict[str, Any]
    routes: List[RouteSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_kind": self.scene_kind,
            "bounds": self.bounds,
            "routes": [route.to_dict() for route in self.routes],
        }


@dataclass
class CaseSkeleton:
    map_name: str
    catalog_path: str
    scene_id: str
    scene_kind: str
    main_route: RouteSummary
    support_routes: List[RouteSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "map_name": self.map_name,
            "catalog_path": self.catalog_path,
            "scene_id": self.scene_id,
            "scene_kind": self.scene_kind,
            "main_route": self.main_route.to_dict(),
            "support_routes": [route.to_dict() for route in self.support_routes],
        }


@dataclass
class CasePlan:
    scenario_name: str
    description: str
    ego_behavior: str
    ego_spawn_time: float
    vehicle_density: str
    sampling_enabled: bool
    support_actors: List["ActorPlan"] = field(default_factory=list)
    planner_source: str = "unknown"
    planner_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "description": self.description,
            "ego_behavior": self.ego_behavior,
            "ego_spawn_time": self.ego_spawn_time,
            "vehicle_density": self.vehicle_density,
            "sampling_enabled": self.sampling_enabled,
            "support_actors": [actor.to_dict() for actor in self.support_actors],
            "planner_source": self.planner_source,
            "planner_error": self.planner_error,
        }


@dataclass
class ActorPlan:
    role: str
    route_id: str
    route_type: str
    behavior: str
    quantity: int = 1
    formation: Optional[str] = None
    spawn_time: float = 0.0
    spawn_interval: Optional[float] = None
    ego: bool = False
    ego_member_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BuiltCase:
    case_id: str
    case_index: int
    case_dir: Path
    scenario_path: Path
    description_path: Path
    metadata_path: Path
    manifest_path: Path
    sampling_policy_path: Path
    skeleton: CaseSkeleton
    plan: CasePlan
    actors: List[ActorPlan]
    scenario_config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_index": self.case_index,
            "case_dir": str(self.case_dir),
            "scenario_path": str(self.scenario_path),
            "description_path": str(self.description_path),
            "metadata_path": str(self.metadata_path),
            "manifest_path": str(self.manifest_path),
            "sampling_policy_path": str(self.sampling_policy_path),
            "skeleton": self.skeleton.to_dict(),
            "plan": self.plan.to_dict(),
            "actors": [actor.to_dict() for actor in self.actors],
            "scenario_config": self.scenario_config,
        }


@dataclass
class ExecutionResult:
    command: List[str]
    returncode: int
    log_path: Path
    elapsed_s: float
    timed_out: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "returncode": self.returncode,
            "log_path": str(self.log_path),
            "elapsed_s": self.elapsed_s,
            "timed_out": self.timed_out,
        }
