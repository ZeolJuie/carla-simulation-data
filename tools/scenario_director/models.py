from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from case_builder.models import DEFAULT_PLANNER_BASE_URL, DEFAULT_PLANNER_MODEL


@dataclass
class CatalogSpec:
    map_name: str
    catalog_path: Path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "map_name": self.map_name,
            "catalog_path": str(self.catalog_path),
        }


@dataclass
class CaseTask:
    map_name: str
    catalog_path: Path
    target_scene_id: str
    target_ego_action: str
    preferred_main_route_ids: List[str] = field(default_factory=list)
    excluded_main_route_ids: List[str] = field(default_factory=list)
    sampling_enabled: bool = True
    execute: bool = True
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "map_name": self.map_name,
            "catalog_path": str(self.catalog_path),
            "target_scene_id": self.target_scene_id,
            "target_ego_action": self.target_ego_action,
            "preferred_main_route_ids": list(self.preferred_main_route_ids),
            "excluded_main_route_ids": list(self.excluded_main_route_ids),
            "sampling_enabled": bool(self.sampling_enabled),
            "execute": bool(self.execute),
            "seed": int(self.seed),
        }


@dataclass
class DirectorRequest:
    output_root: Path
    state_path: Path
    target_count: int
    catalogs: List[CatalogSpec]
    planner: str = "heuristic"
    model: str = DEFAULT_PLANNER_MODEL
    execute: bool = False
    sampling_enabled: bool = True
    seed: int = 2026
    resume: bool = False
    conda_env: str = "carla"
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    carla_root: Optional[Path] = None
    wall_time_limit_s: Optional[float] = None
    restart_min_cases: int = 5
    restart_max_cases: int = 8
    restart_wait_s: float = 12.0
    carla_ready_timeout_s: float = 90.0
    base_url: str = DEFAULT_PLANNER_BASE_URL
    api_key: Optional[str] = None
    llm_strict: bool = False
    llm_retries: int = 2
    llm_timeout_s: float = 90.0
    scenario_prefix: str = "case"
    dry_run: bool = False


@dataclass
class DirectorResult:
    status: str
    accepted_cases: int
    attempted_cases: int
    state_path: Path
    carla_restarts: int = 0
    accepted_case_dirs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "accepted_cases": self.accepted_cases,
            "attempted_cases": self.attempted_cases,
            "state_path": str(self.state_path),
            "carla_restarts": self.carla_restarts,
            "accepted_case_dirs": list(self.accepted_case_dirs),
        }


@dataclass
class RouteRecord:
    route_id: str
    route_type: str
    ego_action: str
    route_length_m: float
    supported_behaviors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CellRecord:
    map_name: str
    scene_id: str
    scene_kind: str
    target_case_count: int
    routes: List[RouteRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "map_name": self.map_name,
            "scene_id": self.scene_id,
            "scene_kind": self.scene_kind,
            "target_case_count": self.target_case_count,
            "routes": [route.to_dict() for route in self.routes],
        }
