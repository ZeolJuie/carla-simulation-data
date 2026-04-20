from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from case_builder.catalog import CatalogInventory, CatalogInventoryError
from case_builder.models import ActorPlan, BuildRequest, BuiltCase, CasePlan, CaseSkeleton


class CaseCompilationError(ValueError):
    pass


VEHICLE_DENSITY_CONFIG = {
    "none": None,
    "low": {"local_count": 8, "global_count": 16},
    "medium": {"local_count": 12, "global_count": 32},
}

def instantiate_case(
    *,
    request: BuildRequest,
    skeleton: CaseSkeleton,
    plan: CasePlan,
    inventory: CatalogInventory,
) -> List[ActorPlan]:
    actors = [actor for actor in plan.support_actors]
    actors.append(
        ActorPlan(
            role="ego",
            route_id=skeleton.main_route.route_id,
            route_type=skeleton.main_route.route_type,
            behavior=plan.ego_behavior,
            quantity=1,
            spawn_time=plan.ego_spawn_time,
            ego=bool(plan.sampling_enabled),
            ego_member_index=0 if plan.sampling_enabled else None,
        )
    )
    return actors


def compile_case_config(
    *,
    request: BuildRequest,
    skeleton: CaseSkeleton,
    plan: CasePlan,
    actors: Sequence[ActorPlan],
    sampling_output_root,
    sequence_id,
    sampling_policy_path,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "scenario_name": plan.scenario_name,
        "step_length": 0.05,
        "map": {
            "map_name": skeleton.map_name,
            "map_path": "Carla/Maps/",
            "carla_timeout": 30.0,
            "scene_id": skeleton.scene_id,
            "scene_catalog_path": skeleton.catalog_path,
            "draw_obstacles": False,
        },
        "weather": {
            "enabled": True,
            "seed": request.seed + 101,
        },
        "walker": {
            "pedestrian_seed": request.seed,
            "default_speed": 1.2,
            "despawn_on_arrival": True,
            "waypoint_threshold": 1.0,
            "draw_bounding_boxes": False,
            "ped_spawner": [render_actor_spawner(actor) for actor in actors],
        },
        "vehicle": {
            "vehicle_seed": request.seed + 17,
            "no_bikes": False,
        },
    }
    if plan.sampling_enabled:
        config["sampling"] = {
            "enabled": True,
            "sensor_rig": "config/sensors/default_multimodal.toml",
            "policy": str(sampling_policy_path),
            "sequence_id": str(sequence_id),
            "output_root": str(sampling_output_root),
        }

    vehicle_cfg = VEHICLE_DENSITY_CONFIG[plan.vehicle_density]
    if vehicle_cfg is not None:
        config["vehicle"]["background_population"] = {
            "enabled": True,
            "seed": request.seed + 31,
            **vehicle_cfg,
        }

    return config


def render_actor_spawner(actor: ActorPlan) -> Dict[str, Any]:
    spawner = {
        "behavior": actor.behavior,
        "route_id": actor.route_id,
        "quantity": actor.quantity,
        "spawn_time": actor.spawn_time,
    }
    if actor.formation is not None:
        spawner["formation"] = actor.formation
    if actor.spawn_interval is not None:
        spawner["spawn_interval"] = actor.spawn_interval
    if actor.ego:
        spawner["ego"] = True
        spawner["ego_member_index"] = int(actor.ego_member_index or 0)
    return spawner


def validate_case_config(config: Dict[str, Any], inventory: CatalogInventory) -> None:
    map_cfg = config.get("map") or {}
    scene_id = map_cfg.get("scene_id")
    if scene_id is None:
        raise CaseCompilationError("map.scene_id is required.")
    inventory.get_scene(str(scene_id))

    ego_count = 0
    walker_cfg = config.get("walker") or {}
    spawners = walker_cfg.get("ped_spawner") or []
    if not spawners:
        raise CaseCompilationError("walker.ped_spawner must not be empty.")

    for index, spawner in enumerate(spawners):
        behavior = spawner.get("behavior")
        route_id = spawner.get("route_id")
        route_type = spawner.get("route_type")
        quantity = int(spawner.get("quantity", 1))
        formation = spawner.get("formation")
        if behavior is None:
            raise CaseCompilationError(f"walker.ped_spawner[{index}] missing behavior.")
        if route_id is None and route_type is None:
            raise CaseCompilationError(
                f"walker.ped_spawner[{index}] must define route_id or route_type in scene-based mode."
            )

        if route_id is not None:
            route = inventory.get_route(str(scene_id), str(route_id))
            if behavior not in route.supported_behaviors:
                raise CaseCompilationError(
                    f"walker.ped_spawner[{index}] behavior={behavior!r} is not supported by route_id={route_id!r}."
                )
            if route_type is not None and route.route_type != route_type:
                raise CaseCompilationError(
                    f"walker.ped_spawner[{index}] route_type does not match route_id."
                )
        else:
            support = inventory.find_support_routes(str(scene_id), behavior=str(behavior))
            if not any(route.route_type == route_type for route in support):
                raise CaseCompilationError(
                    f"walker.ped_spawner[{index}] has no route for behavior={behavior!r} route_type={route_type!r}."
                )

        if formation is not None:
            if formation not in {"abreast", "follow"}:
                raise CaseCompilationError(f"walker.ped_spawner[{index}] has unsupported formation.")
            if route_id is None:
                raise CaseCompilationError(
                    f"walker.ped_spawner[{index}] formation requires route_id."
                )
            if quantity not in (2, 3):
                raise CaseCompilationError(
                    f"walker.ped_spawner[{index}] formation only supports quantity=2 or quantity=3."
                )
        if spawner.get("ego", False):
            ego_count += 1

    sampling_enabled = bool((config.get("sampling") or {}).get("enabled", False))
    if sampling_enabled and ego_count != 1:
        raise CaseCompilationError("sampling.enabled=true requires exactly one ego pedestrian.")


def render_toml(data: Dict[str, Any]) -> str:
    lines: List[str] = []
    _render_table(lines, [], data)
    return "\n".join(_strip_trailing_blanks(lines)).rstrip() + "\n"


def _render_table(lines: List[str], path: List[str], data: Dict[str, Any]) -> None:
    scalar_items = []
    dict_items = []
    list_table_items = []
    for key, value in data.items():
        if isinstance(value, dict):
            dict_items.append((key, value))
        elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            list_table_items.append((key, value))
        else:
            scalar_items.append((key, value))

    if path:
        lines.append(f"[{'.'.join(path)}]")
    for key, value in scalar_items:
        lines.append(f"{key} = {_format_toml_value(value)}")

    if dict_items or list_table_items:
        if lines and lines[-1] != "":
            lines.append("")

    for index, (key, value) in enumerate(dict_items):
        _render_table(lines, path + [key], value)
        if index != len(dict_items) - 1 or list_table_items:
            lines.append("")

    for key, value in list_table_items:
        for item_index, item in enumerate(value):
            lines.append(f"[[{'.'.join(path + [key])}]]")
            for item_key, item_value in item.items():
                if isinstance(item_value, dict):
                    raise CaseCompilationError("Nested tables inside array-of-table items are not supported.")
                lines.append(f"{item_key} = {_format_toml_value(item_value)}")
            if item_index != len(value) - 1:
                lines.append("")


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        raise CaseCompilationError("TOML renderer does not support null values.")
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    raise CaseCompilationError(f"Unsupported TOML value type: {type(value)!r}")


def _strip_trailing_blanks(lines: List[str]) -> List[str]:
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def write_case_outputs(*, built_case: BuiltCase) -> None:
    config_dir = built_case.case_dir / "config"
    simulation_dir = built_case.case_dir / "simulation"
    sampling_dir = built_case.case_dir / "sampling"
    visualization_dir = built_case.case_dir / "visualization"
    logs_dir = built_case.case_dir / "logs"

    config_dir.mkdir(parents=True, exist_ok=True)
    simulation_dir.mkdir(parents=True, exist_ok=True)
    sampling_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    built_case.scenario_path.write_text(
        render_toml(built_case.scenario_config),
        encoding="utf-8",
    )
    built_case.description_path.write_text(
        built_case.plan.description.rstrip() + "\n",
        encoding="utf-8",
    )
    built_case.metadata_path.write_text(
        json.dumps(built_case.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if built_case.plan.sampling_enabled:
        _write_sampling_policy(built_case)
    built_case.manifest_path.write_text(
        json.dumps(
            {
                "case_id": built_case.case_id,
                "case_index": built_case.case_index,
                "map_name": built_case.skeleton.map_name,
                "case_dir": str(built_case.case_dir),
                "config_dir": str(config_dir),
                "simulation_dir": str(simulation_dir),
                "sampling_dir": str(sampling_dir),
                "visualization_dir": str(visualization_dir),
                "logs_dir": str(logs_dir),
                "scenario_name": built_case.plan.scenario_name,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_sampling_policy(built_case: BuiltCase) -> None:
    source_path = Path(__file__).resolve().parents[2] / "config" / "sampling" / "default.toml"
    policy: Dict[str, Any] = {}
    if source_path.exists():
        try:
            import tomllib  # py311+
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib
        policy = tomllib.loads(source_path.read_text(encoding="utf-8"))

    policy["output_root"] = str(built_case.case_dir.parent)
    policy["sequence_id"] = built_case.case_id
    policy["reuse_existing_root"] = True
    built_case.sampling_policy_path.write_text(
        render_toml(policy),
        encoding="utf-8",
    )
