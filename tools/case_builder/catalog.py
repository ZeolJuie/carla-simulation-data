from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from case_builder.models import CaseSkeleton, RouteSummary, SceneSummary

REVERSE_ROUTE_SUFFIX = "__reverse"


class CatalogInventoryError(ValueError):
    pass


class CatalogInventory:
    def __init__(self, catalog_path: Path):
        self.catalog_path = Path(catalog_path).resolve()
        with self.catalog_path.open(encoding="utf-8") as handle:
            catalog = json.load(handle)

        self.map_name = str(catalog.get("map_name") or "")
        if not self.map_name:
            raise CatalogInventoryError("Catalog missing map_name.")

        self.scenes: List[SceneSummary] = []
        self.scenes_by_id: Dict[str, SceneSummary] = {}
        self.routes_by_scene: Dict[str, Dict[str, RouteSummary]] = {}

        for cell in catalog.get("cells", []):
            if not cell.get("attributes", {}).get("usable", False):
                continue
            raw_routes = cell.get("routes") or []
            if not raw_routes:
                continue

            routes = []
            for route in raw_routes:
                routes.append(_route_summary_from_raw(route))
                routes.append(_reverse_route_summary_from_raw(route))

            scene = SceneSummary(
                scene_id=str(cell["id"]),
                scene_kind=str(cell.get("kind", "unknown")),
                bounds=dict(cell.get("bounds", {})),
                routes=routes,
            )
            self.scenes.append(scene)
            self.scenes_by_id[scene.scene_id] = scene
            self.routes_by_scene[scene.scene_id] = {route.route_id: route for route in routes}

        if not self.scenes:
            raise CatalogInventoryError("Catalog does not contain any usable scenes with routes.")

    def select_skeleton(
        self,
        *,
        rng: random.Random,
        scene_kind: Optional[str] = None,
        require_sampling: bool = False,
        target_scene_id: Optional[str] = None,
        preferred_main_route_ids: Optional[List[str]] = None,
        excluded_main_route_ids: Optional[List[str]] = None,
    ) -> CaseSkeleton:
        candidate_scenes = [
            scene
            for scene in self.scenes
            if (scene_kind is None or scene.scene_kind == scene_kind)
            and (target_scene_id is None or scene.scene_id == target_scene_id)
        ]
        if not candidate_scenes:
            raise CatalogInventoryError(
                f"No usable scenes found for scene_kind={scene_kind!r} target_scene_id={target_scene_id!r}."
            )

        weighted_scenes = []
        for scene in candidate_scenes:
            weight = max(len(scene.routes), 1)
            weighted_scenes.extend([scene] * weight)
        scene = rng.choice(weighted_scenes)

        main_route = self._sample_route(
            scene.routes,
            rng=rng,
            prefer_long=require_sampling,
            preferred_route_ids=preferred_main_route_ids,
            excluded_route_ids=excluded_main_route_ids,
        )
        if scene.scene_kind == "sidewalk":
            support_routes = list(scene.routes)
        else:
            support_routes = [route for route in scene.routes if route.route_id != main_route.route_id]

        return CaseSkeleton(
            map_name=self.map_name,
            catalog_path=str(self.catalog_path),
            scene_id=scene.scene_id,
            scene_kind=scene.scene_kind,
            main_route=main_route,
            support_routes=support_routes,
        )

    @staticmethod
    def _sample_route(
        routes: List[RouteSummary],
        *,
        rng: random.Random,
        prefer_long: bool,
        preferred_route_ids: Optional[List[str]] = None,
        excluded_route_ids: Optional[List[str]] = None,
    ) -> RouteSummary:
        if not routes:
            raise CatalogInventoryError("Cannot sample route from empty route list.")

        preferred_route_ids = list(preferred_route_ids or [])
        excluded = set(excluded_route_ids or [])

        candidate_routes = [route for route in routes if route.route_id not in excluded]
        if not candidate_routes:
            candidate_routes = list(routes)

        preferred_routes = [route for route in candidate_routes if route.route_id in preferred_route_ids]
        if preferred_routes:
            candidate_routes = preferred_routes

        if not prefer_long:
            return rng.choice(candidate_routes)

        total = sum(max(route.route_length_m, 1.0) for route in candidate_routes)
        threshold = rng.uniform(0.0, total)
        cumulative = 0.0
        for route in candidate_routes:
            cumulative += max(route.route_length_m, 1.0)
            if cumulative >= threshold:
                return route
        return candidate_routes[-1]

    def get_scene(self, scene_id: str) -> SceneSummary:
        try:
            return self.scenes_by_id[scene_id]
        except KeyError as exc:
            raise CatalogInventoryError(f"Unknown scene_id {scene_id!r}.") from exc

    def get_route(self, scene_id: str, route_id: str) -> RouteSummary:
        try:
            return self.routes_by_scene[scene_id][route_id]
        except KeyError as exc:
            raise CatalogInventoryError(f"Unknown route_id {route_id!r} for scene {scene_id!r}.") from exc

    def find_support_routes(
        self,
        scene_id: str,
        *,
        behavior: str,
        exclude_route_ids: Optional[List[str]] = None,
    ) -> List[RouteSummary]:
        excluded = set(exclude_route_ids or [])
        scene = self.get_scene(scene_id)
        return [
            route
            for route in scene.routes
            if route.route_id not in excluded and behavior in route.supported_behaviors
        ]


def _route_summary_from_raw(route: Dict[str, object]) -> RouteSummary:
    return RouteSummary(
        route_id=str(route["route_id"]),
        route_type=str(route["route_type"]),
        route_length_m=float(route.get("route_length_m", 0.0)),
        supported_behaviors=list(route.get("attributes", {}).get("supported_behaviors", [])),
        crossing_required=bool(route.get("attributes", {}).get("crossing_required", False)),
        spawn_point_xy=_first_xy(route.get("path_waypoints")),
    )


def _reverse_route_summary_from_raw(route: Dict[str, object]) -> RouteSummary:
    return RouteSummary(
        route_id=f"{route['route_id']}{REVERSE_ROUTE_SUFFIX}",
        route_type=str(route["route_type"]),
        route_length_m=float(route.get("route_length_m", 0.0)),
        supported_behaviors=list(route.get("attributes", {}).get("supported_behaviors", [])),
        crossing_required=bool(route.get("attributes", {}).get("crossing_required", False)),
        spawn_point_xy=_last_xy(route.get("path_waypoints")),
    )


def _first_xy(points) -> Optional[List[float]]:
    if not points:
        return None
    first_point = points[0]
    if isinstance(first_point, list) and len(first_point) >= 2:
        return [float(first_point[0]), float(first_point[1])]
    return None


def _last_xy(points) -> Optional[List[float]]:
    if not points:
        return None
    last_point = points[-1]
    if isinstance(last_point, list) and len(last_point) >= 2:
        return [float(last_point[0]), float(last_point[1])]
    return None
