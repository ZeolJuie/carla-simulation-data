from __future__ import annotations

import json
import os
import random
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from case_builder.catalog import CatalogInventory
from case_builder.models import ActorPlan, BuildRequest, CasePlan, CaseSkeleton


SUPPORTED_GROUP_FORMATIONS = ("abreast", "follow")
SUPPORTED_VEHICLE_DENSITIES = ("low", "medium", "high")
MAX_TOTAL_PEDESTRIANS = 16
MAX_SPAWNER_COUNT = 16
MAX_SUPPORT_GROUP_INTERVAL_BEFORE_EGO_S = 0.85
EGO_DELAY_AFTER_LATEST_SUPPORT_RANGE_S = (1.0, 5.0)
EGO_SPAWN_TIME_MAX_S = 12.0

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class SemanticPlannerError(ValueError):
    pass


def build_planner(request: BuildRequest):
    if request.planner == "llm":
        return OpenAICompatibleSemanticPlanner(request)
    if request.planner == "heuristic":
        return HeuristicSemanticPlanner()
    raise SemanticPlannerError(f"Unsupported planner {request.planner!r}.")


def _log_planner(message: str) -> None:
    print(f"[planner] {message}", flush=True)


class HeuristicSemanticPlanner:
    def plan(
        self,
        *,
        request: BuildRequest,
        skeleton: CaseSkeleton,
        inventory: CatalogInventory,
        rng: random.Random,
    ) -> CasePlan:
        ego_behavior = _weighted_choice(
            rng,
            skeleton.main_route.supported_behaviors,
            preferred=_preferred_ego_behaviors(skeleton.main_route.route_type),
        )
        support_actors = _build_support_actors(
            skeleton=skeleton,
            inventory=inventory,
            rng=rng,
        )

        vehicle_density = rng.choices(
            population=list(SUPPORTED_VEHICLE_DENSITIES),
            weights=[0.30, 0.40, 0.30],
            k=1,
        )[0]
        scenario_name = _scenario_name(request.scenario_prefix, skeleton, support_actors)
        ego_spawn_time = _ego_spawn_time(support_actors, rng)
        description = _template_description(
            scene_kind=skeleton.scene_kind,
            main_route_id=skeleton.main_route.route_id,
            main_route_type=skeleton.main_route.route_type,
            ego_behavior=ego_behavior,
            ego_spawn_time=ego_spawn_time,
            support_actors=support_actors,
            vehicle_density=vehicle_density,
        )
        return CasePlan(
            scenario_name=scenario_name,
            description=description,
            ego_behavior=ego_behavior,
            ego_spawn_time=ego_spawn_time,
            vehicle_density=vehicle_density,
            sampling_enabled=bool(request.sampling_enabled),
            support_actors=support_actors,
            planner_source="heuristic",
        )


class OpenAICompatibleSemanticPlanner:
    def __init__(self, request: BuildRequest):
        self.request = request
        self.fallback = HeuristicSemanticPlanner()

    def plan(
        self,
        *,
        request: BuildRequest,
        skeleton: CaseSkeleton,
        inventory: CatalogInventory,
        rng: random.Random,
    ) -> CasePlan:
        api_key = request.api_key or os.environ.get("CASE_BUILDER_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") \
            or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return self._handle_llm_failure(
                reason="missing API key; set CASE_BUILDER_API_KEY, DASHSCOPE_API_KEY, OPENAI_API_KEY, or pass --api-key",
                request=request,
                skeleton=skeleton,
                inventory=inventory,
                rng=rng,
            )

        support_route_specs = [
            {
                "route_id": route.route_id,
                "route_type": route.route_type,
                "supported_behaviors": route.supported_behaviors,
                "route_length_m": route.route_length_m,
            }
            for route in skeleton.support_routes
        ]
        payload = {
            "model": request.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are planning one executable social-navigation case. "
                        "The topology and route skeleton are already fixed. "
                        "Return one JSON object only. Do not emit markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "Choose a realistic semantic plan for this one case.",
                            "skeleton": skeleton.to_dict(),
                            "allowed_ego_behaviors": skeleton.main_route.supported_behaviors,
                            "available_support_routes": support_route_specs,
                            "allowed_vehicle_density": list(SUPPORTED_VEHICLE_DENSITIES),
                            "constraints": {
                                "total_pedestrians_lte": MAX_TOTAL_PEDESTRIANS,
                                "spawner_count_lte": MAX_SPAWNER_COUNT,
                                "group_count_max": 5,
                                "group_size_allowed": [2, 3],
                                "ego_must_be_unique": True,
                                "support_spawn_window_s": [0.0, 8.0],
                                "ego_spawn_window_s": [0.0, EGO_SPAWN_TIME_MAX_S],
                                "ego_delay_after_latest_support_s": list(EGO_DELAY_AFTER_LATEST_SUPPORT_RANGE_S),
                                "sidewalk_guidance": (
                                    "For sidewalk scenes, keep pedestrian density low, spread spawn times, "
                                    "and include bidirectional pedestrian flow when reverse routes are available."
                                ),
                            },
                            "allowed_group_formations": list(SUPPORTED_GROUP_FORMATIONS),
                            "output_schema": {
                                "scenario_name": "string slug",
                                "description": "short natural-language scene description",
                                "ego_behavior": "enum",
                                "ego_spawn_time": "ignored; ego timing is assigned by the generator",
                                "vehicle_density": "enum",
                                "support_actors": [
                                    {
                                        "role": "string",
                                        "route_id": "enum from available_support_routes",
                                        "route_type": "must match route_id",
                                        "behavior": "supported by route",
                                        "quantity": "1,2,3",
                                        "formation": "null or abreast/follow",
                                        "spawn_time": "float in [0, 8]",
                                        "spawn_interval": "float or null",
                                    }
                                ],
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.6,
            "max_tokens": 700,
        }
        attempts = max(1, int(request.llm_retries))
        last_error = "unknown LLM planner failure"
        for attempt in range(1, attempts + 1):
            try:
                response = _call_openai_compatible(
                    api_key=api_key,
                    base_url=request.base_url,
                    payload=payload,
                    timeout_s=float(request.llm_timeout_s),
                )
                content = _extract_content(response)
                raw_plan = _extract_json_object(content)
                scenario_name = str(raw_plan.get("scenario_name") or "").strip()
                if not scenario_name:
                    scenario_name = (
                        f"{request.scenario_prefix}_{skeleton.scene_id.lower()}_"
                        f"{skeleton.main_route.route_type.lower()}"
                    )
                support_actors = [_coerce_support_actor(actor) for actor in raw_plan.get("support_actors") or []]
                if skeleton.scene_kind == "sidewalk":
                    support_actors = _shape_sidewalk_support_actors(
                        support_actors=support_actors,
                        skeleton=skeleton,
                        rng=rng,
                    )
                ego_spawn_time = _ego_spawn_time(support_actors, rng)

                plan = CasePlan(
                    scenario_name=scenario_name,
                    description=str(raw_plan.get("description") or "").strip(),
                    ego_behavior=str(raw_plan.get("ego_behavior")),
                    ego_spawn_time=ego_spawn_time,
                    vehicle_density=str(raw_plan.get("vehicle_density")),
                    sampling_enabled=bool(request.sampling_enabled),
                    support_actors=support_actors,
                    planner_source="llm",
                )
                _validate_plan(plan, skeleton=skeleton, inventory=inventory)
                plan.description = _template_description(
                    scene_kind=skeleton.scene_kind,
                    main_route_id=skeleton.main_route.route_id,
                    main_route_type=skeleton.main_route.route_type,
                    ego_behavior=plan.ego_behavior,
                    ego_spawn_time=plan.ego_spawn_time,
                    support_actors=plan.support_actors,
                    vehicle_density=plan.vehicle_density,
                )
                _log_planner(
                    "llm plan accepted "
                    f"model={request.model} attempt={attempt}/{attempts} "
                    f"scene={skeleton.scene_id} main_route={skeleton.main_route.route_id}"
                )
                return plan
            except Exception as exc:
                last_error = _format_error(exc)
                _log_planner(
                    "llm attempt failed "
                    f"model={request.model} attempt={attempt}/{attempts} "
                    f"scene={skeleton.scene_id} error={last_error}"
                )

        return self._handle_llm_failure(
            reason=last_error,
            request=request,
            skeleton=skeleton,
            inventory=inventory,
            rng=rng,
        )

    def _handle_llm_failure(
        self,
        *,
        reason: str,
        request: BuildRequest,
        skeleton: CaseSkeleton,
        inventory: CatalogInventory,
        rng: random.Random,
    ) -> CasePlan:
        if request.llm_strict:
            raise SemanticPlannerError(f"LLM planner failed and fallback is disabled: {reason}")
        _log_planner(
            "fallback to heuristic "
            f"scene={skeleton.scene_id} main_route={skeleton.main_route.route_id} reason={reason}"
        )
        plan = self.fallback.plan(request=request, skeleton=skeleton, inventory=inventory, rng=rng)
        plan.planner_source = "heuristic_fallback"
        plan.planner_error = reason
        return plan


def _validate_plan(plan: CasePlan, *, skeleton: CaseSkeleton, inventory: CatalogInventory) -> None:
    if plan.ego_behavior not in skeleton.main_route.supported_behaviors:
        raise SemanticPlannerError("Planned ego_behavior is not supported by main route.")
    if plan.vehicle_density not in SUPPORTED_VEHICLE_DENSITIES:
        raise SemanticPlannerError("vehicle_density is invalid.")
    if not (0.0 <= float(plan.ego_spawn_time) <= EGO_SPAWN_TIME_MAX_S):
        raise SemanticPlannerError(f"ego_spawn_time must be within [0, {EGO_SPAWN_TIME_MAX_S:g}].")
    total_peds = 1
    group_count = 0
    for actor in plan.support_actors:
        route = inventory.get_route(skeleton.scene_id, actor.route_id)
        if actor.behavior not in route.supported_behaviors:
            raise SemanticPlannerError("Support actor behavior is not supported by selected route.")
        if route.route_type != actor.route_type:
            raise SemanticPlannerError("Support actor route_type does not match route_id.")
        if not (0.0 <= float(actor.spawn_time) <= 8.0):
            raise SemanticPlannerError("Support actor spawn_time must be within [0, 8].")
        total_peds += int(actor.quantity)
        if actor.quantity > 1:
            group_count += 1
            if actor.formation not in SUPPORTED_GROUP_FORMATIONS:
                raise SemanticPlannerError("Grouped support actor requires a valid formation.")
            if actor.quantity not in (2, 3):
                raise SemanticPlannerError("Grouped support actor quantity must be 2 or 3.")
    if total_peds > MAX_TOTAL_PEDESTRIANS:
        raise SemanticPlannerError(f"total pedestrians must stay at or below {MAX_TOTAL_PEDESTRIANS}.")
    if len(plan.support_actors) + 1 > MAX_SPAWNER_COUNT:
        raise SemanticPlannerError(f"spawner count must stay at or below {MAX_SPAWNER_COUNT}.")
    if group_count > 5:
        raise SemanticPlannerError("group count exceeds limit.")


def _preferred_ego_behaviors(route_type: str) -> List[str]:
    if route_type in {"crosswalk_cross", "unmarked_junction_cross"}:
        return ["normal_cross", "cautious_cross", "assertive_cross"]
    return ["steady_walk", "slow_walk", "fast_pass"]


def _weighted_choice(rng: random.Random, items: List[str], preferred: Optional[List[str]] = None) -> str:
    preferred = preferred or []
    population = []
    for item in items:
        weight = 3 if item in preferred else 1
        population.extend([item] * weight)
    return rng.choice(population)


def _scenario_name(prefix: str, skeleton: CaseSkeleton, support_actors: List[ActorPlan]) -> str:
    total_peds = 1 + sum(actor.quantity for actor in support_actors)
    return "_".join(
        [
            prefix,
            skeleton.map_name.lower(),
            skeleton.scene_id.lower(),
            skeleton.main_route.route_type.lower(),
            f"p{total_peds}",
        ]
    )


def _template_description(
    *,
    scene_kind: str,
    main_route_id: str,
    main_route_type: str,
    ego_behavior: str,
    ego_spawn_time: float,
    support_actors: List[ActorPlan],
    vehicle_density: str,
) -> str:
    scene_sentence = f"This case takes place in {_scene_phrase(scene_kind)}."
    ego_sentence = (
        f"The ego pedestrian enters at {_format_seconds(ego_spawn_time)} and follows route {main_route_id}, "
        f"{_route_phrase(main_route_type)}, with {_behavior_phrase(ego_behavior)}."
    )

    if support_actors:
        support_count = sum(actor.quantity for actor in support_actors)
        route_count = len({actor.route_id for actor in support_actors})
        support_intro = (
            f"Support traffic adds {support_count} additional {_pluralize('pedestrian', support_count)} "
            f"across {route_count} support {_pluralize('route', route_count)}."
        )
        support_details = " ".join(
            _support_actor_sentence(actor)
            for actor in sorted(support_actors, key=lambda actor: (actor.spawn_time, actor.route_id))
        )
    else:
        support_intro = "No additional pedestrians are configured beyond the ego pedestrian."
        support_details = ""

    group_count = sum(1 for actor in support_actors if actor.quantity > 1)
    vehicle_sentence = (
        f"This layout includes {group_count} pedestrian {_pluralize('group', group_count)}, "
        f"and {_vehicle_phrase(vehicle_density)}."
    )
    return " ".join(
        sentence
        for sentence in (scene_sentence, ego_sentence, support_intro, support_details, vehicle_sentence)
        if sentence
    )


def _scene_phrase(scene_kind: str) -> str:
    if scene_kind == "sidewalk":
        return "a sidewalk scene"
    if scene_kind == "crossing":
        return "a junction scene"
    return f"a {scene_kind.replace('_', ' ')} scene"


def _route_phrase(route_type: str) -> str:
    return {
        "sidewalk_through": "a continuous sidewalk through-route",
        "junction_through": "a straight-through junction route",
        "junction_turn": "a turning junction route",
        "junction_u_turn": "a U-turn route inside the junction",
        "crosswalk_cross": "a marked crosswalk crossing",
        "unmarked_junction_cross": "an unmarked junction crossing",
    }.get(route_type, f"a {route_type.replace('_', ' ')} route")


def _behavior_phrase(behavior: str) -> str:
    return {
        "steady_walk": "a steady walking pace",
        "slow_walk": "a slow walking pace",
        "fast_pass": "a brisk passing pace",
        "normal_cross": "a normal crossing pace",
        "cautious_cross": "a cautious crossing pace",
        "assertive_cross": "an assertive crossing pace",
    }.get(behavior, f"a {behavior.replace('_', ' ')} profile")


def _vehicle_phrase(vehicle_density: str) -> str:
    return {
        "low": "background vehicle traffic is light",
        "medium": "background vehicle traffic is moderate",
        "high": "background vehicle traffic is heavy",
    }.get(vehicle_density, f"background vehicle traffic is {vehicle_density.replace('_', ' ')}")


def _support_actor_sentence(actor: ActorPlan) -> str:
    route_fragment = f"route {actor.route_id} ({_route_phrase(actor.route_type)})"
    if actor.quantity > 1:
        spacing = actor.spawn_interval if actor.spawn_interval is not None else 0.15
        formation = actor.formation or "group"
        return (
            f"A {_group_label(actor.quantity, formation)} uses {route_fragment} with "
            f"{_behavior_phrase(actor.behavior)}, starting at {_format_seconds(actor.spawn_time)} with "
            f"{spacing:.2f} s spacing between members."
        )
    return (
        f"One single pedestrian uses {route_fragment} with {_behavior_phrase(actor.behavior)} "
        f"and enters at {_format_seconds(actor.spawn_time)}."
    )


def _group_label(quantity: int, formation: str) -> str:
    if formation == "abreast":
        return f"{quantity}-person abreast group"
    if formation == "follow":
        return f"{quantity}-person single-file group"
    return f"{quantity}-person pedestrian group"


def _format_seconds(value: float) -> str:
    return f"{float(value):.2f} s"


def _pluralize(noun: str, count: int) -> str:
    return noun if count == 1 else noun + "s"


def _call_openai_compatible(
    *,
    api_key: str,
    base_url: str,
    payload: Dict[str, object],
    timeout_s: float,
) -> Dict[str, object]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SemanticPlannerError(f"Planner HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SemanticPlannerError(f"Planner request failed: {exc}") from exc


def _format_error(exc: Exception) -> str:
    message = f"{exc.__class__.__name__}: {exc}"
    if len(message) > 500:
        return message[:497] + "..."
    return message


def _extract_content(response: Dict[str, object]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise SemanticPlannerError("Planner response missing choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise SemanticPlannerError("Planner response missing content.")
    return str(content)


def _extract_json_object(content: str) -> Dict[str, object]:
    match = JSON_BLOCK_RE.search(content)
    if not match:
        raise SemanticPlannerError("Planner did not return a JSON object.")
    return json.loads(match.group(0))


def _normalize_optional(value) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() == "null":
        return None
    return value


def _coerce_support_actor(raw_actor: Dict[str, object]) -> ActorPlan:
    quantity = int(raw_actor.get("quantity", 1))
    formation = _normalize_optional(raw_actor.get("formation"))
    if quantity <= 1:
        formation = None
    return ActorPlan(
        role=str(raw_actor.get("role", "support")),
        route_id=str(raw_actor["route_id"]),
        route_type=str(raw_actor["route_type"]),
        behavior=str(raw_actor["behavior"]),
        quantity=quantity,
        formation=formation,
        spawn_time=float(raw_actor.get("spawn_time", 0.0)),
        spawn_interval=(
            float(raw_actor["spawn_interval"])
            if raw_actor.get("spawn_interval") is not None
            else (0.15 if quantity > 1 else None)
        ),
    )


def _build_support_actors(
    *,
    skeleton: CaseSkeleton,
    inventory: CatalogInventory,
    rng: random.Random,
) -> List[ActorPlan]:
    if skeleton.scene_kind == "sidewalk":
        return _build_sidewalk_support_actors(skeleton=skeleton, rng=rng)

    available_routes = list(skeleton.support_routes)
    fallback_same_route = len(available_routes) == 0
    if fallback_same_route:
        available_routes = [skeleton.main_route]

    target_total_peds = rng.randint(6, MAX_TOTAL_PEDESTRIANS) if skeleton.scene_kind == "crossing" else rng.randint(3, 10)
    remaining_peds = max(target_total_peds - 1, 1)
    max_groups = min(5, len(available_routes))
    support_actors: List[ActorPlan] = []
    used_routes = set()
    used_spawn_keys = set()
    if not fallback_same_route:
        used_routes.add(skeleton.main_route.route_id)
        ego_spawn_key = _route_spawn_key(skeleton.main_route)
        if ego_spawn_key is not None:
            used_spawn_keys.add(ego_spawn_key)
    group_count = 0
    spawn_time = 0.0

    while remaining_peds > 0 and len(support_actors) < MAX_SPAWNER_COUNT - 1:
        candidate_routes = [route for route in available_routes if route.route_id not in used_routes]
        if fallback_same_route:
            candidate_routes = available_routes
        non_overlapping_routes = [
            route for route in candidate_routes if _route_spawn_key(route) not in used_spawn_keys
        ]
        if non_overlapping_routes:
            candidate_routes = non_overlapping_routes
        if not candidate_routes:
            candidate_routes = list(available_routes)
        route = rng.choice(candidate_routes)
        behavior = _weighted_choice(rng, route.supported_behaviors, preferred=_preferred_ego_behaviors(route.route_type))

        if (
            remaining_peds >= 2
            and group_count < max_groups
            and rng.random() < 0.35
        ):
            quantity = 3 if remaining_peds >= 3 and rng.random() < 0.35 else 2
            quantity = min(quantity, remaining_peds)
            formation = rng.choice(list(SUPPORTED_GROUP_FORMATIONS))
            spawn_interval = 0.15
            group_count += 1
        else:
            quantity = 1
            formation = None
            spawn_interval = None

        support_actors.append(
            ActorPlan(
                role="support_group" if quantity > 1 else "support_single",
                route_id=route.route_id,
                route_type=route.route_type,
                behavior=behavior,
                quantity=quantity,
                formation=formation,
                spawn_time=round(min(spawn_time, 8.0), 2),
                spawn_interval=spawn_interval,
            )
        )
        if not fallback_same_route:
            used_routes.add(route.route_id)
        spawn_key = _route_spawn_key(route)
        if spawn_key is not None:
            used_spawn_keys.add(spawn_key)
        remaining_peds -= quantity
        spawn_time += rng.uniform(0.5, 2.0)

    return support_actors


def _build_sidewalk_support_actors(
    *,
    skeleton: CaseSkeleton,
    rng: random.Random,
) -> List[ActorPlan]:
    available_routes = list(skeleton.support_routes) or [skeleton.main_route]
    reverse_routes = [route for route in available_routes if _is_reverse_route(route.route_id)]
    forward_routes = [route for route in available_routes if not _is_reverse_route(route.route_id)]

    target_total_peds = rng.randint(3, 7)
    remaining_peds = max(target_total_peds - 1, 1)
    support_actors: List[ActorPlan] = []
    spawn_time = rng.uniform(0.0, 0.8)
    used_opposite_direction = False

    while remaining_peds > 0 and len(support_actors) < 9:
        if spawn_time > 8.0:
            break
        route_pool = available_routes
        if not used_opposite_direction:
            if _is_reverse_route(skeleton.main_route.route_id) and forward_routes:
                route_pool = forward_routes
            elif not _is_reverse_route(skeleton.main_route.route_id) and reverse_routes:
                route_pool = reverse_routes

        route = rng.choice(route_pool)
        used_opposite_direction = used_opposite_direction or (
            _is_reverse_route(route.route_id) != _is_reverse_route(skeleton.main_route.route_id)
        )
        behavior = _weighted_choice(
            rng,
            route.supported_behaviors,
            preferred=["steady_walk", "slow_walk", "fast_pass"],
        )

        allow_group = (
            remaining_peds >= 2
            and len(support_actors) <= 2
            and rng.random() < 0.18
        )
        if allow_group:
            quantity = min(2, remaining_peds)
            formation = rng.choices(
                population=["follow", "abreast"],
                weights=[0.70, 0.30],
                k=1,
            )[0]
            spawn_interval = round(rng.uniform(0.45, MAX_SUPPORT_GROUP_INTERVAL_BEFORE_EGO_S), 2)
        else:
            quantity = 1
            formation = None
            spawn_interval = None

        support_actors.append(
            ActorPlan(
                role="support_group" if quantity > 1 else "support_single",
                route_id=route.route_id,
                route_type=route.route_type,
                behavior=behavior,
                quantity=quantity,
                formation=formation,
                spawn_time=round(spawn_time, 2),
                spawn_interval=spawn_interval,
            )
        )
        remaining_peds -= quantity
        if remaining_peds > 0:
            max_gap = max((8.0 - spawn_time) / remaining_peds, 0.35)
            gap_low = max(0.35, min(1.2, max_gap * 0.55))
            gap_high = max(gap_low, min(2.0, max_gap * 0.95))
            spawn_time += rng.uniform(gap_low, gap_high)

    return support_actors


def _shape_sidewalk_support_actors(
    *,
    support_actors: List[ActorPlan],
    skeleton: CaseSkeleton,
    rng: random.Random,
) -> List[ActorPlan]:
    if not support_actors:
        return _build_sidewalk_support_actors(skeleton=skeleton, rng=rng)

    shaped: List[ActorPlan] = []
    total_support_peds = 0
    used_opposite_direction = False
    main_is_reverse = _is_reverse_route(skeleton.main_route.route_id)
    available_routes = list(skeleton.support_routes) or [skeleton.main_route]
    reverse_routes = [route for route in available_routes if _is_reverse_route(route.route_id)]
    forward_routes = [route for route in available_routes if not _is_reverse_route(route.route_id)]

    for index, actor in enumerate(support_actors[:5]):
        if total_support_peds >= 5:
            break
        quantity = max(1, min(int(actor.quantity), 2, 5 - total_support_peds))
        formation = actor.formation if quantity > 1 else None
        spawn_interval = actor.spawn_interval
        if quantity > 1:
            if formation not in SUPPORTED_GROUP_FORMATIONS:
                formation = "follow"
            spawn_interval = round(
                min(max(float(spawn_interval or 0.6), 0.45), MAX_SUPPORT_GROUP_INTERVAL_BEFORE_EGO_S),
                2,
            )

        spawn_time = min(max(float(actor.spawn_time), 0.0), 7.2)
        if shaped and spawn_time < shaped[-1].spawn_time + 1.5:
            spawn_time = shaped[-1].spawn_time + rng.uniform(1.8, 3.2)
        if spawn_time > 7.2:
            break

        route_id = actor.route_id
        if index == 0:
            if main_is_reverse and forward_routes:
                route_id = rng.choice(forward_routes).route_id
            elif not main_is_reverse and reverse_routes:
                route_id = rng.choice(reverse_routes).route_id
        used_opposite_direction = used_opposite_direction or (_is_reverse_route(route_id) != main_is_reverse)

        shaped.append(
            ActorPlan(
                role="support_group" if quantity > 1 else "support_single",
                route_id=route_id,
                route_type=actor.route_type,
                behavior=actor.behavior,
                quantity=quantity,
                formation=formation,
                spawn_time=round(spawn_time, 2),
                spawn_interval=spawn_interval,
            )
        )
        total_support_peds += quantity

    if not used_opposite_direction and shaped:
        if main_is_reverse and forward_routes:
            shaped[0].route_id = rng.choice(forward_routes).route_id
        elif not main_is_reverse and reverse_routes:
            shaped[0].route_id = rng.choice(reverse_routes).route_id

    return shaped


def _is_reverse_route(route_id: str) -> bool:
    return str(route_id).endswith("__reverse")


def _ego_spawn_time(support_actors: List[ActorPlan], rng: random.Random) -> float:
    if not support_actors:
        return 0.0
    latest_support_spawn = max(actor.spawn_time for actor in support_actors)
    delay = rng.uniform(*EGO_DELAY_AFTER_LATEST_SUPPORT_RANGE_S)
    return float(min(latest_support_spawn + delay, EGO_SPAWN_TIME_MAX_S))


def _route_spawn_key(route) -> Optional[tuple[float, float]]:
    spawn_point_xy = getattr(route, "spawn_point_xy", None)
    if not spawn_point_xy or len(spawn_point_xy) < 2:
        return None
    return (round(float(spawn_point_xy[0]), 2), round(float(spawn_point_xy[1]), 2))
