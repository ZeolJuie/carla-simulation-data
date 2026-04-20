import argparse
import json
import math
import random
from pathlib import Path

import carla

from agents.navigation.global_route_planner import GlobalRoutePlanner


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate deterministic closed-loop vehicle route candidates from the CARLA driving graph.'
    )
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--map-name', default='Town10HD_Opt')
    parser.add_argument('--sampling-resolution', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=2111)
    parser.add_argument('--max-spawn-points', type=int, default=120)
    parser.add_argument('--max-attempts', type=int, default=4000)
    parser.add_argument('--target-loops', type=int, default=160)
    parser.add_argument('--min-length', type=float, default=800.0)
    parser.add_argument('--min-waypoints', type=int, default=60)
    parser.add_argument('--min-unique-road-keys', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def connect_world(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    current_map = world.get_map()
    if current_map.name.split('/')[-1] != args.map_name:
        world = client.load_world(args.map_name)
        current_map = world.get_map()
    return client, world, current_map


def waypoint_key(waypoint):
    return (
        int(waypoint.road_id),
        int(waypoint.section_id),
        int(waypoint.lane_id),
    )


def plan_segment(grp, start_location, end_location):
    route = grp.trace_route(start_location, end_location)
    if not route:
        return []
    return route


def dedupe_plan_points(plan):
    points = []
    keys = []
    for waypoint, road_option in plan:
        location = waypoint.transform.location
        point = [round(float(location.x), 3), round(float(location.y), 3), round(float(location.z), 3)]
        key = waypoint_key(waypoint)
        if points and math.dist(points[-1][:2], point[:2]) < 0.5:
            continue
        points.append(point)
        keys.append(key)
    return points, keys


def concat_segments(*segments):
    combined = []
    for segment in segments:
        if not segment:
            return []
        if not combined:
            combined.extend(segment)
            continue
        tail_waypoint = combined[-1][0]
        head_waypoint = segment[0][0]
        if tail_waypoint.id == head_waypoint.id:
            combined.extend(segment[1:])
        else:
            combined.extend(segment)
    return combined


def route_length(points):
    total = 0.0
    for p0, p1 in zip(points, points[1:]):
        total += math.dist(p0[:2], p1[:2])
    return total


def sample_signature(points, road_keys):
    stride = max(len(points) // 24, 1)
    sampled = []
    for index in range(0, len(points), stride):
        x, y, _ = points[index]
        sampled.append((round(x, 1), round(y, 1), *road_keys[index]))
    if sampled:
        sampled.append(sampled[0])
    return tuple(sampled)


def build_loop_record(loop_index, spawn_points, triplet, combined_plan):
    points, road_keys = dedupe_plan_points(combined_plan)
    unique_road_keys = sorted({tuple(key) for key in road_keys})
    return {
        'route_id': f'veh_loop_{loop_index:03d}',
        'anchor_spawn_indices': [int(value) for value in triplet],
        'anchor_locations': [
            [
                round(float(spawn_points[value].location.x), 3),
                round(float(spawn_points[value].location.y), 3),
                round(float(spawn_points[value].location.z), 3),
            ]
            for value in triplet
        ],
        'loop_length_m': round(route_length(points), 3),
        'waypoint_count': len(points),
        'unique_road_key_count': len(unique_road_keys),
        'polyline': points,
        'road_keys': [list(key) for key in unique_road_keys],
    }


def main():
    args = parse_args()
    _, world, carla_map = connect_world(args)
    grp = GlobalRoutePlanner(carla_map, args.sampling_resolution)

    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError('No vehicle spawn points found on map.')

    if args.max_spawn_points > 0:
        spawn_points = spawn_points[:args.max_spawn_points]

    rng = random.Random(args.seed)
    loops = []
    seen_triplets = set()
    seen_signatures = set()
    attempts = 0

    while attempts < args.max_attempts and len(loops) < args.target_loops:
        attempts += 1
        triplet = tuple(rng.sample(range(len(spawn_points)), 3))
        if triplet in seen_triplets:
            continue
        seen_triplets.add(triplet)

        a, b, c = triplet
        location_a = spawn_points[a].location
        location_b = spawn_points[b].location
        location_c = spawn_points[c].location

        segment_ab = plan_segment(grp, location_a, location_b)
        segment_bc = plan_segment(grp, location_b, location_c)
        segment_ca = plan_segment(grp, location_c, location_a)
        combined_plan = concat_segments(segment_ab, segment_bc, segment_ca)
        if not combined_plan:
            continue

        points, road_keys = dedupe_plan_points(combined_plan)
        if len(points) < args.min_waypoints:
            continue

        loop_length_m = route_length(points)
        if loop_length_m < args.min_length:
            continue

        unique_road_keys = {tuple(key) for key in road_keys}
        if len(unique_road_keys) < args.min_unique_road_keys:
            continue

        signature = sample_signature(points, road_keys)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        loop_record = build_loop_record(len(loops) + 1, spawn_points, triplet, combined_plan)
        loops.append(loop_record)

    catalog = {
        'map_name': carla_map.name.split('/')[-1],
        'sampling_resolution': args.sampling_resolution,
        'seed': args.seed,
        'spawn_point_count': len(spawn_points),
        'attempt_count': attempts,
        'loop_count': len(loops),
        'filters': {
            'min_length_m': args.min_length,
            'min_waypoints': args.min_waypoints,
            'min_unique_road_keys': args.min_unique_road_keys,
        },
        'routes': loops,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, indent=2), encoding='utf-8')

    lengths = [route['loop_length_m'] for route in loops]
    if lengths:
        print(f'loops={len(loops)} attempts={attempts} min_length={min(lengths):.2f} max_length={max(lengths):.2f} avg_length={sum(lengths)/len(lengths):.2f}')
    else:
        print(f'loops=0 attempts={attempts}')
    print(f'catalog={args.output}')


if __name__ == '__main__':
    main()
