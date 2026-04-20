import copy
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


DEFAULT_SAMPLING_POLICY = {
    'sequence_id': 'auto',
    'output_root': 'artifacts/sequences',
    'frame_stride': 10,
    'valid_distance': 40.0,
    'sensor_sync_timeout': 1.0,
    'stop_on_ego_end': True,
    'export': {
        'ego_csv': True,
        'labels_json': True,
        'log_json': True,
        'calibration_json': True,
        'dynamic_actors': True,
        'city_objects': True,
        'simulation_records': True,
        'pedestrian_bev': True,
        'config_snapshots': True,
    },
    'annotation': {
        'actor_filters': ['*pedestrian*', '*vehicle*'],
        'city_object_labels': [
            'TrafficLight',
            'TrafficSigns',
            'Poles',
            'Static',
            'Fences',
            'Other',
            'Dynamic',
            'GuardRail',
            'Car',
            'Truck',
            'Bus',
            'Motorcycle',
            'Bicycle',
        ],
    },
}

SUPPORTED_SENSOR_TYPES = {
    'camera.rgb',
    'camera.depth',
    'camera.semantic_segmentation',
    'lidar.ray_cast',
    'lidar.ray_cast_semantic',
    'radar',
    'imu',
    'gnss',
}


class SamplingConfigError(ValueError):
    pass


def _deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_toml(path):
    with Path(path).open('rb') as handle:
        return tomllib.load(handle)


def _resolve_path(path_value, scenario_config_path):
    if path_value is None:
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path

    scenario_path = Path(scenario_config_path).resolve()
    scenario_relative = (scenario_path.parent / path).resolve()
    if scenario_relative.exists():
        return scenario_relative

    workspace_relative = (Path.cwd() / path).resolve()
    return workspace_relative


def _validate_sensor_rig(sensor_rig):
    sensors = sensor_rig.get('sensor', [])
    if not isinstance(sensors, list):
        raise SamplingConfigError('sensor rig must define [[sensor]] entries.')

    names = set()
    for index, sensor in enumerate(sensors):
        if not isinstance(sensor, dict):
            raise SamplingConfigError(f'sensor rig entry #{index} must be a table.')

        sensor_type = sensor.get('type')
        sensor_name = sensor.get('name')
        if sensor_type not in SUPPORTED_SENSOR_TYPES:
            raise SamplingConfigError(
                f'Unsupported sampling sensor type {sensor_type!r}. '
                f'Supported types: {sorted(SUPPORTED_SENSOR_TYPES)}.'
            )
        if not sensor_name:
            raise SamplingConfigError(f'sensor rig entry #{index} must define a non-empty name.')
        if sensor_name in names:
            raise SamplingConfigError(f'Duplicate sampling sensor name {sensor_name!r}.')
        names.add(sensor_name)


def load_sampling_setup(scenario_config, scenario_config_path):
    sampling_config = scenario_config.get('sampling')
    if not sampling_config or not sampling_config.get('enabled', False):
        return None

    if scenario_config_path is None:
        raise SamplingConfigError('Sampling requires a scenario config file path.')

    scenario_path = Path(scenario_config_path).resolve()
    sensor_rig_path = _resolve_path(sampling_config.get('sensor_rig'), scenario_path)
    policy_path = _resolve_path(sampling_config.get('policy'), scenario_path)

    sensor_rig = {}
    if sensor_rig_path is not None:
        sensor_rig = _load_toml(sensor_rig_path)
    _validate_sensor_rig(sensor_rig)

    resolved_policy = copy.deepcopy(DEFAULT_SAMPLING_POLICY)
    if policy_path is not None:
        _deep_update(resolved_policy, _load_toml(policy_path))

    scenario_policy_override = {
        key: value
        for key, value in sampling_config.items()
        if key not in {'enabled', 'sensor_rig', 'policy', 'ego'}
    }
    _deep_update(resolved_policy, scenario_policy_override)

    return {
        'enabled': True,
        'scenario_config_path': str(scenario_path),
        'sensor_rig_path': str(sensor_rig_path) if sensor_rig_path is not None else None,
        'policy_path': str(policy_path) if policy_path is not None else None,
        'sensor_rig': sensor_rig,
        'policy': resolved_policy,
        'sampling_config': sampling_config,
    }
