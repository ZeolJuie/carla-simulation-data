#!/usr/bin/env python3
"""Statistics and visualization entry point for TAPSN dataset cases.

The script is intentionally parameter-free: dataset roots, maps, and counting
rules are fixed in code so paper statistics remain reproducible.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from loguru import logger
except ModuleNotFoundError:  # Keep the script runnable in minimal envs.
    import logging

    logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    class _FallbackLogger:
        def __init__(self) -> None:
            self._logger = logging.getLogger("tapsn_dataset_statistics")

        def info(self, message: str, *args: Any) -> None:
            self._logger.info(message.format(*args))

        def warning(self, message: str, *args: Any) -> None:
            self._logger.warning(message.format(*args))

    logger = _FallbackLogger()

DATASET_ROOT = Path("/media/data2/zmh_data/TAPSN")
MAP_NAMES = [
    "Town01_Opt",
    "Town02_Opt",
    "Town03_Opt",
    "Town04_Opt",
    "Town05_Opt",
    "Town10HD_Opt",
    "Town15",
]
REFERENCE_SENSOR_SUBDIR = Path("sampling/image/CAM_FRONT")
OUTPUT_DIR = Path("artifacts/analysis/tapsn_all_maps")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tapsn_matplotlib")
WEATHER_PARAMETER_SPECS = [
    ("sun_altitude_angle", "Sun Altitude (deg)"),
    ("cloudiness", "Cloudiness"),
    ("precipitation", "Precipitation"),
    ("wetness", "Wetness"),
    ("fog_density", "Fog Density"),
    ("wind_intensity", "Wind Intensity"),
]
FLOW_OUTPUT_PREFIX = "background_flow_density_speed"
EGO_MOTION_OUTPUT_PREFIX = "ego_motion_statistics_panel"
SENSOR_JITTER_OUTPUT_PREFIX = "sensor_pose_motion_jitter"
JITTER_SENSOR_NAME = "CAM_FRONT"
JITTER_SAMPLE_RATE_HZ = 2.0
OBSTACLE_POLAR_OUTPUT_PREFIX = "obstacles_polar_distribution"
SPATIAL_GROUP_CLASS_MAP = {
    "pedestrians": {"Pedestrian"},
    "vehicles": {"Car", "Truck", "Bus", "Van"},
    "two-Wheeled_vehicles": {"Bicycle", "Motorcycle"},
    "obstacles": {"Static", "Poles", "Other", "Dynamic"},
    "fences": {"Fences", "GuardRail"},
}
OBJECT_GROUP_OUTPUT_PREFIX = "stats_sqrt_robust"
OBJECT_GROUP_CLASS_GROUP = {
    "Vehicles": ["Car", "Truck", "Bus", "Van"],
    "Cyclists": ["Bicycle", "Motorcycle"],
    "Pedestrians": ["Pedestrian"],
    "Traffic Fac.": ["TrafficLight", "TrafficSigns"],
    "Obstacles": ["Static", "Other", "Poles", "Dynamic"],
    "Barriers": ["Fences", "GuardRail"],
}
OBJECT_GROUP_SUB_COLORS = {
    "Vehicles": {"Car": "#2E86AB", "Truck": "#1A5276", "Bus": "#4D908E", "Van": "#577590"},
    "Cyclists": {"Bicycle": "#F18F01", "Motorcycle": "#B76900"},
    "Pedestrians": {"Pedestrian": "#A23B72"},
    "Traffic Fac.": {"TrafficLight": "#C73E1D", "TrafficSigns": "#9E2A1B"},
    "Obstacles": {"Static": "#6A994E", "Other": "#4A7031", "Poles": "#8DB38B", "Dynamic": "#A7C957"},
    "Barriers": {"Fences": "#BB9457", "GuardRail": "#8B5A2B"},
}


@dataclass(frozen=True)
class CasePaths:
    case_dir: Path
    manifest: Path
    case_json: Path
    scenario_toml: Path
    sampling_log: Path
    calibration: Path
    labels: Path
    ego_csv: Path
    sensor_pose_csv: Path
    pedestrian_csv: Path
    vehicle_csv: Path
    obstacles_csv: Path
    run_summary: Path
    run_result: Path

    @property
    def reference_sensor_dir(self) -> Path:
        return self.case_dir / REFERENCE_SENSOR_SUBDIR

    @property
    def case_id(self) -> str:
        return self.case_dir.name


@dataclass
class CaseRecord:
    paths: CasePaths
    manifest: dict[str, Any]
    case_json: dict[str, Any]
    sampling_log: dict[str, Any]

    @property
    def case_id(self) -> str:
        return self.paths.case_id

    @property
    def map_name(self) -> str:
        return (
            self.case_json.get("skeleton", {}).get("map_name")
            or self.sampling_log.get("location")
            or self.paths.case_dir.parent.name
        )

    @property
    def is_valid_sampled(self) -> bool:
        if not self.paths.labels.exists() or not self.paths.ego_csv.exists():
            return False
        frame_num = safe_int(self.sampling_log.get("frame_num"))
        duration = safe_float(self.sampling_log.get("duration"))
        return (frame_num is not None and frame_num > 0) and (
            duration is not None and duration > 0
        )

    @property
    def sensor_count(self) -> int | None:
        sensor_count = safe_int(self.manifest.get("sensor_count"))
        if sensor_count is not None:
            return sensor_count
        calibration = read_json(self.paths.calibration, default={})
        sensors = calibration.get("sensors")
        return len(sensors) if isinstance(sensors, list) else None

    @property
    def frame_num(self) -> int | None:
        return safe_int(self.sampling_log.get("frame_num"))

    @property
    def duration_sec(self) -> float | None:
        return safe_float(self.sampling_log.get("duration"))


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default


def iter_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def make_case_paths(case_dir: Path) -> CasePaths:
    return CasePaths(
        case_dir=case_dir,
        manifest=case_dir / "manifest.json",
        case_json=case_dir / "config" / "case.json",
        scenario_toml=case_dir / "config" / "scenario.toml",
        sampling_log=case_dir / "sampling" / "log.json",
        calibration=case_dir / "sampling" / "calibration.json",
        labels=case_dir / "sampling" / "labels.json",
        ego_csv=case_dir / "sampling" / "ego.csv",
        sensor_pose_csv=case_dir / "sampling" / "sensor_pose.csv",
        pedestrian_csv=case_dir / "simulation" / "pedestrian.csv",
        vehicle_csv=case_dir / "simulation" / "vehicle.csv",
        obstacles_csv=case_dir / "simulation" / "obstacles.csv",
        run_summary=case_dir / "simulation" / "run_summary.json",
        run_result=case_dir / "logs" / "run_result.json",
    )


def discover_cases(map_root: Path) -> list[CaseRecord]:
    if not map_root.exists():
        raise FileNotFoundError(f"Map root does not exist: {map_root}")

    records: list[CaseRecord] = []
    for case_dir in sorted(p for p in map_root.iterdir() if p.is_dir()):
        paths = make_case_paths(case_dir)
        if not (
            paths.case_json.exists()
            or paths.manifest.exists()
            or paths.sampling_log.exists()
        ):
            continue
        records.append(
            CaseRecord(
                paths=paths,
                manifest=read_json(paths.manifest, default={}),
                case_json=read_json(paths.case_json, default={}),
                sampling_log=read_json(paths.sampling_log, default={}),
            )
        )
    return records


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            filepath = Path(dirpath) / filename
            try:
                total += filepath.stat().st_size
            except OSError:
                continue
    return total


def count_label_annotations(labels_path: Path) -> tuple[int, Counter[str]]:
    labels = read_json(labels_path, default=[])
    if not isinstance(labels, list):
        return 0, Counter()
    total = 0
    class_counts: Counter[str] = Counter()
    for frame in labels:
        objects = frame.get("objects", []) if isinstance(frame, dict) else []
        if not isinstance(objects, list):
            continue
        total += len(objects)
        for obj in objects:
            if isinstance(obj, dict):
                class_counts[str(obj.get("class", "Unknown"))] += 1
    return total, class_counts


def summarize_counter(counter: Counter[str], top_n: int | None = None) -> str:
    items = counter.most_common(top_n)
    if not items:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in items)


def count_sensor_frames(sensor_dir: Path) -> int:
    if not sensor_dir.exists():
        return 0
    return sum(1 for p in sensor_dir.iterdir() if p.is_file())


def parse_ego_ped_id(ego_selection_path: Path, manifest_path: Path) -> int | None:
    for path in (ego_selection_path, manifest_path):
        payload = read_json(path, default={})
        resolved = payload.get("resolved_ego") if isinstance(payload, dict) else None
        if not isinstance(resolved, dict):
            continue
        ped_name = resolved.get("ped_name")
        if isinstance(ped_name, str) and ped_name.startswith("ped_"):
            suffix = ped_name.split("_", 1)[1]
            try:
                return int(suffix)
            except ValueError:
                continue
    return None


def load_all_cases() -> list[CaseRecord]:
    all_records: list[CaseRecord] = []
    for map_name in MAP_NAMES:
        map_root = DATASET_ROOT / map_name
        logger.info(f"Scanning map root: {map_root}")
        records = discover_cases(map_root)
        logger.info(f"Discovered {len(records)} cases for {map_name}")
        all_records.extend(records)
    logger.info(f"Discovered {len(all_records)} cases in total")
    return all_records


def find_sensor_count(records: list[CaseRecord]) -> tuple[int | None, list[str]]:
    for record in records:
        calibration = read_json(record.paths.calibration, default={})
        sensors = calibration.get("sensors")
        if isinstance(sensors, list) and sensors:
            names = [str(s.get("name", "Unknown")) for s in sensors if isinstance(s, dict)]
            return len(sensors), names
    return None, []


def build_dataset_scale_overview(records: list[CaseRecord]) -> dict[str, Any]:
    logger.info("Building Dataset Scale Overview")
    maps = Counter(record.map_name for record in records)
    case_count = len(records)

    total_frames = 0
    missing_reference_sensor = []
    for record in records:
        frame_count = count_sensor_frames(record.paths.reference_sensor_dir)
        if frame_count == 0:
            missing_reference_sensor.append(record.case_id)
        total_frames += frame_count

    if missing_reference_sensor:
        logger.warning(
            "Reference sensor directory missing or empty for {} cases: {}",
            len(missing_reference_sensor),
            ", ".join(missing_reference_sensor[:10])
            + ("..." if len(missing_reference_sensor) > 10 else ""),
        )

    total_duration = 0.0
    missing_duration = []
    for record in records:
        duration = record.duration_sec
        if duration is None or not math.isfinite(duration):
            missing_duration.append(record.case_id)
            continue
        total_duration += duration

    if missing_duration:
        logger.warning(
            "Missing sampling duration for {} cases: {}",
            len(missing_duration),
            ", ".join(missing_duration[:10])
            + ("..." if len(missing_duration) > 10 else ""),
        )

    sensor_count, sensor_names = find_sensor_count(records)
    if sensor_count is None:
        logger.warning("No usable calibration.json found for sensor count")

    total_annotations = 0
    object_counts: Counter[str] = Counter()
    for record in records:
        annotation_count, class_counts = count_label_annotations(record.paths.labels)
        total_annotations += annotation_count
        object_counts.update(class_counts)

    logger.info("Computing storage size")
    storage_bytes = sum(directory_size_bytes(record.paths.case_dir) for record in records)

    return {
        "maps": maps,
        "case_count": case_count,
        "reference_sensor": str(REFERENCE_SENSOR_SUBDIR),
        "total_frames": total_frames,
        "total_duration_sec": total_duration,
        "sensor_count": sensor_count,
        "sensor_names": sensor_names,
        "total_annotations": total_annotations,
        "object_counts": object_counts,
        "storage_bytes": storage_bytes,
        "missing_reference_sensor_cases": missing_reference_sensor,
        "missing_duration_cases": missing_duration,
    }


def format_duration(seconds: float) -> str:
    minutes = seconds / 60.0
    hours = seconds / 3600.0
    return f"{seconds:,.2f} s ({minutes:,.2f} min / {hours:,.2f} h)"


def format_int(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}"


def print_table(title: str, rows: list[tuple[str, str]]) -> None:
    label_width = max(len(label) for label, _ in rows)
    value_width = max(len(value) for _, value in rows)
    title_width = len(title) + 2
    inner_width = max(label_width + value_width + 5, title_width)

    print()
    print("+" + "-" * inner_width + "+")
    print("| " + title.ljust(inner_width - 2) + " |")
    print("+" + "-" * (label_width + 2) + "+" + "-" * (inner_width - label_width - 3) + "+")
    for label, value in rows:
        print(
            "| "
            + label.ljust(label_width)
            + " | "
            + value.ljust(inner_width - label_width - 5)
            + " |"
        )
    print("+" + "-" * (label_width + 2) + "+" + "-" * (inner_width - label_width - 3) + "+")


def print_dataset_scale_overview(summary: dict[str, Any]) -> None:
    maps = summary["maps"]
    rows = [
        ("Dataset root", str(DATASET_ROOT)),
        ("Maps", f"{len(maps)} ({', '.join(maps.keys())})"),
        ("Cases", format_int(summary["case_count"])),
        ("Reference sensor", summary["reference_sensor"]),
        ("Sensor frames", format_int(summary["total_frames"])),
        ("Total duration", format_duration(summary["total_duration_sec"])),
        ("Sensor count", format_int(summary["sensor_count"])),
        ("Annotations", format_int(summary["total_annotations"])),
        ("Storage size", human_bytes(summary["storage_bytes"])),
    ]
    print_table("Dataset Scale Overview", rows)

    sensor_names = summary["sensor_names"]
    if sensor_names:
        print()
        print("Sensors")
        print("  " + ", ".join(sensor_names))

    object_counts = summary["object_counts"]
    if object_counts:
        print()
        print("Top annotation classes")
        print("  " + summarize_counter(object_counts, top_n=12))

    missing_reference = summary["missing_reference_sensor_cases"]
    missing_duration = summary["missing_duration_cases"]
    if missing_reference or missing_duration:
        print()
        print("Notes")
        if missing_reference:
            print(
                "  "
                + f"{len(missing_reference)} cases have no files under "
                + str(REFERENCE_SENSOR_SUBDIR)
            )
        if missing_duration:
            print(f"  {len(missing_duration)} cases have no sampling duration")


def run_dataset_scale_overview() -> list[CaseRecord]:
    records = load_all_cases()
    summary = build_dataset_scale_overview(records)
    print_dataset_scale_overview(summary)
    return records


def collect_route_type_coverage(records: list[CaseRecord]) -> list[dict[str, Any]]:
    logger.info("Building Route Type Coverage")
    stats: dict[str, dict[str, Any]] = {}
    missing_route_type = []
    missing_route_length = []

    for record in records:
        skeleton = record.case_json.get("skeleton", {})
        main_route = skeleton.get("main_route") or {}
        route_type = main_route.get("route_type")
        route_length = safe_float(main_route.get("route_length_m"))

        if not route_type:
            missing_route_type.append(record.case_id)
            route_type = "Unknown"
        if route_length is None:
            missing_route_length.append(record.case_id)
            route_length = 0.0

        bucket = stats.setdefault(
            str(route_type),
            {
                "route_type": str(route_type),
                "count": 0,
                "total_length_m": 0.0,
                "lengths": [],
            },
        )
        bucket["count"] += 1
        bucket["total_length_m"] += route_length
        bucket["lengths"].append(route_length)

    if missing_route_type:
        logger.warning(
            "Missing main route type for {} cases: {}",
            len(missing_route_type),
            ", ".join(missing_route_type[:10])
            + ("..." if len(missing_route_type) > 10 else ""),
        )
    if missing_route_length:
        logger.warning(
            "Missing main route length for {} cases: {}",
            len(missing_route_length),
            ", ".join(missing_route_length[:10])
            + ("..." if len(missing_route_length) > 10 else ""),
        )

    rows = []
    total_cases = sum(item["count"] for item in stats.values())
    for item in stats.values():
        count = item["count"]
        lengths = item["lengths"]
        rows.append(
            {
                "route_type": item["route_type"],
                "count": count,
                "ratio": count / total_cases if total_cases else 0.0,
                "total_length_m": item["total_length_m"],
                "mean_length_m": item["total_length_m"] / count if count else 0.0,
                "min_length_m": min(lengths) if lengths else 0.0,
                "max_length_m": max(lengths) if lengths else 0.0,
            }
        )

    return sorted(rows, key=lambda row: (-row["count"], row["route_type"]))


def print_route_type_coverage(rows: list[dict[str, Any]]) -> None:
    table_rows = [
        (
            row["route_type"],
            (
                f"count={row['count']:,}, "
                f"ratio={row['ratio'] * 100:.1f}%, "
                f"total={row['total_length_m']:.1f} m, "
                f"mean={row['mean_length_m']:.1f} m"
            ),
        )
        for row in rows
    ]
    print_table("Route Type Coverage", table_rows)


def plot_route_type_coverage(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        logger.warning("No route type rows to plot")
        return

    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    route_types = [row["route_type"] for row in rows]
    counts = [row["count"] for row in rows]
    total_lengths = [row["total_length_m"] for row in rows]
    x = np.arange(len(route_types))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "figure.dpi": 300,
        }
    )

    fig, ax_count = plt.subplots(figsize=(8.4, 4.5), dpi=300)
    bar_color = "#4E79A7"
    length_color = "#E15759"
    width = 0.36

    count_bars = ax_count.bar(
        x - width / 2,
        counts,
        width=width,
        color=bar_color,
        alpha=0.88,
        label="Case Count",
        zorder=3,
    )
    ax_count.set_ylabel("Case Count", color=bar_color, fontweight="bold")
    ax_count.tick_params(axis="y", labelcolor=bar_color)
    ax_count.set_xticks(x)
    ax_count.set_xticklabels(route_types, rotation=25, ha="right")
    ax_count.grid(axis="y", linestyle="--", alpha=0.28, zorder=0)
    ax_count.set_ylim(0, max(counts) * 1.25 if counts else 1)

    for bar, row in zip(count_bars, rows):
        ax_count.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.025,
            f"{row['count']}\n{row['ratio'] * 100:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1F2933",
        )

    ax_length = ax_count.twinx()
    length_bars = ax_length.bar(
        x + width / 2,
        total_lengths,
        width=width,
        color=length_color,
        alpha=0.76,
        label="Total Route Length",
        zorder=4,
    )
    ax_length.set_ylabel("Total Route Length (m)", color=length_color, fontweight="bold")
    ax_length.tick_params(axis="y", labelcolor=length_color)
    ax_length.set_ylim(0, max(total_lengths) * 1.25 if total_lengths else 1)

    for bar, length in zip(length_bars, total_lengths):
        ax_length.text(
            bar.get_x() + bar.get_width() / 2,
            length + max(total_lengths) * 0.035,
            f"{length:.0f} m",
            ha="center",
            va="bottom",
            fontsize=8,
            color=length_color,
        )

    handles_1, labels_1 = ax_count.get_legend_handles_labels()
    handles_2, labels_2 = ax_length.get_legend_handles_labels()
    ax_count.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="upper right",
        frameon=False,
    )
    ax_count.set_title("Route Type Coverage", fontweight="bold", pad=10)
    fig.tight_layout()

    png_path = output_dir / "route_type_coverage_dual_axis.png"
    pdf_path = output_dir / "route_type_coverage_dual_axis.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved route type coverage figure: {png_path}")
    logger.info(f"Saved route type coverage figure: {pdf_path}")


def run_route_type_coverage(records: list[CaseRecord]) -> None:
    rows = collect_route_type_coverage(records)
    print_route_type_coverage(rows)
    plot_route_type_coverage(rows, OUTPUT_DIR)


def summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "min": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0}
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return {
        "n": n,
        "min": min(values),
        "mean": mean,
        "max": max(values),
        "std": math.sqrt(variance),
    }


def collect_weather_parameter_distributions(
    records: list[CaseRecord],
) -> tuple[list[dict[str, Any]], list[str]]:
    logger.info("Building Weather Diversity Statistics")
    missing_weather_cases = []
    series_map: dict[str, list[float]] = {key: [] for key, _ in WEATHER_PARAMETER_SPECS}

    for record in records:
        weather = record.sampling_log.get("weather", {})
        params = weather.get("parameters", {}) if isinstance(weather, dict) else {}
        if not isinstance(params, dict) or not params:
            missing_weather_cases.append(record.case_id)
            continue

        for key, _ in WEATHER_PARAMETER_SPECS:
            value = safe_float(params.get(key))
            if value is not None and math.isfinite(value):
                series_map[key].append(value)

    if missing_weather_cases:
        logger.warning(
            "Missing weather.parameters for {} cases: {}",
            len(missing_weather_cases),
            ", ".join(missing_weather_cases[:10])
            + ("..." if len(missing_weather_cases) > 10 else ""),
        )

    rows = []
    for key, label in WEATHER_PARAMETER_SPECS:
        values = series_map[key]
        summary = summarize_numeric(values)
        rows.append(
            {
                "key": key,
                "label": label,
                "values": values,
                **summary,
            }
        )
    return rows, missing_weather_cases


def print_weather_parameter_summary(rows: list[dict[str, Any]]) -> None:
    table_rows = [
        (
            row["label"],
            (
                f"n={int(row['n'])}, "
                f"min={row['min']:.2f}, "
                f"mean={row['mean']:.2f}, "
                f"max={row['max']:.2f}, "
                f"std={row['std']:.2f}"
            ),
        )
        for row in rows
    ]
    print_table("Weather Diversity Summary", table_rows)


def _gaussian_smooth_histogram(
    counts: Any,
    sigma_bins: float = 1.0,
) -> Any:
    import numpy as np

    if len(counts) == 0:
        return counts
    radius = max(1, int(math.ceil(3 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(counts, kernel, mode="same")
    if len(smoothed) == len(counts):
        return smoothed
    if len(smoothed) > len(counts):
        start = (len(smoothed) - len(counts)) // 2
        return smoothed[start : start + len(counts)]
    padded = np.zeros(len(counts), dtype=float)
    padded[: len(smoothed)] = smoothed
    return padded


def plot_weather_parameter_distributions(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2), dpi=300)
    axes = axes.flatten()
    hist_color = "#4E79A7"
    curve_color = "#D1495B"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
        }
    )

    for ax, row in zip(axes, rows):
        values = row["values"]
        label = row["label"]
        if not values:
            ax.set_title(label, fontweight="bold", pad=8)
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        values_np = np.asarray(values, dtype=float)
        bin_count = min(12, max(6, int(round(math.sqrt(len(values_np))))))
        counts, edges, _ = ax.hist(
            values_np,
            bins=bin_count,
            color=hist_color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        centers = (edges[:-1] + edges[1:]) / 2.0
        smooth_counts = _gaussian_smooth_histogram(counts, sigma_bins=1.0)
        ax.plot(centers, smooth_counts, color=curve_color, linewidth=2.0, zorder=3)

        ax.set_title(label, fontweight="bold", pad=8)
        ax.grid(axis="y", linestyle="--", alpha=0.24, zorder=0)
        ax.set_ylabel("Case Count")
        stat_text = (
            f"n={int(row['n'])}\n"
            f"mean={row['mean']:.1f}\n"
            f"std={row['std']:.1f}"
        )
        ax.text(
            0.97,
            0.95,
            stat_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="#D9D9D9"),
        )

    fig.suptitle("Weather Parameter Distributions", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = output_dir / "weather_parameter_distributions.png"
    pdf_path = output_dir / "weather_parameter_distributions.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved weather diversity figure: {png_path}")
    logger.info(f"Saved weather diversity figure: {pdf_path}")


def run_weather_diversity_statistics(records: list[CaseRecord]) -> None:
    rows, _ = collect_weather_parameter_distributions(records)
    print_weather_parameter_summary(rows)
    plot_weather_parameter_distributions(rows, OUTPUT_DIR)


def build_frame_flow_rows(records: list[CaseRecord]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    logger.info("Building Background Flow Analysis")
    pedestrian_rows: list[dict[str, float]] = []
    vehicle_rows: list[dict[str, float]] = []
    missing_ped_csv = []
    missing_vehicle_csv = []
    missing_ego_mapping = []

    for record in records:
        if not record.paths.pedestrian_csv.exists():
            missing_ped_csv.append(record.case_id)
        else:
            ego_ped_id = parse_ego_ped_id(record.paths.run_result.parent / "ego_selection.json", record.paths.manifest)
            if ego_ped_id is None:
                missing_ego_mapping.append(record.case_id)
            frame_to_speeds: dict[int, list[float]] = {}
            for row in iter_csv_rows(record.paths.pedestrian_csv) or []:
                ped_id = safe_int(row.get("ped_id"))
                frame = safe_int(row.get("frame"))
                vx = safe_float(row.get("v_x"))
                vy = safe_float(row.get("v_y"))
                if frame is None or vx is None or vy is None:
                    continue
                if ego_ped_id is not None and ped_id == ego_ped_id:
                    continue
                speed = math.hypot(vx, vy)
                frame_to_speeds.setdefault(frame, []).append(speed)

            for frame, speeds in frame_to_speeds.items():
                if not speeds:
                    continue
                pedestrian_rows.append(
                    {
                        "case_id": float(record.case_id),
                        "frame": float(frame),
                        "density": float(len(speeds)),
                        "mean_speed": float(sum(speeds) / len(speeds)),
                    }
                )

        if not record.paths.vehicle_csv.exists():
            missing_vehicle_csv.append(record.case_id)
        else:
            frame_to_speeds = {}
            for row in iter_csv_rows(record.paths.vehicle_csv) or []:
                frame = safe_int(row.get("frame"))
                speed = safe_float(row.get("vel"))
                if frame is None or speed is None:
                    continue
                frame_to_speeds.setdefault(frame, []).append(speed)

            for frame, speeds in frame_to_speeds.items():
                if not speeds:
                    continue
                vehicle_rows.append(
                    {
                        "case_id": float(record.case_id),
                        "frame": float(frame),
                        "density": float(len(speeds)),
                        "mean_speed": float(sum(speeds) / len(speeds)),
                    }
                )

    if missing_ped_csv:
        logger.warning(
            "Missing pedestrian.csv for {} cases: {}",
            len(missing_ped_csv),
            ", ".join(missing_ped_csv[:10]) + ("..." if len(missing_ped_csv) > 10 else ""),
        )
    if missing_vehicle_csv:
        logger.warning(
            "Missing vehicle.csv for {} cases: {}",
            len(missing_vehicle_csv),
            ", ".join(missing_vehicle_csv[:10]) + ("..." if len(missing_vehicle_csv) > 10 else ""),
        )
    if missing_ego_mapping:
        logger.warning(
            "Failed to map ego ped_id for {} cases: {}",
            len(missing_ego_mapping),
            ", ".join(missing_ego_mapping[:10]) + ("..." if len(missing_ego_mapping) > 10 else ""),
        )

    return pedestrian_rows, vehicle_rows


def summarize_flow_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    import numpy as np

    densities = [row["density"] for row in rows]
    speeds = [row["mean_speed"] for row in rows]
    return {
        "frames": float(len(rows)),
        "density_mean": summarize_numeric(densities)["mean"] if densities else 0.0,
        "density_median": float(np.median(densities)) if densities else 0.0,
        "density_p95": float(np.percentile(densities, 95)) if densities else 0.0,
        "speed_mean": summarize_numeric(speeds)["mean"] if speeds else 0.0,
        "speed_std": summarize_numeric(speeds)["std"] if speeds else 0.0,
    }


def print_flow_summary(
    pedestrian_rows: list[dict[str, float]],
    vehicle_rows: list[dict[str, float]],
) -> None:
    ped = summarize_flow_rows(pedestrian_rows)
    veh = summarize_flow_rows(vehicle_rows)
    rows = [
        (
            "Pedestrian flow",
            (
                f"frames={int(ped['frames'])}, "
                f"density_mean={ped['density_mean']:.2f}, "
                f"density_median={ped['density_median']:.2f}, "
                f"density_p95={ped['density_p95']:.2f}, "
                f"speed_mean={ped['speed_mean']:.2f}, "
                f"speed_std={ped['speed_std']:.2f}"
            ),
        ),
        (
            "Vehicle flow",
            (
                f"frames={int(veh['frames'])}, "
                f"density_mean={veh['density_mean']:.2f}, "
                f"density_median={veh['density_median']:.2f}, "
                f"density_p95={veh['density_p95']:.2f}, "
                f"speed_mean={veh['speed_mean']:.2f}, "
                f"speed_std={veh['speed_std']:.2f}"
            ),
        ),
    ]
    print_table("Background Flow Summary", rows)


def plot_density_speed_relation(
    pedestrian_rows: list[dict[str, float]],
    vehicle_rows: list[dict[str, float]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), dpi=300)

    def _plot_one(ax: Any, rows: list[dict[str, float]], title: str) -> None:
        if not rows:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontweight="bold", pad=8)
            return

        density = np.asarray([row["density"] for row in rows], dtype=float)
        speed = np.asarray([row["mean_speed"] for row in rows], dtype=float)

        hb = ax.hexbin(
            density,
            speed,
            gridsize=20,
            mincnt=1,
            cmap="YlOrRd",
            linewidths=0.2,
            edgecolors="none",
        )
        unique_density = sorted(set(int(v) for v in density))
        mean_x = []
        mean_y = []
        p25 = []
        p75 = []
        for d in unique_density:
            mask = density == d
            speeds = speed[mask]
            if len(speeds) == 0:
                continue
            mean_x.append(d)
            mean_y.append(float(np.mean(speeds)))
            p25.append(float(np.percentile(speeds, 25)))
            p75.append(float(np.percentile(speeds, 75)))

        mean_x_np = np.asarray(mean_x, dtype=float)
        mean_y_np = np.asarray(mean_y, dtype=float)
        p25_np = np.asarray(p25, dtype=float)
        p75_np = np.asarray(p75, dtype=float)

        ax.plot(mean_x_np, mean_y_np, color="#2C3E50", linewidth=2.0, label="Mean speed")
        ax.fill_between(mean_x_np, p25_np, p75_np, color="#2C3E50", alpha=0.16, label="P25-P75")
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xlabel("Density (actors / frame)")
        ax.set_ylabel("Mean Speed (m/s)")
        ax.grid(alpha=0.18, linestyle="--")
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Frame Count")
        ax.legend(frameon=False, loc="upper right")

    _plot_one(axes[0], pedestrian_rows, "Pedestrian Density-Speed Relation")
    _plot_one(axes[1], vehicle_rows, "Vehicle Density-Speed Relation")

    fig.tight_layout()
    png_path = output_dir / f"{FLOW_OUTPUT_PREFIX}.png"
    pdf_path = output_dir / f"{FLOW_OUTPUT_PREFIX}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved background flow figure: {png_path}")
    logger.info(f"Saved background flow figure: {pdf_path}")


def run_background_flow_analysis(records: list[CaseRecord]) -> None:
    pedestrian_rows, vehicle_rows = build_frame_flow_rows(records)
    print_flow_summary(pedestrian_rows, vehicle_rows)
    plot_density_speed_relation(pedestrian_rows, vehicle_rows, OUTPUT_DIR)


def unwrap_degrees(angle_deg: float, prev_angle_deg: float) -> float:
    delta = angle_deg - prev_angle_deg
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return prev_angle_deg + delta


def collect_ego_motion_samples(records: list[CaseRecord]) -> dict[str, list[float]]:
    logger.info("Building Ego Motion Statistics")
    speed_samples: list[float] = []
    acceleration_samples: list[float] = []
    yaw_rate_samples: list[float] = []
    missing_ego_csv = []

    for record in records:
        if not record.paths.ego_csv.exists():
            missing_ego_csv.append(record.case_id)
            continue

        rows = list(iter_csv_rows(record.paths.ego_csv) or [])
        if not rows:
            continue

        prev_time = None
        prev_speed = None
        prev_yaw_unwrapped = None
        for row in rows:
            vx = safe_float(row.get("velocity_x"))
            vy = safe_float(row.get("velocity_y"))
            vz = safe_float(row.get("velocity_z"))
            timestamp = safe_float(row.get("timestamp"))
            yaw = safe_float(row.get("rotation_yaw"))
            if None in (vx, vy, vz, timestamp, yaw):
                continue

            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            speed_samples.append(speed)

            if prev_yaw_unwrapped is None:
                yaw_unwrapped = yaw
            else:
                yaw_unwrapped = unwrap_degrees(yaw, prev_yaw_unwrapped)

            if prev_time is not None and timestamp > prev_time:
                dt = timestamp - prev_time
                if prev_speed is not None:
                    acceleration_samples.append((speed - prev_speed) / dt)
                if prev_yaw_unwrapped is not None:
                    yaw_rate_samples.append((yaw_unwrapped - prev_yaw_unwrapped) / dt)

            prev_time = timestamp
            prev_speed = speed
            prev_yaw_unwrapped = yaw_unwrapped

    if missing_ego_csv:
        logger.warning(
            "Missing ego.csv for {} cases: {}",
            len(missing_ego_csv),
            ", ".join(missing_ego_csv[:10]) + ("..." if len(missing_ego_csv) > 10 else ""),
        )

    return {
        "speed": speed_samples,
        "acceleration": acceleration_samples,
        "yaw_rate": yaw_rate_samples,
    }


def summarize_motion_metric(values: list[float]) -> dict[str, float]:
    import numpy as np

    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "n": float(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def print_ego_motion_summary(samples: dict[str, list[float]]) -> None:
    rows = []
    for key, label in (
        ("speed", "Speed (m/s)"),
        ("acceleration", "Acceleration (m/s^2)"),
        ("yaw_rate", "Yaw Rate (deg/s)"),
    ):
        stats = summarize_motion_metric(samples[key])
        rows.append(
            (
                label,
                (
                    f"n={int(stats['n'])}, "
                    f"mean={stats['mean']:.2f}, "
                    f"median={stats['median']:.2f}, "
                    f"p95={stats['p95']:.2f}, "
                    f"max={stats['max']:.2f}, "
                    f"std={stats['std']:.2f}"
                ),
            )
        )
    print_table("Ego Motion Summary", rows)


def clip_for_plot(values: list[float], low_pct: float = 0.5, high_pct: float = 99.5) -> list[float]:
    import numpy as np

    if not values:
        return values
    arr = np.asarray(values, dtype=float)
    low = np.percentile(arr, low_pct)
    high = np.percentile(arr, high_pct)
    clipped = arr[(arr >= low) & (arr <= high)]
    return clipped.tolist()


def world_to_ego(object_world_loc: list[float], ego_world_loc: list[float], ego_yaw_deg: float) -> list[float]:
    dx = object_world_loc[0] - ego_world_loc[0]
    dy = object_world_loc[1] - ego_world_loc[1]
    dz = object_world_loc[2] - ego_world_loc[2]
    ego_yaw_rad = math.radians(ego_yaw_deg)
    cos_yaw = math.cos(-ego_yaw_rad)
    sin_yaw = math.sin(-ego_yaw_rad)
    x_ego = dx * cos_yaw - dy * sin_yaw
    y_ego = dx * sin_yaw + dy * cos_yaw
    z_ego = dz
    return [x_ego, y_ego, z_ego]


def calculate_relative_angle_and_distance(object_ego_loc: list[float]) -> tuple[float, float]:
    x, y, _ = object_ego_loc
    distance = math.hypot(x, y)
    relative_angle_deg = math.degrees(math.atan2(y, x))
    if relative_angle_deg < 0:
        relative_angle_deg += 360.0
    return relative_angle_deg, distance


def plot_ego_motion_statistics(samples: dict[str, list[float]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), dpi=300)
    specs = [
        ("speed", "Speed Distribution", "Speed (m/s)", "#4E79A7"),
        ("acceleration", "Acceleration Distribution", "Acceleration (m/s^2)", "#E15759"),
        ("yaw_rate", "Yaw Rate Distribution", "Yaw Rate (deg/s)", "#59A14F"),
    ]

    for ax, (key, title, xlabel, color) in zip(axes, specs):
        values = clip_for_plot(samples[key])
        stats = summarize_motion_metric(samples[key])
        if not values:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontweight="bold", pad=8)
            continue

        arr = np.asarray(values, dtype=float)
        bins = min(60, max(20, int(round(math.sqrt(len(arr))))))
        counts, edges, _ = ax.hist(
            arr,
            bins=bins,
            color=color,
            alpha=0.80,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        centers = (edges[:-1] + edges[1:]) / 2.0
        smooth = _gaussian_smooth_histogram(counts, sigma_bins=1.2)
        ax.plot(centers, smooth, color="#1F2933", linewidth=2.0, zorder=3)
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Sample Count")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        stat_text = (
            f"mean={stats['mean']:.2f}\n"
            f"std={stats['std']:.2f}\n"
            f"p95={stats['p95']:.2f}\n"
            f"max={stats['max']:.2f}"
        )
        ax.text(
            0.97,
            0.95,
            stat_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="#D9D9D9"),
        )

    fig.tight_layout()
    png_path = output_dir / f"{EGO_MOTION_OUTPUT_PREFIX}.png"
    pdf_path = output_dir / f"{EGO_MOTION_OUTPUT_PREFIX}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ego motion figure: {png_path}")
    logger.info(f"Saved ego motion figure: {pdf_path}")


def compute_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def load_sensor_jitter_series(record: CaseRecord, sensor_name: str) -> dict[str, list[float] | float]:
    frames: list[float] = []
    z_offsets = []
    pitch_offsets = []
    ego_speeds = []
    for row in iter_csv_rows(record.paths.sensor_pose_csv) or []:
        if row.get("sensor_name") != sensor_name:
            continue
        frame = safe_int(row.get("frame"))
        z_offset = safe_float(row.get("gait_z_offset"))
        pitch_offset = safe_float(row.get("gait_pitch_offset_deg"))
        ego_speed = safe_float(row.get("ego_speed"))
        if None in (frame, z_offset, pitch_offset, ego_speed):
            continue
        frames.append(float(frame))
        z_offsets.append(z_offset)
        pitch_offsets.append(pitch_offset)
        ego_speeds.append(ego_speed)
    sample_count = len(frames)
    sample_interval = 1.0 / JITTER_SAMPLE_RATE_HZ
    time_seconds = [index * sample_interval for index in range(sample_count)]
    return {
        "frames": frames,
        "time_seconds": time_seconds,
        "z_offset": z_offsets,
        "pitch_offset": pitch_offsets,
        "ego_speed": ego_speeds,
        "sample_count": float(sample_count),
        "duration_seconds": max(sample_count - 1, 0) * sample_interval,
        "speed_mean": float(sum(ego_speeds) / sample_count) if sample_count else 0.0,
        "speed_std": compute_std(ego_speeds),
        "z_std": compute_std(z_offsets),
        "pitch_std": compute_std(pitch_offsets),
    }


def score_jitter_series(series: dict[str, list[float] | float]) -> float:
    sample_count = int(series.get("sample_count", 0.0))
    if sample_count < 100:
        return -1.0

    speed_mean = float(series.get("speed_mean", 0.0))
    speed_std = float(series.get("speed_std", 0.0))
    z_std = float(series.get("z_std", 0.0))
    pitch_std = float(series.get("pitch_std", 0.0))

    length_score = min(sample_count, 180) / 180.0
    # Prefer a normal walking sequence, rather than a mostly standing case.
    speed_mean_score = max(0.0, 1.0 - min(abs(speed_mean - 1.1) / 0.5, 1.0))
    # Some variation in ego speed helps the gait signal read naturally.
    speed_std_score = min(speed_std / 0.35, 1.0)
    # Vertical bobbing is the main signal we want to show.
    z_score = min(z_std / 0.02, 1.0)
    # Pitch offset is optional; some cases appear to have it fixed at zero.
    pitch_score = min(pitch_std / 0.5, 1.0)

    return (
        0.30 * length_score
        + 0.25 * speed_mean_score
        + 0.20 * speed_std_score
        + 0.20 * z_score
        + 0.05 * pitch_score
    )


def choose_representative_jitter_case(
    records: list[CaseRecord], sensor_name: str
) -> tuple[CaseRecord | None, dict[str, list[float] | float] | None, float]:
    best_record = None
    best_series = None
    best_score = -1.0
    for record in records:
        if not record.paths.sensor_pose_csv.exists():
            continue
        series = load_sensor_jitter_series(record, sensor_name)
        if not series["frames"]:
            continue
        score = score_jitter_series(series)
        if score > best_score:
            best_record = record
            best_series = series
            best_score = score
    return best_record, best_series, best_score


def plot_sensor_pose_motion_jitter(records: list[CaseRecord], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    record, series, score = choose_representative_jitter_case(records, JITTER_SENSOR_NAME)
    if record is None or series is None:
        logger.warning("No usable sensor_pose.csv found for jitter plot")
        return

    logger.info(
        "Selected representative jitter case: {}/{} ({}) with {} samples, duration={:.1f}s, score={:.3f}",
        record.map_name,
        record.case_id,
        JITTER_SENSOR_NAME,
        int(series["sample_count"]),
        float(series["duration_seconds"]),
        score,
    )
    if not series["time_seconds"]:
        logger.warning("Selected jitter case has no usable rows after parsing")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 6.2), dpi=300, sharex=True)
    x = series["time_seconds"]

    axes[0].plot(x, series["z_offset"], color="#4E79A7", linewidth=1.7)
    axes[0].set_ylabel("gait_z_offset (m)")
    axes[0].set_title(
        f"Sensor Pose Motion Jitter: {record.map_name}/{record.case_id} [{JITTER_SENSOR_NAME}]",
        fontweight="bold",
        pad=8,
    )
    axes[0].grid(alpha=0.25, linestyle="--")

    axes[1].plot(x, series["pitch_offset"], color="#E15759", linewidth=1.7)
    axes[1].set_ylabel("gait_pitch_offset_deg")
    axes[1].grid(alpha=0.25, linestyle="--")

    axes[2].plot(x, series["ego_speed"], color="#59A14F", linewidth=1.7)
    axes[2].set_ylabel("ego_speed (m/s)")
    axes[2].set_xlabel("Time Since Start (s)")
    axes[2].grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    png_path = output_dir / f"{SENSOR_JITTER_OUTPUT_PREFIX}.png"
    pdf_path = output_dir / f"{SENSOR_JITTER_OUTPUT_PREFIX}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved sensor jitter figure: {png_path}")
    logger.info(f"Saved sensor jitter figure: {pdf_path}")


def collect_ego_centric_spatial_objects(
    records: list[CaseRecord],
) -> tuple[list[dict[str, Any]], Counter[str], list[str], list[str]]:
    logger.info("Building Ego-Centric Obstacle Spatial Distribution")
    rows: list[dict[str, Any]] = []
    grouped_counts: Counter[str] = Counter()
    missing_labels = []
    missing_ego = []

    for record in records:
        if not record.paths.labels.exists():
            missing_labels.append(f"{record.map_name}/{record.case_id}")
            continue
        if not record.paths.ego_csv.exists():
            missing_ego.append(f"{record.map_name}/{record.case_id}")
            continue

        ego_frame_map: dict[str, dict[str, Any]] = {}
        for ego_row in iter_csv_rows(record.paths.ego_csv) or []:
            frame = ego_row.get("frame")
            x = safe_float(ego_row.get("location_x"))
            y = safe_float(ego_row.get("location_y"))
            z = safe_float(ego_row.get("location_z"))
            yaw = safe_float(ego_row.get("rotation_yaw"))
            if frame is None or None in (x, y, z, yaw):
                continue
            ego_frame_map[frame] = {"location": [x, y, z], "yaw": yaw}

        labels = read_json(record.paths.labels, default=[])
        if not isinstance(labels, list):
            continue

        for frame in labels:
            if not isinstance(frame, dict):
                continue
            frame_id = str(frame.get("frame_id"))
            ego_state = ego_frame_map.get(frame_id)
            if ego_state is None:
                continue
            ego_world_loc = ego_state["location"]
            ego_yaw = ego_state["yaw"]
            objects = frame.get("objects", [])
            if not isinstance(objects, list):
                continue

            for group_name, class_names in SPATIAL_GROUP_CLASS_MAP.items():
                grouped = [
                    obj
                    for obj in objects
                    if isinstance(obj, dict)
                    and obj.get("class") in class_names
                    and safe_int(obj.get("occlusion")) == 0
                ]
                grouped_counts[group_name] += len(grouped)
                for obj in grouped:
                    obj_loc = obj.get("location")
                    if not (
                        isinstance(obj_loc, list)
                        and len(obj_loc) >= 3
                        and all(isinstance(v, (int, float)) for v in obj_loc[:3])
                    ):
                        continue
                    object_ego_loc = world_to_ego(obj_loc[:3], ego_world_loc, ego_yaw)
                    angle, distance = calculate_relative_angle_and_distance(object_ego_loc)
                    rows.append(
                        {
                            "map_name": record.map_name,
                            "case_id": record.case_id,
                            "frame_id": frame_id,
                            "angle": angle % 360.0,
                            "distance": distance,
                            "class": group_name,
                        }
                    )

    if missing_labels:
        logger.warning(
            "Missing labels.json for {} cases: {}",
            len(missing_labels),
            ", ".join(missing_labels[:10]) + ("..." if len(missing_labels) > 10 else ""),
        )
    if missing_ego:
        logger.warning(
            "Missing ego.csv for {} cases in spatial statistic: {}",
            len(missing_ego),
            ", ".join(missing_ego[:10]) + ("..." if len(missing_ego) > 10 else ""),
        )

    return rows, grouped_counts, missing_labels, missing_ego


def print_obstacle_spatial_summary(rows: list[dict[str, Any]], grouped_counts: Counter[str]) -> None:
    obstacle_rows = [row for row in rows if row["class"] == "obstacles"]
    near_field_count = sum(1 for row in obstacle_rows if row["distance"] <= 10.0)
    ratio = (near_field_count / len(obstacle_rows) * 100.0) if obstacle_rows else 0.0
    summary_rows = [
        ("Obstacle samples", format_int(len(obstacle_rows))),
        ("Near-field (0-10m)", f"{near_field_count:,} ({ratio:.1f}%)"),
        ("Grouped counts", summarize_counter(grouped_counts)),
    ]
    print_table("Ego-Centric Obstacle Spatial Summary", summary_rows)


def plot_polar_heatmap(frequency: Any, angle_edges: Any, distance_edges: Any, title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 9

    theta = np.radians(angle_edges)
    positive = frequency[frequency > 0]
    norm = PowerNorm(
        gamma=1.5,
        vmin=float(np.quantile(positive, 0.05)) if positive.size > 0 else 0.0,
        vmax=float(np.quantile(frequency, 0.98)) if frequency.size > 0 else 0.0,
    )

    colors = [
        (0.95, 0.98, 0.98, 0.0),
        (0.75, 0.92, 0.90, 0.8),
        (0.20, 0.70, 0.65, 0.9),
        (0.05, 0.35, 0.35, 1.0),
    ]
    cmap = LinearSegmentedColormap.from_list("academic_teal", colors, N=256)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 5), dpi=300)
    mesh = ax.pcolormesh(
        theta,
        distance_edges,
        frequency.T,
        norm=norm,
        cmap=cmap,
        shading="auto",
        edgecolors="face",
        linewidth=0,
        rasterized=True,
    )
    ax.grid(True, color="#B0B0B0", linestyle="--", linewidth=0.5, alpha=0.5, zorder=10)
    ax.set_rticks(distance_edges[::3])
    ax.set_yticklabels([f"{int(d)}m" for d in distance_edges[::3]], fontsize=8, color="#404040")
    ax.set_rlabel_position(25)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    angles = list(range(0, 360, 45))
    labels = ["Front", "45°", "Right", "135°", "Rear", "225°", "Left", "315°"]
    ax.set_xticks(np.radians(angles))
    ax.set_xticklabels(labels, fontsize=9, fontweight="medium")

    cbar = fig.colorbar(mesh, ax=ax, pad=0.12, shrink=0.7, aspect=25, location="right")
    cbar.solids.set_edgecolor("face")
    cbar.set_label("Spectral Density", fontsize=10, labelpad=10)
    cbar.ax.tick_params(labelsize=8, color="#666666")

    if title:
        ax.set_title(title, fontsize=12, pad=15, y=1.05)
    ax.spines["polar"].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", transparent=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)


def plot_obstacle_spatial_distribution(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import numpy as np

    if not rows:
        logger.warning("No spatial object rows found for obstacle polar plot")
        return

    obstacle_rows = [row for row in rows if row["class"] == "obstacles"]
    if not obstacle_rows:
        logger.warning("No obstacle rows found for obstacle polar plot")
        return

    custom_min_distance = 0.0
    custom_max_distance = 40.0
    filtered_rows = [
        row
        for row in obstacle_rows
        if custom_min_distance <= float(row["distance"]) <= custom_max_distance
    ]
    if not filtered_rows:
        logger.warning("No obstacle rows remain after distance filtering")
        return

    angle_bins = np.linspace(0.0, 360.0, 19)
    distance_bins = np.linspace(custom_min_distance, custom_max_distance, 9)
    frequency, angle_edges, distance_edges = np.histogram2d(
        [float(row["angle"]) for row in filtered_rows],
        [float(row["distance"]) for row in filtered_rows],
        bins=[angle_bins, distance_bins],
    )
    png_path = output_dir / f"{OBSTACLE_POLAR_OUTPUT_PREFIX}.png"
    plot_polar_heatmap(
        frequency,
        angle_edges,
        distance_edges,
        title="Distribution of obstacles Around Ego Agent",
        output_path=png_path,
    )
    logger.info(f"Saved obstacle polar figure: {png_path}")
    logger.info(f"Saved obstacle polar figure: {png_path.with_suffix('.pdf')}")


def run_ego_centric_obstacle_spatial_distribution(records: list[CaseRecord]) -> None:
    rows, grouped_counts, _, _ = collect_ego_centric_spatial_objects(records)
    print_obstacle_spatial_summary(rows, grouped_counts)
    plot_obstacle_spatial_distribution(rows, OUTPUT_DIR)


def collect_object_group_counts(records: list[CaseRecord]) -> tuple[dict[str, Counter[str]], list[str]]:
    logger.info("Building Object Group Counts")
    class_to_group = {
        cls: group for group, classes in OBJECT_GROUP_CLASS_GROUP.items() for cls in classes
    }
    all_classes = set(class_to_group.keys())
    grouped_counts: dict[str, Counter[str]] = {
        group: Counter() for group in OBJECT_GROUP_CLASS_GROUP.keys()
    }
    missing_labels = []

    for record in records:
        if not record.paths.labels.exists():
            missing_labels.append(f"{record.map_name}/{record.case_id}")
            continue
        labels = read_json(record.paths.labels, default=[])
        if not isinstance(labels, list):
            continue
        for frame in labels:
            if not isinstance(frame, dict):
                continue
            objects = frame.get("objects", [])
            if not isinstance(objects, list):
                continue
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                cls = obj.get("class")
                occlusion = safe_int(obj.get("occlusion"))
                if cls not in all_classes or occlusion is None or occlusion >= 3:
                    continue
                grouped_counts[class_to_group[str(cls)]][str(cls)] += 1

    if missing_labels:
        logger.warning(
            "Missing labels.json for {} cases in object-group statistic: {}",
            len(missing_labels),
            ", ".join(missing_labels[:10]) + ("..." if len(missing_labels) > 10 else ""),
        )

    return grouped_counts, missing_labels


def print_object_group_summary(grouped_counts: dict[str, Counter[str]]) -> None:
    rows = []
    for group, classes in OBJECT_GROUP_CLASS_GROUP.items():
        total = sum(grouped_counts[group].values())
        parts = [f"{cls}={grouped_counts[group].get(cls, 0):,}" for cls in classes]
        rows.append((group, f"total={total:,}; " + ", ".join(parts)))
    print_table("Object Group Summary", rows)


def plot_object_group_counts(grouped_counts: dict[str, Counter[str]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.ticker as ticker
    from matplotlib import rcParams

    groups = list(OBJECT_GROUP_CLASS_GROUP.keys())
    totals = [sum(grouped_counts[group].values()) for group in groups]
    if max(totals, default=0) <= 0:
        logger.warning("No grouped object counts found for object-group plot")
        return

    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 5.0), dpi=300)
    ax.set_yscale("function", functions=(lambda x: x**0.5, lambda x: x**2))
    ax.yaxis.set_major_locator(
        ticker.FixedLocator([0, 1000, 4000, 8000, 20000, 40000, 80000, 120000])
    )

    def y_fmt(x: float, _: Any) -> str:
        if x == 0:
            return "0"
        if x >= 1000:
            return f"{int(x / 1000)}k"
        return f"{int(x)}"

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))

    x = np.arange(len(groups))
    bar_width = 0.65
    max_visual_height = math.sqrt(max(totals))

    for i, group in enumerate(groups):
        total_group = sum(grouped_counts[group].values())
        bottom = 0
        sub_classes = [
            cls
            for cls in OBJECT_GROUP_CLASS_GROUP[group]
            if grouped_counts[group].get(cls, 0) > 0
        ]
        sub_classes.sort(key=lambda cls: grouped_counts[group].get(cls, 0), reverse=True)

        small_items = []
        for cls in sub_classes:
            count = grouped_counts[group][cls]
            vis_h = (math.sqrt(bottom + count) - math.sqrt(bottom)) / max_visual_height
            if vis_h <= 0.04:
                small_items.append(cls)
            bottom += count

        bottom = 0
        small_item_idx = 0
        for cls in sub_classes:
            count = grouped_counts[group][cls]
            color = OBJECT_GROUP_SUB_COLORS[group][cls]
            ratio_in_group = count / total_group if total_group > 0 else 0.0
            ax.bar(
                x[i],
                count,
                bottom=bottom,
                width=bar_width,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=cls,
            )

            label_text = f"{ratio_in_group:.1%}"
            visual_height_ratio = (math.sqrt(bottom + count) - math.sqrt(bottom)) / max_visual_height
            center_y = bottom + count / 2.0
            if visual_height_ratio > 0.04:
                ax.text(
                    x[i],
                    center_y,
                    label_text,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=6,
                    fontweight="bold",
                )
            elif ratio_in_group > 0.001:
                is_last_cols = i >= len(groups) - 3
                direction = -1 if is_last_cols else 1
                anchor_x = x[i] + (bar_width / 2 * direction)
                anchor_y = center_y
                offset_x_points = 12 * direction
                if len(small_items) == 2:
                    offset_y_points = 12
                else:
                    offset_y_points = 12
                small_item_idx += 1
                ax.annotate(
                    label_text,
                    xy=(anchor_x, anchor_y),
                    xytext=(offset_x_points, offset_y_points),
                    textcoords="offset points",
                    ha="right" if is_last_cols else "left",
                    va="center",
                    fontsize=6,
                    color="black",
                    annotation_clip=False,
                    arrowprops=dict(
                        arrowstyle="-",
                        color="black",
                        lw=0.5,
                        shrinkA=0,
                        shrinkB=2,
                    ),
                )
            bottom += count

        if total_group > 0:
            ax.text(
                x[i],
                bottom * 1.05,
                f"{int(total_group):,}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="black",
                fontweight="bold",
            )

    ax.set_ylabel("Object Count")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")

    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for group in groups:
        for cls in OBJECT_GROUP_CLASS_GROUP[group]:
            if cls in labels:
                idx = labels.index(cls)
                by_label[cls] = handles[idx]

    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=7,
    )

    fig.tight_layout()
    png_path = output_dir / f"{OBJECT_GROUP_OUTPUT_PREFIX}.png"
    pdf_path = output_dir / f"{OBJECT_GROUP_OUTPUT_PREFIX}.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved object-group figure: {png_path}")
    logger.info(f"Saved object-group figure: {pdf_path}")


def run_object_group_counts(records: list[CaseRecord]) -> None:
    grouped_counts, _ = collect_object_group_counts(records)
    print_object_group_summary(grouped_counts)
    plot_object_group_counts(grouped_counts, OUTPUT_DIR)


def run_ego_motion_statistics(records: list[CaseRecord]) -> None:
    samples = collect_ego_motion_samples(records)
    print_ego_motion_summary(samples)
    plot_ego_motion_statistics(samples, OUTPUT_DIR)
    plot_sensor_pose_motion_jitter(records, OUTPUT_DIR)


def audit_dataset(records: list[CaseRecord], include_storage: bool) -> dict[str, Any]:
    file_presence = Counter()
    maps = Counter()
    scene_kinds = Counter()
    route_types = Counter()
    ego_behaviors = Counter()
    weather_regimes = Counter()
    sun_buckets = Counter()
    precipitation_regimes = Counter()

    total_frames = 0
    total_duration = 0.0
    sensor_counts: list[int] = []
    total_annotations = 0
    object_counts: Counter[str] = Counter()

    for record in records:
        maps[record.map_name] += 1
        paths = record.paths
        for name, path in (
            ("manifest", paths.manifest),
            ("case_json", paths.case_json),
            ("sampling_log", paths.sampling_log),
            ("calibration", paths.calibration),
            ("labels", paths.labels),
            ("ego_csv", paths.ego_csv),
            ("sensor_pose_csv", paths.sensor_pose_csv),
            ("pedestrian_csv", paths.pedestrian_csv),
            ("vehicle_csv", paths.vehicle_csv),
            ("obstacles_csv", paths.obstacles_csv),
            ("run_summary", paths.run_summary),
            ("run_result", paths.run_result),
        ):
            if path.exists():
                file_presence[name] += 1

        skeleton = record.case_json.get("skeleton", {})
        scene_kinds[str(skeleton.get("scene_kind", "Unknown"))] += 1
        main_route = skeleton.get("main_route") or {}
        route_types[str(main_route.get("route_type", "Unknown"))] += 1

        plan = record.case_json.get("plan", {})
        ego_behaviors[str(plan.get("ego_behavior", "Unknown"))] += 1

        weather = record.sampling_log.get("weather", {})
        if isinstance(weather, dict):
            weather_regimes[str(weather.get("regime") or weather.get("preset") or "Unknown")] += 1
            sun_buckets[str(weather.get("sun_bucket", "Unknown"))] += 1
            precipitation_regimes[
                str(
                    weather.get("precipitation_regime")
                    or weather.get("precipitation_bucket")
                    or "Unknown"
                )
            ] += 1

        if record.frame_num is not None:
            total_frames += record.frame_num
        if record.duration_sec is not None and math.isfinite(record.duration_sec):
            total_duration += record.duration_sec
        if record.sensor_count is not None:
            sensor_counts.append(record.sensor_count)

        annotation_count, class_counts = count_label_annotations(paths.labels)
        total_annotations += annotation_count
        object_counts.update(class_counts)

    storage_bytes = (
        sum(directory_size_bytes(r.paths.case_dir) for r in records)
        if include_storage
        else None
    )

    return {
        "cases": len(records),
        "valid_sampled_cases": sum(r.is_valid_sampled for r in records),
        "maps": maps,
        "file_presence": file_presence,
        "scene_kinds": scene_kinds,
        "route_types": route_types,
        "ego_behaviors": ego_behaviors,
        "weather_regimes": weather_regimes,
        "sun_buckets": sun_buckets,
        "precipitation_regimes": precipitation_regimes,
        "total_frames": total_frames,
        "total_duration_sec": total_duration,
        "sensor_counts": sensor_counts,
        "total_annotations": total_annotations,
        "object_counts": object_counts,
        "storage_bytes": storage_bytes,
    }


def print_audit(summary: dict[str, Any]) -> None:
    sensor_counts = summary["sensor_counts"]
    unique_sensor_counts = sorted(set(sensor_counts))
    print("Dataset audit")
    print(f"  cases: {summary['cases']}")
    print(f"  valid_sampled_cases: {summary['valid_sampled_cases']}")
    print(f"  maps: {summarize_counter(summary['maps'])}")
    print(f"  total_frames: {summary['total_frames']}")
    print(f"  total_duration_sec: {summary['total_duration_sec']:.3f}")
    if sensor_counts:
        print(
            "  sensor_count_per_case: "
            f"min={min(sensor_counts)}, max={max(sensor_counts)}, "
            f"unique={unique_sensor_counts}"
        )
    else:
        print("  sensor_count_per_case: -")
    print(f"  total_annotations: {summary['total_annotations']}")
    storage_bytes = summary["storage_bytes"]
    if storage_bytes is not None:
        print(f"  storage_size: {human_bytes(storage_bytes)}")
    print()
    print("File availability")
    print(f"  {summarize_counter(summary['file_presence'])}")
    print()
    print("Semantic preview")
    print(f"  scene_kinds: {summarize_counter(summary['scene_kinds'])}")
    print(f"  route_types: {summarize_counter(summary['route_types'])}")
    print(f"  ego_behaviors: {summarize_counter(summary['ego_behaviors'])}")
    print(f"  weather_regimes: {summarize_counter(summary['weather_regimes'], top_n=12)}")
    print(f"  sun_buckets: {summarize_counter(summary['sun_buckets'])}")
    print(
        "  precipitation_regimes: "
        f"{summarize_counter(summary['precipitation_regimes'])}"
    )
    print()
    print("Object preview")
    print(f"  {summarize_counter(summary['object_counts'], top_n=20)}")


PENDING_STATISTICS = [
    "dataset_scale_overview",
    "route_type_coverage",
    "weather_continuous_diversity",
    "background_flow_analysis",
    "ego_motion_statistics_panel",
    "ego_centric_obstacle_spatial_distribution",
    "object_group_counts",
]


def print_pending() -> None:
    print("Pending statistic nodes:")
    for name in PENDING_STATISTICS:
        print(f"  - {name}")
    print()
    print("Each node should be implemented after confirming:")
    print("  - case filtering rules")
    print("  - exact counting fields")
    print("  - aggregation level")
    print("  - visual encoding and output formats")


def main() -> None:
    records = run_dataset_scale_overview()
    run_route_type_coverage(records)
    run_weather_diversity_statistics(records)
    run_background_flow_analysis(records)
    run_ego_motion_statistics(records)
    run_ego_centric_obstacle_spatial_distribution(records)
    run_object_group_counts(records)


if __name__ == "__main__":
    main()
