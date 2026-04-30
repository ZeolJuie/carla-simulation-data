#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "occ" / "default.toml"
DEFAULT_FREE_LABEL = 10
DEFAULT_EMPTY_LABEL = 0


OCC_LABEL_NAMES = {
    0: "Road",
    1: "Sidewalk",
    2: "Building",
    3: "TrafficElem",
    4: "Vegetation",
    5: "Human",
    6: "TwoWheel",
    7: "Vehicle",
    8: "Obstacle",
    9: "Ground",
    10: "Free",
}


OCC_COLORS = {
    0: "#4E79A7",
    1: "#F28E2B",
    2: "#9C755F",
    3: "#B07AA1",
    4: "#59A14F",
    5: "#E15759",
    6: "#EDC948",
    7: "#76B7B2",
    8: "#BAB0AC",
    9: "#8CD17D",
    10: "#E6EEF7",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one OCC sample directory or labels.npz by rendering a 3D voxel scatter "
            "and orthographic occupancy slices into a PNG."
        )
    )
    parser.add_argument(
        "input_path",
        help="Path to an OCC sample directory like .../occ/sample_002120 or directly to labels.npz.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="OCC TOML config used to recover voxel size and point-cloud range.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to <sample_dir>/occ_preview.png or <labels_parent>/occ_preview.png.",
    )
    parser.add_argument(
        "--free-label",
        type=int,
        default=DEFAULT_FREE_LABEL,
        help="Semantic label treated as free space and hidden by default.",
    )
    parser.add_argument(
        "--empty-label",
        type=int,
        default=DEFAULT_EMPTY_LABEL,
        help="Semantic label treated as empty/unlabeled and hidden by default.",
    )
    parser.add_argument(
        "--show-free",
        action="store_true",
        help="Render free voxels as well.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=30000,
        help="Maximum number of voxels drawn in the 3D panel.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=28.0,
        help="3D camera elevation in degrees.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-58.0,
        help="3D camera azimuth in degrees.",
    )
    return parser.parse_args()


def resolve_labels_path(input_path):
    path = Path(input_path).resolve()
    if path.is_file():
        return path
    candidate = path / "labels.npz"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("Could not find labels.npz under {}".format(path))


def load_occ_config(path):
    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    occ_cfg = payload.get("occ", payload)
    return {
        "voxel_size": float(occ_cfg["voxel_size"]),
        "pc_range": np.asarray(occ_cfg["pc_range"], dtype=np.float64),
        "occ_size": tuple(int(value) for value in occ_cfg["occ_size"]),
    }


def load_sample(labels_path):
    payload = np.load(str(labels_path))
    semantics = payload["semantics"]
    mask_camera = payload["mask_camera"] if "mask_camera" in payload.files else None
    mask_lidar = payload["mask_lidar"] if "mask_lidar" in payload.files else None
    return semantics, mask_camera, mask_lidar


def voxel_indices_to_world(voxels, voxel_size, pc_range, occ_size):
    if voxels.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    voxels = np.asarray(voxels, dtype=np.float64).copy()
    voxels[:, 1] = (occ_size[1] - 1) - voxels[:, 1]
    world_points = (voxels + 0.5) * voxel_size
    world_points[:, 0] += pc_range[0]
    world_points[:, 1] += pc_range[1]
    world_points[:, 2] += pc_range[2]
    return world_points


def build_point_cloud(semantics, voxel_size, pc_range, occ_size, free_label, empty_label, show_free):
    occupied = semantics != empty_label
    if not show_free:
        occupied &= semantics != free_label
    voxels = np.argwhere(occupied)
    if voxels.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int8)
    labels = semantics[occupied].astype(np.int8)
    points = voxel_indices_to_world(voxels, voxel_size, pc_range, occ_size)
    return points, labels


def color_for_labels(labels):
    colors = []
    for label in labels:
        colors.append(OCC_COLORS.get(int(label), "#111111"))
    return np.asarray(colors)


def downsample_points(points, labels, max_points):
    if points.shape[0] <= max_points:
        return points, labels
    step = int(math.ceil(points.shape[0] / float(max_points)))
    indices = np.arange(0, points.shape[0], step, dtype=np.int64)
    return points[indices], labels[indices]


def build_bev_label_map(semantics, empty_label, free_label):
    label_map = np.full(semantics.shape[:2], free_label, dtype=np.int16)
    for z_index in range(semantics.shape[2] - 1, -1, -1):
        slice_labels = semantics[:, :, z_index]
        visible = (slice_labels != empty_label) & (slice_labels != free_label)
        replace = visible & (label_map == free_label)
        label_map[replace] = slice_labels[replace]
    return label_map


def build_front_label_map(semantics, empty_label, free_label):
    label_map = np.full((semantics.shape[0], semantics.shape[2]), free_label, dtype=np.int16)
    for y_index in range(semantics.shape[1] - 1, -1, -1):
        slice_labels = semantics[:, y_index, :]
        visible = (slice_labels != empty_label) & (slice_labels != free_label)
        replace = visible & (label_map == free_label)
        label_map[replace] = slice_labels[replace]
    return label_map


def build_side_label_map(semantics, empty_label, free_label):
    label_map = np.full((semantics.shape[1], semantics.shape[2]), free_label, dtype=np.int16)
    for x_index in range(semantics.shape[0] - 1, -1, -1):
        slice_labels = semantics[x_index, :, :]
        visible = (slice_labels != empty_label) & (slice_labels != free_label)
        replace = visible & (label_map == free_label)
        label_map[replace] = slice_labels[replace]
    return label_map


def build_matplotlib_cmap():
    max_label = max(OCC_COLORS.keys())
    color_list = [OCC_COLORS.get(index, "#000000") for index in range(max_label + 1)]
    return ListedColormap(color_list)


def plot_3d(ax, points, labels, voxel_size, elev, azim):
    if points.size == 0:
        ax.text(0.5, 0.5, 0.5, "No occupied voxels", ha="center", va="center")
        ax.set_axis_off()
        return

    colors = color_for_labels(labels)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=max(4.0, voxel_size * 90.0),
        marker="s",
        linewidths=0.0,
        alpha=0.95,
        depthshade=False,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("3D Occupied Voxels")


def plot_label_image(ax, label_map, title, cmap, free_label):
    masked = np.ma.masked_where(label_map == free_label, label_map)
    ax.imshow(masked.T, origin="lower", interpolation="nearest", cmap=cmap, vmin=0, vmax=max(OCC_COLORS.keys()))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def add_legend(ax, labels_present):
    ax.axis("off")
    labels_present = sorted(int(value) for value in labels_present)
    if not labels_present:
        ax.text(0.0, 1.0, "No occupied labels", va="top")
        return

    y = 1.0
    for label in labels_present:
        name = OCC_LABEL_NAMES.get(label, "Class {}".format(label))
        color = OCC_COLORS.get(label, "#111111")
        ax.add_patch(plt.Rectangle((0.0, y - 0.06), 0.08, 0.04, color=color, transform=ax.transAxes, clip_on=False))
        ax.text(0.12, y - 0.04, "{}: {}".format(label, name), transform=ax.transAxes, va="center", fontsize=10)
        y -= 0.09


def summarize_sample(semantics, mask_camera, mask_lidar):
    values, counts = np.unique(semantics, return_counts=True)
    rows = []
    for value, count in zip(values.tolist(), counts.tolist()):
        rows.append("{}={}".format(value, count))
    summary = "labels: " + ", ".join(rows)
    if mask_camera is not None:
        summary += "\nmask_camera sum={}".format(int(mask_camera.sum()))
    if mask_lidar is not None:
        summary += "\nmask_lidar sum={}".format(int(mask_lidar.sum()))
    return summary


def default_output_path(labels_path):
    return labels_path.parent / "occ_preview.png"


def resolve_output_path(requested_output, labels_path):
    if requested_output:
        return Path(requested_output).resolve()

    preferred = default_output_path(labels_path)
    try:
        preferred.parent.mkdir(parents=True, exist_ok=True)
        probe = preferred.parent / ".occ_preview_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return preferred
    except OSError:
        fallback = Path("/tmp") / "occ_preview_{}.png".format(labels_path.parent.name)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback


def main():
    args = parse_args()
    labels_path = resolve_labels_path(args.input_path)
    output_path = resolve_output_path(args.output, labels_path)

    config = load_occ_config(args.config)
    semantics, mask_camera, mask_lidar = load_sample(labels_path)
    if semantics.shape != tuple(config["occ_size"]):
        raise ValueError(
            "labels.npz semantics shape {} does not match config occ_size {}".format(
                semantics.shape,
                config["occ_size"],
            )
        )

    points, labels = build_point_cloud(
        semantics=semantics,
        voxel_size=config["voxel_size"],
        pc_range=config["pc_range"],
        occ_size=config["occ_size"],
        free_label=args.free_label,
        empty_label=args.empty_label,
        show_free=args.show_free,
    )
    points, labels = downsample_points(points, labels, args.max_points)

    bev_map = build_bev_label_map(semantics, args.empty_label, args.free_label)
    front_map = build_front_label_map(semantics, args.empty_label, args.free_label)
    side_map = build_side_label_map(semantics, args.empty_label, args.free_label)
    cmap = build_matplotlib_cmap()
    labels_present = np.unique(labels) if labels.size > 0 else np.empty((0,), dtype=np.int8)

    figure = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 0.65], height_ratios=[1.0, 1.0])

    ax_3d = figure.add_subplot(grid[:, 0], projection="3d")
    plot_3d(ax_3d, points, labels, config["voxel_size"], args.elev, args.azim)

    ax_bev = figure.add_subplot(grid[0, 1])
    plot_label_image(ax_bev, bev_map, "Top-Down BEV", cmap, args.free_label)

    ax_front = figure.add_subplot(grid[1, 1])
    plot_label_image(ax_front, front_map, "Front X-Z Slice", cmap, args.free_label)

    ax_side = figure.add_subplot(grid[0, 2])
    plot_label_image(ax_side, side_map, "Side Y-Z Slice", cmap, args.free_label)

    ax_text = figure.add_subplot(grid[1, 2])
    add_legend(ax_text, labels_present)
    ax_text.text(
        0.0,
        -0.05,
        summarize_sample(semantics, mask_camera, mask_lidar),
        transform=ax_text.transAxes,
        va="top",
        fontsize=9,
        family="monospace",
    )

    figure.suptitle(str(labels_path.parent.name), fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
