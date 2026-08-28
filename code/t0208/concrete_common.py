from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any
import numpy as np


SensorMap = dict[int, np.ndarray]
SpecimenMap = dict[str, np.ndarray]
CuboidMap = dict[str, list[str]]

@dataclass(frozen=True)
class GeometryConfig:
    sensor_positions: SensorMap
    specimen: SpecimenMap
    cuboids: CuboidMap
    cuboid_bounds: list[tuple[float, float, float, float, float, float]]
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    specimen_shape: str = "box"
    cylinder_radius: float | None = None
    cylinder_center_xy: tuple[float, float] = (0.0, 0.0)


def build_default_geometry() -> GeometryConfig:
    #sensor座標
    sensor_cyl_cm = {
        1: (5, 0, -6.5),
        2: (5, 120, -7),
        3: (5, 240, -7.5),
        4: (5, 60, -11.65),
        5: (5, 180, -11.15),
        6: (5, 300, -10.65)
    }
    def cyl_to_xyz(radius, theta_deg, z):
        theta = np.deg2rad(theta_deg)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return (x, y, z)
    sensor_cm = {
        sid: cyl_to_xyz(radius, theta_deg, z)
        for sid, (radius, theta_deg, z) in sensor_cyl_cm.items()
    }
    sensor_positions = {
        sid: np.asarray(position, dtype=float) / 100.0
        for sid, position in sensor_cm.items()
    }

    
    # =========================
    # 試體座標：圓柱，單位 cm
    # 半徑 5 cm，高度 20 cm，上下表面要幾個點
    # 上表面中心為 (0,0,0)
    # 下表面中心為 (0,0,-20)
    # =========================
    CYL_RADIUS_CM = 5.0
    CYL_HEIGHT_CM = 18.85
    N_CIRCLE = 36

    specimen_cm: dict[str, tuple[float, float, float]] = {}

    for i, theta in enumerate(np.linspace(0, 2 * np.pi, N_CIRCLE, endpoint=False)):
        x = CYL_RADIUS_CM * np.cos(theta)
        y = CYL_RADIUS_CM * np.sin(theta)

        specimen_cm[f"T{i+1}"] = (x, y, 0.0)
        specimen_cm[f"B{i+1}"] = (x, y, -CYL_HEIGHT_CM)

    specimen_cm["CENTER_TOP"] = (0.0, 0.0, 0.0)
    specimen_cm["CENTER_BOTTOM"] = (0.0, 0.0, -CYL_HEIGHT_CM)
    specimen = {
        point_id: np.asarray(position, dtype=float) / 100.0
        for point_id, position in specimen_cm.items()
    }
    
    cuboids: CuboidMap = {}
    xmin = -CYL_RADIUS_CM / 100.0
    xmax = CYL_RADIUS_CM / 100.0
    ymin = -CYL_RADIUS_CM / 100.0
    ymax = CYL_RADIUS_CM / 100.0
    zmin = -CYL_HEIGHT_CM / 100.0
    zmax = 0.0

    cuboid_bounds = [
        (
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
        )
    ]

    return GeometryConfig(
        sensor_positions=sensor_positions,
        specimen=specimen,
        cuboids=cuboids,
        cuboid_bounds=cuboid_bounds,
        xmin=float(xmin),
        xmax=float(xmax),
        ymin=float(ymin),
        ymax=float(ymax),
        zmin=float(zmin),
        zmax=float(zmax),
        specimen_shape="cylinder",
        cylinder_radius=CYL_RADIUS_CM / 100.0,
        cylinder_center_xy=(0.0, 0.0),
    )

def ensure_directory(path: Path | str) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def project_to_union(
    point4: np.ndarray,
    cuboid_bounds: list[tuple[float, float, float, float, float, float]],
) -> np.ndarray:
    best_point: np.ndarray | None = None
    best_distance = float("inf")
    for xmin, xmax, ymin, ymax, zmin, zmax in cuboid_bounds:
        projected = np.asarray(point4, dtype=float).copy()
        projected[0] = np.clip(projected[0], xmin, xmax)
        projected[1] = np.clip(projected[1], ymin, ymax)
        projected[2] = np.clip(projected[2], zmin, zmax)
        distance = float(np.sum((projected[:3] - point4[:3]) ** 2))
        if distance < best_distance:
            best_distance = distance
            best_point = projected

    if best_point is None:
        raise ValueError("No cuboid bounds were provided.")
    return best_point


def point_mask_for_geometry(points: np.ndarray, geometry: GeometryConfig) -> np.ndarray:
    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(1, -1)

    if geometry.specimen_shape == "cylinder" and geometry.cylinder_radius is not None:
        center_x, center_y = geometry.cylinder_center_xy
        radial_squared = (coordinates[:, 0] - center_x) ** 2 + (coordinates[:, 1] - center_y) ** 2
        return (
            (radial_squared <= (geometry.cylinder_radius + 1e-12) ** 2)
            & (coordinates[:, 2] >= geometry.zmin)
            & (coordinates[:, 2] <= geometry.zmax)
        )

    inside = np.zeros(coordinates.shape[0], dtype=bool)
    for xmin, xmax, ymin, ymax, zmin, zmax in geometry.cuboid_bounds:
        inside |= (
            (coordinates[:, 0] >= xmin)
            & (coordinates[:, 0] <= xmax)
            & (coordinates[:, 1] >= ymin)
            & (coordinates[:, 1] <= ymax)
            & (coordinates[:, 2] >= zmin)
            & (coordinates[:, 2] <= zmax)
        )

    return inside


def project_to_geometry(point4: np.ndarray, geometry: GeometryConfig) -> np.ndarray:
    if geometry.specimen_shape == "cylinder" and geometry.cylinder_radius is not None:
        projected = np.asarray(point4, dtype=float).copy()
        center_x, center_y = geometry.cylinder_center_xy
        if not np.isfinite(projected[0]):
            projected[0] = center_x
        if not np.isfinite(projected[1]):
            projected[1] = center_y
        if not np.isfinite(projected[2]):
            projected[2] = 0.5 * (geometry.zmin + geometry.zmax)

        projected[2] = np.clip(projected[2], geometry.zmin, geometry.zmax)
        dx = projected[0] - center_x
        dy = projected[1] - center_y
        radius = float(np.hypot(dx, dy))
        if radius > geometry.cylinder_radius and radius > 0:
            scale = geometry.cylinder_radius / radius
            projected[0] = center_x + dx * scale
            projected[1] = center_y + dy * scale
        return projected

    return project_to_union(np.asarray(point4, dtype=float), geometry.cuboid_bounds)


def save_results_pickle(path: Path | str, results: list[dict[str, Any]], geometry: GeometryConfig) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    payload = {
        "results": results,
        "sensor_positions": geometry.sensor_positions,
        "cuboids": geometry.cuboids,
        "specimen": geometry.specimen,
        "cuboid_bounds": geometry.cuboid_bounds,
        "xmin": geometry.xmin,
        "xmax": geometry.xmax,
        "ymin": geometry.ymin,
        "ymax": geometry.ymax,
        "zmin": geometry.zmin,
        "zmax": geometry.zmax,
        "specimen_shape": geometry.specimen_shape,
        "cylinder_radius": geometry.cylinder_radius,
        "cylinder_center_xy": geometry.cylinder_center_xy,
    }
    with output_path.open("wb") as stream:
        pickle.dump(payload, stream)
    return output_path


def load_results_pickle(path: Path | str) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        return pickle.load(stream)

