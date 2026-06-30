from __future__ import annotations

from pathlib import Path

import numpy as np


IMAGE_SIZE = (1280, 720)

WORLD_POINTS = np.array(
    [
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 3.0, 0.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
        [0.0, 2.0, 1.0],
    ],
    dtype=np.float64,
)

IMAGE_POINTS = {
    "camera_1": np.array(
        [
            [894.0, 487.0],
            [767.0, 232.0],
            [780.0, 640.0],
            [894.0, 232.0],
            [1060.0, 360.0],
            [767.0, 487.0],
        ],
        dtype=np.float64,
    ),
    "camera_2": np.array(
        [
            [494.0, 455.0],
            [554.0, 274.0],
            [346.0, 465.0],
            [614.0, 334.0],
            [612.0, 465.0],
            [433.0, 395.0],
        ],
        dtype=np.float64,
    ),
}

RED_PIXELS = {
    "camera_1": np.array([906.0, 626.0], dtype=np.float64),
    "camera_2": np.array([423.0, 523.0], dtype=np.float64),
}


def homogeneous(points: np.ndarray) -> np.ndarray:
    return np.column_stack([points, np.ones(points.shape[0], dtype=points.dtype)])


def normalize_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2.0) / mean_distance if mean_distance > 0.0 else 1.0
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalized = (transform @ homogeneous(points).T).T[:, :2]
    return normalized, transform


def normalize_3d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(3.0) / mean_distance if mean_distance > 0.0 else 1.0
    transform = np.array(
        [
            [scale, 0.0, 0.0, -scale * centroid[0]],
            [0.0, scale, 0.0, -scale * centroid[1]],
            [0.0, 0.0, scale, -scale * centroid[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalized = (transform @ homogeneous(points).T).T[:, :3]
    return normalized, transform


def estimate_projection_matrix(world_points: np.ndarray, image_points: np.ndarray) -> np.ndarray:
    world_normalized, world_transform = normalize_3d(world_points)
    image_normalized, image_transform = normalize_2d(image_points)

    rows: list[list[float]] = []
    for point_3d, point_2d in zip(world_normalized, image_normalized):
        x, y, z = point_3d
        u, v = point_2d
        xh = [x, y, z, 1.0]
        rows.append([*xh, 0.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u * z, -u])
        rows.append([0.0, 0.0, 0.0, 0.0, *xh, -v * x, -v * y, -v * z, -v])

    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    projection_normalized = vt[-1].reshape(3, 4)
    projection = np.linalg.inv(image_transform) @ projection_normalized @ world_transform
    return projection / np.linalg.norm(projection[2, :3])


def rq_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # RQ via QR on a reversed matrix. This keeps the implementation numpy-only.
    q_reversed, r_reversed = np.linalg.qr(np.flipud(np.fliplr(matrix)).T)
    upper = np.flipud(np.fliplr(r_reversed.T))
    orthogonal = np.flipud(np.fliplr(q_reversed.T))
    return upper, orthogonal


def decompose_projection_matrix(projection: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    best_score = np.inf

    # A projection matrix has arbitrary scale. Try both global scales and all diagonal
    # RQ sign corrections, then keep the physically conventional decomposition.
    for scale in (1.0, -1.0):
        scaled_projection = scale * projection
        left = scaled_projection[:, :3]
        raw_intrinsics, raw_rotation = rq_decomposition(left)

        for signs in ((sx, sy, sz) for sx in (1.0, -1.0) for sy in (1.0, -1.0) for sz in (1.0, -1.0)):
            sign_correction = np.diag(signs)
            intrinsics = raw_intrinsics @ sign_correction
            rotation = sign_correction @ raw_rotation
            if abs(intrinsics[2, 2]) < 1e-12:
                continue

            intrinsics = intrinsics / intrinsics[2, 2]
            determinant = np.linalg.det(rotation)
            if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0 or determinant <= 0.0:
                continue

            camera_center = -np.linalg.inv(left) @ scaled_projection[:, 3]
            score = (
                abs(determinant - 1.0)
                + np.linalg.norm(rotation.T @ rotation - np.eye(3))
                + abs(intrinsics[2, 2] - 1.0)
            )
            if score < best_score:
                best_score = score
                best = (intrinsics, rotation, camera_center)

    if best is None:
        raise RuntimeError("Could not decompose projection matrix into positive K and proper R.")
    return best


def triangulate_point(
    projections: list[np.ndarray],
    image_points: list[np.ndarray],
) -> np.ndarray:
    rows = []
    for projection, point in zip(projections, image_points):
        u, v = point
        rows.append(u * projection[2] - projection[0])
        rows.append(v * projection[2] - projection[1])

    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    point_h = vt[-1]
    return point_h[:3] / point_h[3]


def project_points(projection: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    projected_h = (projection @ homogeneous(world_points).T).T
    return projected_h[:, :2] / projected_h[:, 2:3]


def format_vector(vector: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in vector) + "]"


def format_matrix(matrix: np.ndarray) -> str:
    return "\n".join("  " + format_vector(row) for row in matrix)


def solve() -> dict[str, object]:
    projections = {
        camera_name: estimate_projection_matrix(WORLD_POINTS, image_points)
        for camera_name, image_points in IMAGE_POINTS.items()
    }
    decompositions = {
        camera_name: decompose_projection_matrix(projection)
        for camera_name, projection in projections.items()
    }
    red_point = triangulate_point(
        [projections["camera_1"], projections["camera_2"]],
        [RED_PIXELS["camera_1"], RED_PIXELS["camera_2"]],
    )

    blue_errors = {}
    red_errors = {}
    rotation_checks = {}
    for camera_name, projection in projections.items():
        reprojected_blue = project_points(projection, WORLD_POINTS)
        blue_errors[camera_name] = np.linalg.norm(reprojected_blue - IMAGE_POINTS[camera_name], axis=1)

        reprojected_red = project_points(projection, red_point.reshape(1, 3))[0]
        red_errors[camera_name] = float(np.linalg.norm(reprojected_red - RED_PIXELS[camera_name]))

        _, rotation, _ = decompositions[camera_name]
        rotation_checks[camera_name] = {
            "orthogonality_error": float(np.linalg.norm(rotation.T @ rotation - np.eye(3))),
            "determinant": float(np.linalg.det(rotation)),
        }

    return {
        "image_size": IMAGE_SIZE,
        "projections": projections,
        "decompositions": decompositions,
        "red_point": red_point,
        "blue_errors": blue_errors,
        "red_errors": red_errors,
        "rotation_checks": rotation_checks,
    }


def build_report(results: dict[str, object]) -> str:
    red_point = results["red_point"]
    decompositions = results["decompositions"]
    blue_errors = results["blue_errors"]
    red_errors = results["red_errors"]
    rotation_checks = results["rotation_checks"]

    lines = [
        "BeeX Task 3: 3D Computer Vision Answer",
        "",
        "Method:",
        "- I estimated each camera projection matrix with normalized DLT.",
        "- I decomposed each projection matrix into K, R, and camera center using a numpy-only RQ decomposition.",
        "- I triangulated the red feature point with a linear SVD solve.",
        "",
        "Red feature point XYZ in world ENU metres:",
        format_vector(red_point),
        "",
    ]

    for camera_name in ("camera_1", "camera_2"):
        intrinsics, rotation, center = decompositions[camera_name]
        checks = rotation_checks[camera_name]
        lines.extend(
            [
                f"{camera_name}:",
                "Intrinsic matrix K:",
                format_matrix(intrinsics),
                "Rotation matrix R:",
                format_matrix(rotation),
                "Camera position XYZ in world ENU metres:",
                format_vector(center),
                f"Rotation determinant: {checks['determinant']:.12f}",
                f"Rotation orthogonality error: {checks['orthogonality_error']:.3e}",
                f"Max blue reprojection error: {np.max(blue_errors[camera_name]):.6f} px",
                f"Mean blue reprojection error: {np.mean(blue_errors[camera_name]):.6f} px",
                f"Red reprojection error: {red_errors[camera_name]:.6f} px",
                "",
            ]
        )

    lines.extend(
        [
            "Run instruction:",
            "python3 task3_cv.py",
            "",
            "Dependency note:",
            "This script uses numpy only for numerical computation and does not call OpenCV.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    results = solve()
    report = build_report(results)
    output_path = Path(__file__).resolve().parent / "answers.txt"
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
