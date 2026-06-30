from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np

import task3_cv


class Task3ComputerVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = task3_cv.solve()

    def test_solution_imports_only_numpy_and_standard_library(self) -> None:
        source_path = Path(__file__).resolve().parent / "task3_cv.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imports: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.add(node.module.split(".")[0])

        self.assertLessEqual(imports, {"__future__", "pathlib", "numpy"})

    def test_projection_matrices_reproject_blue_points_subpixel(self) -> None:
        for camera_name, projection in self.results["projections"].items():
            reprojected = task3_cv.project_points(projection, task3_cv.WORLD_POINTS)
            errors = np.linalg.norm(reprojected - task3_cv.IMAGE_POINTS[camera_name], axis=1)
            self.assertLess(float(np.max(errors)), 1.0)

    def test_camera_decomposition_has_valid_intrinsics_and_rotations(self) -> None:
        for intrinsics, rotation, camera_center in self.results["decompositions"].values():
            self.assertGreater(float(intrinsics[0, 0]), 0.0)
            self.assertGreater(float(intrinsics[1, 1]), 0.0)
            self.assertAlmostEqual(float(intrinsics[2, 2]), 1.0, places=12)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)
            self.assertLess(float(np.linalg.norm(rotation.T @ rotation - np.eye(3))), 1e-12)
            self.assertTrue(np.all(np.isfinite(camera_center)))

    def test_red_point_matches_expected_solution_and_reprojects(self) -> None:
        expected_red = np.array([0.993713, 2.990969, 0.470963])
        red_point = self.results["red_point"]

        self.assertLess(float(np.linalg.norm(red_point - expected_red)), 1e-5)
        for camera_name, projection in self.results["projections"].items():
            reprojected = task3_cv.project_points(projection, red_point.reshape(1, 3))[0]
            error = np.linalg.norm(reprojected - task3_cv.RED_PIXELS[camera_name])
            self.assertLess(float(error), 1.0)

    def test_report_contains_required_answer_sections(self) -> None:
        report = task3_cv.build_report(self.results)

        self.assertIn("Red feature point XYZ", report)
        self.assertIn("Intrinsic matrix K", report)
        self.assertIn("Rotation matrix R", report)
        self.assertIn("Camera position XYZ", report)
        self.assertIn("numpy only", report)


if __name__ == "__main__":
    unittest.main()
