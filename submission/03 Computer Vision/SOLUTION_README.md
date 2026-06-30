# BeeX Task 3: 3D Computer Vision Notes

## Objective

I solved for:

- the red feature point XYZ position in the ENU world frame
- both camera intrinsic matrices
- both camera rotation matrices
- both camera positions in the ENU world frame

I wrote the solution in Python 3 and used `numpy` only. I did not use OpenCV or any camera calibration / 3D reconstruction helper.

## Method

I used the six known blue 3D-to-2D correspondences from the PDF for each camera.
Both cameras produce `1280 x 720 px` images, but I estimate each camera independently because the PDF states that their intrinsic matrices may be different. The final `K` matrices are therefore not forced to match.

My implementation does the following:

- estimates a `3 x 4` projection matrix for each camera using normalized DLT
- decomposes each projection matrix into `K`, `R`, and camera center with a hand-written RQ decomposition
- sign-corrects the decomposition so focal lengths are positive and `det(R) = +1`
- triangulates the red point from both camera observations using a linear SVD solve
- reprojects the known blue points and the red point to check the result

## Run

From this folder:

```bash
python3 task3_cv.py
```

The script prints the answer and writes:

```text
answers.txt
```

## Tests

I added numpy-only unit tests for the solver:

```bash
python3 -B -m unittest test_task3_cv.py
```

These tests check the solution module imports, projection reprojection error, valid `K/R` decomposition, red-point triangulation, and required answer sections.

## Results

Red feature point XYZ in world ENU metres:

```text
[0.993713, 2.990969, 0.470963]
```

Camera #1 intrinsic matrix:

```text
[1384.092916, 1.955734, 627.497035]
[-0.000000, 1390.367473, 367.395058]
[-0.000000, -0.000000, 1.000000]
```

Camera #1 rotation matrix:

```text
[0.999970, -0.000375, 0.007692]
[0.000434, 0.999970, -0.007744]
[-0.007689, 0.007747, 0.999940]
```

Camera #1 position in world ENU metres:

```text
[-1.015553, 0.977322, -9.918187]
```

Camera #2 intrinsic matrix:

```text
[928.181300, 0.955467, 640.338932]
[-0.000000, 928.096173, 356.915589]
[-0.000000, -0.000000, 1.000000]
```

Camera #2 rotation matrix:

```text
[0.707866, -0.706345, -0.001247]
[0.706346, 0.707867, 0.000198]
[0.000743, -0.001021, 0.999999]
```

Camera #2 position in world ENU metres:

```text
[1.395927, -0.016399, -9.874015]
```

## Verification

The reprojection checks are comfortably below one pixel:

- camera #1 max blue reprojection error: `0.082956 px`
- camera #2 max blue reprojection error: `0.330400 px`
- camera #1 red reprojection error: `0.251117 px`
- camera #2 red reprojection error: `0.376869 px`

The rotation checks are also clean:

- camera #1 `det(R) = 1.000000000000`
- camera #2 `det(R) = 1.000000000000`
- camera #1 orthogonality error: `2.221e-16`
- camera #2 orthogonality error: `9.774e-16`

## Maintainability Notes

- `WORLD_POINTS`, `IMAGE_POINTS`, and `RED_PIXELS` hold the PDF inputs.
- `estimate_projection_matrix()` implements normalized DLT.
- `rq_decomposition()` and `decompose_projection_matrix()` recover `K`, `R`, and camera center without OpenCV.
- `triangulate_point()` solves the red point with a linear SVD system.
- `test_task3_cv.py` checks import restrictions and numerical tolerances.

## LLM Use

I used LLM tools, including OpenAI Codex/ChatGPT-style assistance, under my direction while completing this task. I provided the geometry constraints, implementation direction, review feedback, and verification expectations. I reviewed plans and code changes frequently, gave corrections where needed, and approved the direction before moving on. I reviewed and verified the submitted code, numeric answer, and documentation before packaging.
