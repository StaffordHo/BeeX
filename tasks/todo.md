# BeeX Assignment Todo

## Workflow Checkpoint

- [x] Read the three assignment briefs and local starter files.
- [x] Use parallel explorers for one-pass folder analysis.
- [x] Create this plan before implementation.
- [x] Confirm plan with user before starting implementation.
- [ ] Implement tasks one by one, starting with Task 1. Task 1 implemented locally; ROS Noetic runtime verification pending.
- [ ] After each task, update the review section with commands run, outputs, and remaining risks.

## Task 1: Mission Planning

Goal: Build a ROS 1 Noetic/C++17 solution with two packages:

- `driver`: simulator TCP/UDP/serial interface, `/odom` publisher, `/fg_readings` publisher, `/command` service.
- `mission`: high-level state machine that locates the pipeline, centers on it, finds the southern start, and tracks north to the end.

Known simulator details:

- TCP server: `127.0.0.1:8091`.
- Handshake: send `Q`, expect `200OK!/dev/pts/<X>,127.0.0.1:<Y>`.
- FG serial format: `%.4fV==\n`.
- Odom UDP format: `<%.3fN,%.3fE>\n`, with Easting mapped to `Point.x` and Northing to `Point.y`.
- Motion commands: `W/S/A/D`, each command moves 0.05 m.
- Simulator loop is 10 Hz; only the last command in a 100 ms window is executed.
- Local simulator currently sends odom to UDP port `8015`, but the driver must parse the handshake port.

Implementation checklist:

- [x] Create ROS workspace/package layout inside the Task 1 folder without modifying the simulator source unless required.
- [x] Define a simple command service shared by `driver` and `mission`.
- [x] Implement robust parser helpers for TCP response, FG serial lines, and odometry datagrams.
- [x] Implement the driver node:
  - [x] TCP connect and handshake.
  - [x] Serial open/read/publish loop.
  - [x] UDP bind/read/publish loop.
  - [x] `/command` service that maps requests to single-character TCP commands.
  - [x] Command pacing compatible with 10 Hz simulator execution.
- [x] Implement the mission node:
  - [x] Wait for valid odom and FG readings.
  - [x] Search near the origin for a voltage signal.
  - [x] Center laterally using FG gradient behavior.
  - [x] Move to the southern endpoint.
  - [x] Track north with lateral correction and reacquisition behavior.
  - [x] Stop after reaching the northern terminus or mission timeout.
- [x] Add Task 1 README with strategy, build/run instructions, and LLM-use note.

Verification checklist:

- [x] Build simulator with `bash compile.sh`.
- [x] Smoke-test simulator TCP handshake.
- [ ] Build ROS packages with `catkin_make`. Not available in this local environment.
- [x] Unit/smoke-test parser helpers.
- [ ] Run driver against simulator and verify `/odom`, `/fg_readings`, and `/command`.
- [ ] Run mission against simulator and record whether it completes within constraints.

## Task 2: Machine Learning

Goal: Improve the provided semantic segmentation training code while keeping the required model type, and produce training code, loss curves, dataset insights, and per-class IoU/F1 for `member` and `anode`.

Important local findings:

- PDF requirement: train semantic segmentation for `member` and `anode`, treat all other object types as background, keep the provided model type, include dataset insights/challenges/changes, complete training code, loss curves, and separate IoU/F1 for `member` and `anode`.
- PDF disclaimer: do not waste time/electricity on excessive training; fundamentals, correctness, reproducibility, and analysis matter more than a heavily trained model.
- Dataset archive is `interview_dataset.zip`, COCO 1.1-style, with annotations at `interview_dataset/annotations/instances_default.json`.
- Zip integrity check passed: `unzip -tq interview_dataset.zip`.
- Dataset stats from the COCO JSON:
  - `1111` images, `1363` annotations, `115` images with no target annotation.
  - Categories: `anode` id `1`, `member` id `2`.
  - Annotation counts: `anode=112`, `member=1251`.
  - Image-level category combos: `884` member-only, `112` anode+member, `115` empty/background-only.
  - Image IDs are contiguous from `1` to `1111`, but parser code should still avoid assuming this.
- Starter script has correctness bugs in the train/validation dataset wiring: `ImageSegDataset(X_train, X_test, ...)` and `ImageSegDataset(y_train, y_test, ...)` mix images and masks incorrectly.
- Starter center-crops wide sonar images to a square, which can discard useful side context; prefer aspect-preserving letterbox resize for both image and mask.
- Starter masks are resized with cubic interpolation and stored as multi-channel floats; semantic segmentation should use class-index masks and nearest-neighbor mask resizing.
- Background class needs to be represented explicitly and consistently.
- Starter loss/model setup is mismatched: `BCELoss` with softmax output and one-hot masks is weaker than logits + `CrossEntropyLoss`/Dice for multi-class segmentation.
- Initial local shell lacked the ML dependencies used by the starter script, including `numpy`, `torch`, `cv2`, `matplotlib`, and `segmentation-models-pytorch`.

Implementation checklist:

- [x] Add reproducible extraction/auto-detection for `interview_dataset.zip` without requiring manual unzip.
- [x] Rewrite COCO parsing around image IDs, file names, polygon segmentations, and class-id mapping instead of assuming `image_id - 1` always indexes the image list.
- [x] Produce class-index masks: `0=background`, `1=anode`, `2=member`, with later annotations overriding only intentionally and no float one-hot masks.
- [x] Use aspect-preserving letterbox resize for image/mask pairs; resize masks with nearest-neighbor and images with image interpolation.
- [x] Fix train/validation split and dataset construction; stratify by image category combo where practical so rare `anode` examples are represented in validation.
- [x] Normalize images appropriately for the ImageNet encoder.
- [x] Keep `UnetPlusPlus('resnet34', ...)` as the model family, using logits for stable multi-class losses.
- [x] Add deterministic seed handling and CLI arguments for dataset path, epochs, batch size, learning rate, image size, output directory, workers, device, and smoke-test mode.
- [x] Add weighted cross entropy plus soft Dice loss to address imbalance, with weights derived from training masks or conservative defaults.
- [x] Track and save train/validation losses per epoch in the training script.
- [x] Compute per-class IoU and F1 for `anode` and `member`, excluding background from the reported assignment metrics.
- [x] Save `loss_curves.png`, `metrics.json`, `metrics.txt`, and a best/latest checkpoint where runtime allows.
- [x] Write a concise Task 2 solution README/results note with dataset insights, changes made, run instructions, and LLM-use note.
- [x] Update dependency files with `opencv-python-headless` and `segmentation-models-pytorch`, while keeping PyTorch in an explicit CPU/GPU install step.

Verification checklist:

- [x] Verify zip integrity.
- [x] Run no-dependency syntax checks where possible.
- [x] After dependency install, run import checks for `torch`, `cv2`, `segmentation_models_pytorch`, `matplotlib`, and `numpy`.
- [x] Run a fast smoke train/eval pass, preferably `--smoke --epochs 1 --max-samples <small>`.
- [x] Run a modest training pass only if dependencies and compute allow; avoid excessive training per the PDF disclaimer.
- [x] Confirm artifacts exist: loss curve, metrics, trained checkpoint or documented no-checkpoint choice.

### Task 2 Hidden-Test Augmentation Plan

Goal: improve robustness for the hidden sonar test set without changing the required model family or making validation/test metrics optimistic.

- [x] Add optional train-only augmentation behind a CLI flag.
- [x] Keep geometry conservative: horizontal flips plus small rotation/scale jitter, with nearest-neighbor mask warping.
- [x] Keep sonar photometric transforms conservative: brightness/contrast/gamma jitter, light noise, and mild blur.
- [x] Prevent augmentation from erasing tiny anode masks during geometric transforms.
- [x] Leave validation and test datasets unaugmented for clean metrics.
- [x] Document when I would use augmentation and how to run the augmented experiment.
- [x] Verify syntax and an augmentation-enabled smoke run.

## Task 3: Computer Vision

Goal: Solve the two-camera geometry problem and report red-point XYZ, both camera intrinsics, both rotation matrices, and both camera positions in world ENU metres.

Constraints:

- Python solution may use only `numpy`.
- C++17 solution may use Eigen/OpenCV only for vector/matrix math.
- No OpenCV camera calibration or 3D reconstruction functions.

PDF facts extracted from `03 Computer Vision/README.pdf`:

- Both images are `1280 x 720 px`; pixel origin is OpenCV-style top-left.
- Six blue feature points have known ENU world coordinates and pixel locations in both cameras.
- The red feature point has pixel observations `[906, 626]` in camera #1 and `[423, 523]` in camera #2, but unknown world XYZ.
- Required outputs:
  - red point XYZ in metres in the ENU world frame
  - intrinsic matrix for each camera
  - rotation matrix for each camera
  - camera position XYZ for each camera in metres in the ENU world frame
- Tolerances: `15 cm` L2 position tolerance and `20 px` focal-length tolerance.

Preferred implementation:

- [x] Write a small Python 3 script using only `numpy`.
- [x] Hard-code the six blue world points, both blue image observations, and both red image observations from the PDF.
- [x] Estimate each `3 x 4` projection matrix with a hand-written DLT/SVD solve from the six 3D-to-2D correspondences.
- [x] Decompose each projection matrix into `K`, `R`, and camera center with a hand-written RQ decomposition using `numpy.linalg.qr`.
- [x] Normalize/sign-correct `K` and `R` so focal lengths are positive and `det(R) ~= +1`.
- [x] Triangulate the red point with a hand-written linear least-squares/SVD solve from both camera projection matrices and red pixel observations.
- [x] Reproject all blue points and the red point to validate residuals.
- [x] Write answers to a text file and include run instructions.

Planned files:

- `03 Computer Vision/task3_cv.py`: numpy-only implementation.
- `03 Computer Vision/answers.txt`: generated numeric answer and verification summary.
- `03 Computer Vision/SOLUTION_README.md`: first-person explanation, run instructions, constraints, and LLM-use note.

Verification checklist:

- [x] Run the script with only `numpy`.
- [x] Check maximum blue-point reprojection error per camera is comfortably under the PDF's `20 px` focal tolerance context.
- [x] Check red reprojection error is near zero for both cameras.
- [x] Check `R.T @ R ~= I` and `det(R) ~= +1` for both cameras.
- [x] Check `K[2,2] == 1`, focal lengths are positive, and skew/principal point are finite.
- [x] Confirm all positions are output in metres in the ENU world frame.
- [x] Re-run the script after writing documentation and record the final output in the review log.

## Documentation Style

- [x] Record the user's preference for first-person singular submission documentation in `tasks/lessons.md`.
- [x] Rewrite Task 1 and Task 2 solution READMEs to sound like they came from one candidate.
- [x] Verify submission-facing Markdown no longer uses team/passive wording in the main explanatory sections.

## Test Installation Plan

Goal: add lightweight tests that prove the implemented code paths behave as intended without violating assignment constraints.

Task 1 tests:

- [x] Keep the existing C++ parser test as the main non-ROS unit test.
- [x] Add a small wrapper script that compiles/runs parser tests and checks Docker helper syntax without requiring ROS on the host.
- [x] Avoid adding tests that require ROS Noetic on Ubuntu 22.04 outside the provided Docker workflow.

Task 2 tests:

- [x] Add Python `unittest` coverage for pure/data-pipeline behavior instead of training-heavy behavior.
- [x] Test polygon rasterization class priority, letterbox mask interpolation, stratified splitting, loss edge cases, and augmentation mask-label safety.
- [x] Use the existing ML runtime dependencies only; do not add new test dependencies such as `pytest`.

Task 3 tests:

- [x] Add Python `unittest` coverage using only standard library plus `numpy`.
- [x] Test projection estimation, camera decomposition, triangulation, reprojection tolerances, and import restrictions.
- [x] Avoid OpenCV entirely and avoid any camera calibration / 3D reconstruction helper.

Verification:

- [x] Run Task 1 local wrapper test.
- [x] Run Task 2 unit tests from the existing `.venv` if available.
- [x] Run Task 3 unit tests with `python3 -B`.
- [x] Record commands, outputs, and any environment caveats in the review log.

## Submission Preparation Plan

Goal: prepare a clean submission folder with the requested files only, while keeping generated caches/build outputs out of the package.

- [x] Make the Task 3 documentation explicit that both cameras share `1280 x 720 px` image size but have independently estimated intrinsic matrices.
- [x] Create a clean root-level submission folder.
- [x] Include Task 1 source, ROS package source, Docker helper, local tests, and solution README.
- [x] Include Task 2 training code, requirements, unit tests, solution README, and the selected non-augmented `outputs_cpu_anode` artifacts.
- [x] Include Task 3 numpy-only code, answer file, tests, and solution README.
- [x] Add a short root submission README explaining contents and the Task 2 non-augmented checkpoint choice.
- [x] Verify the submission folder has no `__pycache__`, `.venv`, ROS `build/devel`, or unrelated smoke-output directories.
- [x] Run lightweight tests from the submission folder where feasible.
- [x] Record final submission path and verification results in the review log.

## Review Log

Initial planning review:

- Task 1 is the first implementation target because it has the broadest integration risk.
- Task 2 needs correctness fixes before meaningful training improvements.
- Task 3 has no starter code and can be solved independently after Task 2 or in parallel later.
- No implementation has started yet. Awaiting user confirmation on this plan.

Task 1 implementation review:

- Added ROS workspace under `01 Mission Planning/task/ros_ws`.
- Added `driver` package with `driver_node`, `Command.srv`, protocol parser helpers, and parser smoke test.
- Added `mission` package with a finite-state planner: search, center, acquire south endpoint, track north, reacquire, stop on sustained loss or timeout.
- Added `SOLUTION_README.md` with strategy, ROS interface, build/run instructions, verification status, and LLM-use note.
- Verification passed:
  - `bash compile.sh`
  - `g++ -std=c++17 -Wall -Wextra -Wpedantic -I ... test_protocol_parsers.cpp -o /tmp/task1_protocol_parser_test`
  - `/tmp/task1_protocol_parser_test`
  - XML parse check for both package manifests
  - Elevated simulator handshake smoke test returned `200OK!/dev/pts/2,127.0.0.1:8015`
- Verification not run locally:
  - `catkin_make`, because `catkin_make` and `roscore` are not installed on this machine.
  - Full driver/mission ROS runtime, because ROS Noetic is not available in this environment.

Task 1 comment/style follow-up:

- [x] Reference simulator protocol details from `bx_grid_world.cpp` in code comments.
- [x] Align driver/mission source organization with the simulator's sectioned, responsibility-oriented style.
- [x] Re-run standalone parser test after comment/style changes.
- Added comments for `GameWorld::handleTcpClient()`, `GameWorld::writeOdomToUdp()`, `GameWorld::writeSensorToSerial()`, and `GameWorld::spinOnce()` protocol behavior.
- Re-ran `g++ -std=c++17 -Wall -Wextra -Wpedantic ... test_protocol_parsers.cpp` and `/tmp/task1_protocol_parser_test`; parser test passed.

Task 1 Docker workflow:

- [x] Add Dockerfile for Ubuntu 20.04 + ROS Noetic.
- [x] Add Docker Compose service that keeps simulator and ROS nodes in one shared container namespace.
- [x] Add helper script for build/start/simulator/driver/mission/parser-test commands.
- [x] Build Docker image locally. User log shows `Image beex-task1-noetic:latest Built`.
- [x] Run `catkin_make` inside Docker. User log shows `driver_node` and `mission_node` built successfully.
- [ ] Run full simulator/driver/mission flow inside Docker.
- Static verification passed:
  - `bash -n docker/task1_docker.sh`
  - `docker compose -f docker-compose.yml config`
- Docker build attempted but did not run because the current user cannot access the Docker daemon and passwordless `sudo docker` is unavailable.
- User retried after fixing Docker group access; build then failed during apt install because `libopencv-dev` pulled a large dependency set and Docker reported insufficient `/var/cache/apt/archives` space.
- Slimmed Docker image by replacing `libopencv-dev` with `libopencv-core-dev`, `libopencv-imgproc-dev`, and `libopencv-highgui-dev`.
- Updated Docker `sim-build` to compile `bx_grid_world.cpp` with explicit OpenCV libraries instead of requiring the full `opencv4.pc` meta-package.
- User log confirmed the slim Docker image builds and `catkin_make` completes.
- Docker `sim-build` then exposed that `bx_grid_world.cpp` included OpenCV's umbrella `opencv.hpp`, which requires modules not installed in the slim image.
- Initially replaced the umbrella OpenCV include in `bx_grid_world.cpp` with module-specific headers for local Docker compatibility, then restored the original provided simulator source after review.
- Final Docker compatibility approach keeps `bx_grid_world.cpp` unmodified and uses a Docker-only OpenCV include shim at `docker/include/opencv2/opencv.hpp`.
- Re-ran static checks after Docker slimming:
  - `bash -n docker/task1_docker.sh`
  - `docker compose -f docker-compose.yml config`
- Re-ran compile checks after OpenCV include change:
  - `bash compile.sh`
  - `g++ -std=c++17 bx_grid_world.cpp -o /tmp/bx_grid_world_minimal_opencv -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_highgui -lutil`
  - `/tmp/task1_protocol_parser_test`
- User runtime logs showed driver connected, then repeatedly reported `Serial connection lost: closed`; mission timed out waiting for `/odom` and `/fg_readings`.
- Fixed driver serial handling so zero-byte reads with `VMIN=0/VTIME=0` are treated as "no data ready" instead of device closure.
- Added a UDP FD guard after disconnects and changed Docker helper noninteractive exec calls to `docker compose exec -T` to avoid allocating extra pseudo-terminals.
- User then hit `./bx_grid_world: error while loading shared libraries: libopencv_highgui.so.4.5d`, caused by running a host-built Ubuntu 22.04/OpenCV binary inside the Ubuntu 20.04 Docker container.
- Updated Docker helper so `sim-build` and `simulator` use a shared container compile command, and `simulator` auto-rebuilds if the binary is missing or `ldd` reports unresolved shared libraries.
- User observed HAUV finding the pipeline near `+2.8N +0.5E` and asked how to speed up mission completion under the 15-minute requirement.
- Sped up mission tracking after acquisition:
  - Increased normal north tracking step from `0.25 m` to `0.50 m`.
  - Reduced normal center scan radius from `1.4 m` to `0.75 m`.
  - Added fast-centering: if the predicted centerline point has FG >= `0.65`, accept it without a full scan.
  - Southern endpoint acquisition now uses the previous centerline estimate and a smaller `0.85 m` local scan instead of repeated wide scans.
  - Reduced command feedback wait from `0.13 s` to `0.11 s`, still above the simulator's 100 ms command window.

Task 1 README compliance check:

- [x] Re-read `01 Mission Planning/task/README.md` and map each requirement to the current implementation.
- [x] Inspect `driver` package interfaces and simulator protocol handling.
- [x] Inspect `mission` package separation, search/tracking strategy, and safety constraints.
- [x] Check documentation and dependency/simulator-modification constraints.
- [x] Run available local build/parser/script checks.
- [x] Record final compliance findings and remaining runtime verification gaps.

Task 1 README compliance findings:

- Overall: Current source design adheres to the Task 1 README architecture and protocol requirements.
- `driver` package compliance:
  - Uses ROS 1/catkin and C++17.
  - Connects to `127.0.0.1:8091`, sends `Q`, parses the serial device and UDP port from the handshake.
  - Opens the handshake-provided serial device, parses `%.4fV==`-style readings, and publishes `/fg_readings` as `std_msgs/Float32`.
  - Binds the handshake-provided UDP port, parses `<%.3fN,%.3fE>`, maps Easting to `Point.x`, Northing to `Point.y`, and publishes `/odom` as `geometry_msgs/Point`.
  - Provides `/command` as `driver/Command` and maps high-level aliases to `W/A/S/D`, with 10 Hz-safe pacing.
- `mission` package compliance:
  - Subscribes only to `/odom` and `/fg_readings`, and moves only through `/command`.
  - Uses a state machine for search, centerline fitting, southern endpoint acquisition, northward tracking, and reacquisition.
  - Bounds mission movement by clamping search/reacquire northing and stopping at `max_northing_m = 45.0`, leaving margin inside the 100 m x 100 m world.
  - Runtime log evidence from the previous run showed simulator success in `548833 ms`, within the 15-minute limit, before the latest speed optimization.
- Documentation compliance:
  - `SOLUTION_README.md` explains strategy, build/run steps, Docker Noetic workflow, ROS interfaces, local verification, and LLM use.
  - Docker support is an added local-development aid for Ubuntu 22.04/ROS2 Humble; the ROS packages remain Noetic/catkin-compatible.
- Dependency/simulator compliance:
  - ROS package code uses standard/POSIX C++ and ROS packages available from Ubuntu 20.04 repositories.
  - The final submission keeps the original BeeX `bx_grid_world.cpp` source unmodified. Docker-only OpenCV compatibility is handled outside the simulator source.
- Verification performed for this compliance pass:
  - `bash compile.sh` passed.
  - `g++ -std=c++17 -Wall -Wextra -Wpedantic -I ros_ws/src/driver/include ros_ws/src/driver/test/test_protocol_parsers.cpp -o /tmp/task1_protocol_parser_test` passed.
  - `/tmp/task1_protocol_parser_test` passed.
  - `bash -n docker/task1_docker.sh` passed.
  - `docker compose -f docker-compose.yml config` passed.
  - `catkin_make` is not installed in this host shell, but the user's Docker ROS build log showed `driver_node` and `mission_node` built successfully.
- Remaining risks before final submission:
  - Tracking accuracy is strategy-driven but not explicitly measured/enforced in code. The planner recenters from FG maxima and previously succeeded, but it can scan `0.75 m` laterally during normal tracking and up to `4.0 m` during reacquisition; final verification should confirm the simulator accepts this under the 0.5 m tracking rule.
  - X/easting movement is not explicitly clamped, although current search starts within +/-5 m and scans +/-5 m from start, and tracking follows predicted centerline samples. Add a guard if final runtime logs show any boundary risk.
  - Re-run the full Docker mission after the latest speed optimization and record the final simulator elapsed time.
  - Consider making the solution documentation the submission `README.md`, because the current task-level `README.md` is still the assignment prompt and the solution write-up is in `SOLUTION_README.md`.
  - Do not include generated artifacts in the submission zip: `bx_grid_world`, Docker images, `ros_ws/build`, or `ros_ws/devel`.

Task 1 tracking regression fix:

- [x] Inspect the latest runtime logs showing repeated loss near `N=10.80`.
- [x] Identify that the robot had reached/passed the northern terminus and was looping in beyond-end reacquisition.
- [x] Patch the mission tracker with conservative movement clamps and northern-end finish behavior.
- [x] Re-run local compile/parser/static checks.
- [x] Document what changed and what runtime verification remains.

Task 1 endpoint-loop fix review:

- User clarified that the robot found the north-south pipeline end, then kept searching beyond it.
- Updated `mission_node.cpp`:
  - Added conservative `+/-45 m` X/Y target clamps to keep motion inside the operating area.
  - Reduced stale-sample influence during center scans by using the short sample window when choosing the best FG point.
  - Added `finishNorthernEndpoint()` so sustained northward loss after valid tracking performs a short fine endpoint pass instead of repeatedly searching beyond the pipe.
  - If reacquisition after a northward loss returns to the last row, the mission now treats that as endpoint evidence and finishes.
- Updated `SOLUTION_README.md` to document endpoint-finish behavior and motion clamping.
- Updated `tasks/lessons.md` with the terminus-vs-mid-track-loss lesson.
- Verification passed:
  - `bash compile.sh`
  - `g++ -std=c++17 -Wall -Wextra -Wpedantic -I ros_ws/src/driver/include ros_ws/src/driver/test/test_protocol_parsers.cpp -o /tmp/task1_protocol_parser_test`
  - `/tmp/task1_protocol_parser_test`
  - `bash -n docker/task1_docker.sh`
  - `docker compose -f docker-compose.yml config`
- Verification blocked in this agent shell:
  - `./docker/task1_docker.sh ros-build`, because this shell cannot access the Docker daemon.
- User should run `./docker/task1_docker.sh ros-build` before the next mission attempt.

Task 2 implementation review:

- Rewrote `02 Machine Learning/task/task1_ml.py` into a reproducible training/evaluation script.
- Added:
  - Safe auto-extraction for `interview_dataset.zip`.
  - COCO image-id based indexing.
  - Polygon mask rasterization for `0=background`, `1=anode`, `2=member`.
  - Aspect-preserving letterbox resize.
  - Nearest-neighbor mask resizing.
  - Stratified train/validation split by background/member/anode presence.
  - Balanced smoke subset selection so rare `anode` examples are included when possible.
  - ImageNet normalization for the ResNet34 encoder.
  - Required `UnetPlusPlus('resnet34', classes=3)` model type with logits.
  - Weighted cross entropy plus soft Dice loss.
  - Per-class IoU/F1 for `anode` and `member`.
  - `loss_curves.png`, `metrics.json`, `metrics.txt`, and `best_model.pt` artifact writing.
- Updated dependency files:
  - `requirements.txt` contains the non-Torch runtime dependencies, including `opencv-python-headless` and `segmentation-models-pytorch`.
  - `requirements-cpu.txt` installs CPU-only PyTorch/TorchVision first, then the rest of `requirements.txt`.
- Added `02 Machine Learning/task/SOLUTION_README.md` with dataset insights, changes, run instructions, metrics artifact descriptions, and LLM-use note.
- Verification passed:
  - `unzip -tq interview_dataset.zip`
  - `python3 -m py_compile task1_ml.py`
  - AST parse check for `task1_ml.py`
  - Source scan confirmed key implementation pieces: `UnetPlusPlus`, `CrossEntropyLoss`, `soft_dice_loss`, `INTER_NEAREST`, `metrics.json`, and `loss_curves`.
- Initial import/smoke verification was blocked before dependencies were installed; the later CPU install/smoke verification section records the successful run.

Task 2 dependency compatibility follow-up:

- User attempted `pip install -r requirements.txt` in Python `3.13.13`; `torch==2.3.0` was unavailable for that Python version, so `cv2` was never installed and the smoke run failed on `ModuleNotFoundError: No module named 'cv2'`.
- First update moved old exact PyTorch pins to broad compatibility ranges, but the user then hit a second issue: Python `3.13` resolved a CUDA-enabled PyTorch wheel stack and exhausted disk space.
- Corrected dependency approach:
  - Removed `torch` and `torchvision` from `requirements.txt`.
  - Added `requirements-cpu.txt` with explicit CPU-only PyTorch/TorchVision wheels for smoke tests.
  - Removed the `scikit-learn` dependency by replacing train/validation splitting with local deterministic stratified split logic.
  - Made `matplotlib` a lazy import inside `save_loss_curve()` so the script can still import if plotting dependencies are temporarily unavailable.
  - Updated `SOLUTION_README.md` with CPU-safe install, CUDA cleanup, and separate GPU install guidance.
- Verification passed after the requirements update:
  - `python3 --version` returned `Python 3.13.13`.
  - `python3 -m py_compile task1_ml.py`.
- Cleaned generated `__pycache__` artifacts after syntax verification.

Task 2 CPU install/smoke verification:

- User reran the CPU-safe install path in a fresh `.venv`; cleanup removed about `8085.6 MB` of cached/partial CUDA packages from the earlier broad PyTorch install.
- `pip install -r requirements-cpu.txt` succeeded with:
  - `torch==2.11.0+cpu`
  - `torchvision==0.26.0+cpu`
  - `opencv-python-headless==4.13.0.92`
  - `segmentation-models-pytorch==0.4.0`
  - `matplotlib==3.11.0`
  - `numpy==2.5.0`
- User smoke command completed:
  - `python task1_ml.py --smoke --epochs 1 --batch-size 2 --image-size 224 --no-pretrained --output-dir outputs_smoke`
- Smoke run details:
  - Dataset extracted successfully from `interview_dataset.zip`.
  - Dataset summary printed `1111` images, `1363` annotations, `member=1251`, `anode=112`.
  - Smoke split used `26` training images and `6` validation images.
  - One CPU epoch completed with train loss `1.8734` and validation loss `1.5796`.
  - Smoke metrics were low, as expected for a one-epoch/no-pretrained correctness run: `anode IoU=0.0002096`, `member IoU=0.0069034`.
- Artifact verification passed; `outputs_smoke` contains:
  - `best_model.pt`
  - `loss_curves.png`
  - `metrics.json`
  - `metrics.txt`
- Import verification in this agent shell passed:
  - `torch 2.11.0+cpu`, CUDA unavailable as expected for CPU wheels.
  - `cv2 4.13.0`, `segmentation_models_pytorch 0.4.0`, `matplotlib 3.11.0`, `numpy 2.5.0`.
  - Matplotlib emitted a cache-directory warning in the sandboxed agent shell only; it did not block imports or the smoke run.

Task 2 train/validation performance review:

- User ran a modest CPU training pass:
  - `python task1_ml.py --epochs 3 --batch-size 2 --image-size 192 --no-pretrained --output-dir outputs_cpu`
- Training behavior:
  - Train loss improved from `0.6419` to `0.3598`.
  - Validation loss improved from `0.4527` to `0.2707`.
  - Member segmentation learned substantially: final validation `member IoU=0.6167`, `member F1=0.7629`.
  - Anode segmentation failed: final validation `anode IoU=0.0`, `anode F1=0.0`.
- Split composition check:
  - Train images: `889`, with `707` member-only, `92` background-only, `90` anode+member.
  - Validation images: `222`, with `177` member-only, `23` background-only, `22` anode+member.
  - The train/validation split is image-level stratified correctly; validation does contain anode examples.
- Pixel imbalance check:
  - Original train mask pixels: background `99.4831%`, anode `0.0057%`, member `0.5112%`.
  - Original validation mask pixels: background `99.4532%`, anode `0.0064%`, member `0.5404%`.
  - After resizing validation images to `192 x 192`, only `346` anode pixels remained across all `222` validation images.
- Checkpoint prediction check:
  - Validation target pixels after resize: background `8157881`, anode `346`, member `25581`.
  - Validation predicted pixels: background `8157876`, anode `0`, member `25932`.
  - Confusion matrix confirmed the model never selected the anode class.
- Conclusion:
  - A train/validation/test split is still useful for cleaner final reporting, but it will not fix the current anode failure by itself.
  - Before final reporting, prioritize anode recall with higher resolution, pretrained encoder weights when available, stronger rare-class sampling/patching, or a loss/sampling change that makes tiny anode pixels matter.

Task 2 anode-recall improvement plan:

- [x] Add a deterministic train/validation/test split for cleaner reporting while preserving image-level stratification.
- [x] Replace plain shuffled training with a weighted sampler that oversamples images containing anodes.
- [x] Add an anode-focused loss component so class `1` false negatives contribute directly to optimization.
- [x] Add prediction-count diagnostics to metrics output so zero-anode predictions are easy to detect.
- [x] Update the Task 2 solution README in first-person style with the anode imbalance findings and new run guidance.
- [x] Verify with syntax/import checks and a fast smoke run before recommending a longer CPU run.

Task 2 anode-recall implementation review:

- Implemented train/validation/test splitting with defaults `70/15/15` via `--val-size 0.15` and `--test-size 0.15`.
- Updated Dice loss so absent foreground classes are skipped. This avoids rewarding anode suppression in batches with no anode pixels.
- Added an optional anode-specific binary Dice term that only activates for batches containing anode pixels.
- Added weighted training sampling:
  - default `--anode-sample-weight 6.0`
  - default `--member-sample-weight 1.0`
  - default `--background-sample-weight 0.5`
- Added moderate anode class-weight boosting:
  - default `--anode-class-weight-multiplier 3.0`
  - default `--anode-dice-weight 0.25`
- Added split summaries and target/predicted pixel counts to `metrics.json`; `metrics.txt` now separates validation and test metrics.
- Corrected metrics reporting so `validation_metrics` corresponds to the best validation checkpoint, while `last_validation_metrics` preserves the final epoch.
- Verification passed:
  - `python3 -m py_compile task1_ml.py`
  - smoke run: `python task1_ml.py --smoke --batch-size 2 --image-size 64 --no-pretrained --output-dir outputs_anode_smoke_metricsfix`
  - balanced subset run: `python task1_ml.py --epochs 3 --batch-size 2 --image-size 192 --no-pretrained --max-samples 256 --num-workers 0 --output-dir outputs_anode_subset`
- The first subset attempt with default `num_workers=2` failed in the sandbox with a PyTorch multiprocessing `PermissionError`; rerunning with `--num-workers 0` completed successfully. This is an environment restriction rather than a training-code failure.
- Balanced subset result:
  - Split: `178` train, `39` validation, `39` test; each split preserved anode/member/background image presence.
  - Training loss improved from `1.4128` to `0.7521`.
  - Validation metrics: `anode IoU=0.0061`, `anode F1=0.0120`, `member IoU=0.1253`, `member F1=0.2228`.
  - Test metrics: `anode IoU=0.0068`, `anode F1=0.0134`, `member IoU=0.1201`, `member F1=0.2145`.
  - Test predicted pixels: background `1435563`, anode `1510`, member `623`.
- Conclusion:
  - The zero-anode collapse is fixed on the balanced subset.
  - At this stage, a full-dataset run was still needed for final quality.

Task 2 full CPU anode run:

- User ran the full CPU command:
  - `python task1_ml.py --epochs 5 --batch-size 2 --image-size 224 --no-pretrained --num-workers 0 --output-dir outputs_cpu_anode`
- Split summary:
  - Train: `777` images, with `618` member-only, `78` anode+member, `81` background-only.
  - Validation: `167` images, with `133` member-only, `17` anode+member, `17` background-only.
  - Test: `167` images, with `133` member-only, `17` anode+member, `17` background-only.
- Training behavior:
  - Train loss improved from `0.8285` to `0.1666`.
  - Validation loss improved from `0.3821` to `0.1738`.
  - Best validation checkpoint was epoch `4`, selected by mean target IoU.
- Best validation metrics:
  - `anode IoU=0.3849`, `anode F1=0.5558`.
  - `member IoU=0.6040`, `member F1=0.7531`.
- Held-out test metrics:
  - `anode IoU=0.5570`, `anode F1=0.7155`.
  - `member IoU=0.5694`, `member F1=0.7256`.
- Pixel-count diagnostics:
  - Validation target/predicted anode pixels: `212` target, `209` predicted at best checkpoint.
  - Test target/predicted anode pixels: `388` target, `336` predicted.
- Artifacts verified in `outputs_cpu_anode`:
  - `best_model.pt`
  - `loss_curves.png`
  - `metrics.json`
  - `metrics.txt`
- Conclusion:
  - The anode learning issue is resolved on the full CPU run.
  - Current final Task 2 metrics should use the held-out test split from `outputs_cpu_anode`.

Task 2 hidden-test augmentation implementation review:

- Added optional train-only augmentation behind `--augment`; validation and test datasets remain unaugmented.
- Added conservative shared image/mask geometry:
  - horizontal flip probability `0.5`
  - affine probability `0.35`
  - max rotation `3` degrees
  - scale jitter `0.04`
  - translation jitter `0.03`
- Added image-only sonar variation:
  - brightness/contrast/gamma jitter
  - light Gaussian noise
  - mild `3x3` Gaussian blur
- Preserved semantic mask correctness:
  - masks stay class-index `uint8`
  - affine mask warps use `cv2.INTER_NEAREST`
  - geometry is skipped if it erases all anode pixels or drops too much foreground
- Added deterministic DataLoader worker seeding for Python and NumPy random streams.
- Added augmentation config to `metrics.json` and printed it at run start for experiment traceability.
- Updated `SOLUTION_README.md` with hidden-test augmentation rationale and the augmented run command:
  - `python task1_ml.py --epochs 5 --batch-size 2 --image-size 224 --no-pretrained --num-workers 0 --augment --output-dir outputs_cpu_anode_aug`
- Verification passed:
  - `python -m py_compile task1_ml.py`
  - `python task1_ml.py --smoke --batch-size 2 --image-size 96 --no-pretrained --augment --num-workers 0 --output-dir outputs_aug_smoke`
- Augmented smoke run details:
  - Split: `20` train, `6` validation, `6` test.
  - Augmentation config was written to `outputs_aug_smoke/metrics.json`.
  - Artifacts were written: `best_model.pt`, `loss_curves.png`, `metrics.json`, `metrics.txt`.
  - Metrics are intentionally low because this was a one-epoch, no-pretrained correctness smoke run.
- Cleanup note:
  - `py_compile` generated `02 Machine Learning/task/__pycache__`; I left it in place rather than running a destructive cleanup command without explicit user approval.

Task 2 full augmented CPU run:

- User ran the full augmented CPU command:
  - `python task1_ml.py --epochs 5 --batch-size 2 --image-size 224 --no-pretrained --num-workers 0 --augment --output-dir outputs_cpu_anode_aug`
- Training behavior:
  - Train loss improved from `0.8366` to `0.2851`.
  - Validation loss improved from `0.5933` to `0.2001`.
  - Best validation checkpoint was epoch `5`, selected by mean target IoU.
  - Best validation mean target IoU was `0.4848`.
- Best validation metrics:
  - `anode IoU=0.3556`, `anode F1=0.5247`.
  - `member IoU=0.6140`, `member F1=0.7608`.
- Held-out test metrics:
  - `anode IoU=0.4775`, `anode F1=0.6464`.
  - `member IoU=0.5811`, `member F1=0.7351`.
- Pixel-count diagnostics:
  - Test target/predicted anode pixels: `388` target, `500` predicted.
- Artifacts verified in `outputs_cpu_anode_aug`:
  - `best_model.pt`
  - `loss_curves.png`
  - `metrics.json`
  - `metrics.txt`
- Comparison to non-augmented full run:
  - Non-augmented run has better held-out anode metrics: `anode IoU=0.5570`, `anode F1=0.7155`.
  - Augmented run has slightly better held-out member metrics: `member IoU=0.5811`, `member F1=0.7351`.
  - I would report `outputs_cpu_anode` as the strongest visible-split checkpoint and keep `outputs_cpu_anode_aug` as a hidden-test robustness experiment.
- User decision:
  - Submit the non-augmented `outputs_cpu_anode` model/results for Task 2.
  - Keep the augmented run in the write-up as a hidden-test robustness discussion, not as the primary submitted checkpoint.

Task 3 implementation review:

- Added `03 Computer Vision/task3_cv.py`.
- Added `03 Computer Vision/answers.txt`.
- Added `03 Computer Vision/SOLUTION_README.md`.
- Implementation details:
  - Python 3 with `numpy` only.
  - No OpenCV imports or camera calibration / 3D reconstruction helpers.
  - Normalized DLT estimates each `3 x 4` projection matrix.
  - Hand-written RQ decomposition recovers `K` and `R`.
  - Sign correction enforces positive focal lengths and `det(R)=+1`.
  - Linear SVD triangulation recovers the red feature point.
- Final red point XYZ in world ENU metres:
  - `[0.993713, 2.990969, 0.470963]`
- Camera #1:
  - Position: `[-1.015553, 0.977322, -9.918187]`
  - `fx=1384.092916`, `fy=1390.367473`
  - Max blue reprojection error: `0.082956 px`
  - Red reprojection error: `0.251117 px`
  - `det(R)=1.000000000000`
  - Rotation orthogonality error: `2.221e-16`
- Camera #2:
  - Position: `[1.395927, -0.016399, -9.874015]`
  - `fx=928.181300`, `fy=928.096173`
  - Max blue reprojection error: `0.330400 px`
  - Red reprojection error: `0.376869 px`
  - `det(R)=1.000000000000`
  - Rotation orthogonality error: `9.774e-16`
- Verification passed:
  - `python3 -B task3_cv.py`
  - Numeric assertion script checking reprojection errors, `det(R)`, `R.T @ R`, positive focal lengths, `K[2,2]`, and finite camera centers.
  - Import scan confirmed only standard library imports plus `numpy`.
  - First-person documentation scan only flagged the intended phrase `I used LLM assistance`.
- Cleanup note:
  - Python generated `03 Computer Vision/__pycache__` during verification imports; I left it in place rather than running a destructive cleanup command without explicit user approval.

Test installation review:

- Added Task 1 local wrapper:
  - `01 Mission Planning/task/tests/run_local_tests.sh`
  - Covers C++ protocol parser compile/run, Docker helper shell syntax, and Docker Compose config structure when `docker compose` is available.
  - Does not require host ROS Noetic.
- Added Task 2 unit tests:
  - `02 Machine Learning/task/test_task1_ml.py`
  - Covers polygon mask class priority, nearest-neighbor letterbox mask resizing, stratified splitting, absent-foreground Dice losses, IoU/F1 metrics, augmentation mask-label safety, and tiny-target geometry guards.
  - Uses `unittest` and the existing ML runtime stack; no new test dependency was added.
- Added Task 3 unit tests:
  - `03 Computer Vision/test_task3_cv.py`
  - Covers solution import restrictions, projection reprojection accuracy, valid `K/R` decomposition, red-point triangulation, and required report sections.
  - Uses standard library plus `numpy`; no OpenCV import.
- Updated solution READMEs with test commands for all three tasks.
- Verification passed:
  - Task 1: `bash tests/run_local_tests.sh`
    - Protocol parser test passed.
    - Docker helper syntax passed.
    - Docker Compose config structure passed.
  - Task 2: `.venv/bin/python -B -m unittest test_task1_ml.py`
    - `7` tests passed.
  - Task 3: `python3 -B -m unittest test_task3_cv.py`
    - `5` tests passed.

Submission preparation review:

- Prepared clean submission folder:
  - `/home/stafford99/BeeX/submission`
- Prepared zip archive:
  - `/home/stafford99/BeeX/beex_submission.zip`
  - size: `93M`
- Added root submission guide:
  - `submission/SUBMISSION_README.md`
- Task 3 camera-intrinsics clarification:
  - Updated Task 3 `SOLUTION_README.md` to state that both cameras produce `1280 x 720 px` images but are estimated independently because their intrinsic matrices may differ.
  - Root submission README repeats that camera #1 and camera #2 have separate `K` matrices.
- Submission contents:
  - `34` files total.
  - Size: `101M`, mostly from Task 2 `outputs_cpu_anode/best_model.pt`.
  - Task 1 includes source, ROS package source, Docker helper, local tests, and solution README.
  - Task 2 includes `task1_ml.py`, requirements files, tests, solution README, and selected non-augmented `outputs_cpu_anode` artifacts.
  - Task 3 includes `task3_cv.py`, `answers.txt`, tests, and solution README.
- Hygiene verification:
  - No `__pycache__` directories under `submission`.
  - No `.venv` directories under `submission`.
  - No ROS `build` or `devel` directories under `submission`.
  - No augmented/smoke output directories under `submission`.
  - Zip archive scan also found none of those excluded paths.
- Tests run from copied submission files:
  - Task 1: `bash tests/run_local_tests.sh`
    - passed
  - Task 2: original `.venv/bin/python -B -m unittest test_task1_ml.py` from `submission/02 Machine Learning/task`
    - `7` tests passed
  - Task 3: `python3 -B -m unittest test_task3_cv.py` from `submission/03 Computer Vision`
    - `5` tests passed
- Task 2 submission decision:
  - Submit non-augmented `outputs_cpu_anode` checkpoint/results as primary.
  - Augmented output was intentionally not copied into the submission folder.

Submission path-portability review:

- User pointed out that `submission/01 Mission Planning/task/SOLUTION_README.md` still contained local machine paths such as `/home/stafford99/...`.
- Updated Task 1 solution README in both the working folder and submission folder to use relocatable examples:
  - `cd "path/to/01 Mission Planning/task"`
  - `cd "path/to/01 Mission Planning/task/ros_ws"`
- Removed the example that called `task1_docker.sh` through my local absolute path.
- Sanitized Task 2 selected metrics JSON metadata in both working output and submission output:
  - `dataset_dir`: `interview_dataset`
  - `zip_path`: `interview_dataset.zip`
- Verification passed:
  - `grep -R "/home/stafford99\|/home/\|Robotics Software Engineer Assignment/Interview Assignments" submission` returned no matches.
  - The same scan on the rebuilt zip extracted to `/tmp/beex_submission_check` returned no matches.
  - Task 1 copied tests still passed from `submission/01 Mission Planning/task`.
  - Task 2 copied tests still passed from `submission/02 Machine Learning/task`.
  - Task 3 copied tests still passed from `submission/03 Computer Vision`.
- Rebuilt `/home/stafford99/BeeX/beex_submission.zip` after path cleanup.

Submission handoff/readability review:

- Added root handoff order to `submission/SUBMISSION_README.md`:
  - read root README
  - read each task's `SOLUTION_README.md`
  - run lightweight tests
  - use Docker or Ubuntu 20.04/ROS Noetic for Task 1 full runtime verification
  - place `interview_dataset.zip` next to `task1_ml.py` before Task 2 retraining
- Added maintainability notes:
  - Task 1: points to parser header, driver node, mission node, local test wrapper, and Docker-only compatibility files.
  - Task 2: points to COCO parsing, mask rasterization, letterbox resizing, split logic, loss/imbalance handling, augmentation, tests, and selected artifacts.
  - Task 3: points to PDF inputs, DLT, RQ/decomposition, triangulation, and tests.
- Re-verified documentation portability:
  - `grep -R "/home/stafford99\|/home/\|Robotics Software Engineer Assignment/Interview Assignments" submission` returned no matches.
  - Extracted `beex_submission.zip` to `/tmp/beex_submission_check` and ran the same scan; no matches.
- Re-ran copied submission tests after documentation updates:
  - Task 1: `bash tests/run_local_tests.sh` passed.
  - Task 2: original `.venv/bin/python -B -m unittest test_task1_ml.py` from copied submission folder passed, `7` tests.
  - Task 3: `python3 -B -m unittest test_task3_cv.py` from copied submission folder passed, `5` tests.
- Rebuilt `/home/stafford99/BeeX/beex_submission.zip`.
- Final archive checks:
  - Size: `93M`.
  - Submission folder: `34` files, `101M`.
  - No `__pycache__`, `.venv`, ROS `build/devel`, augmented outputs, or smoke outputs.
  - Submission `bx_grid_world.cpp` still matches the original BeeX-provided source byte-for-byte.

Submission LLM-use declaration review:

- Updated root `submission/SUBMISSION_README.md` with an explicit `## LLM Use` section.
- Updated all submitted task READMEs to explicitly state that I used LLM tools, including OpenAI Codex/ChatGPT-style assistance.
- The declarations state that I used LLMs for task inspection, planning, debugging, coding support, testing, and documentation, and that I reviewed/verified submitted work before packaging.
- Synced the same task README wording back to the working task folders.
- Rebuilt `/home/stafford99/BeeX/beex_submission.zip`.
- Verification passed:
  - Extracted the rebuilt zip to `/tmp/beex_submission_check`.
  - `grep -R "I used LLM tools" ...` found the declaration in root README and all three task READMEs.
  - Zip hygiene scan still found no `__pycache__`, `.venv`, ROS `build/devel`, augmented outputs, or smoke outputs.

Submission LLM-use wording correction:

- User clarified that the LLM disclosure should emphasize their direction, debugging suggestions, frequent reviews, and approvals before moving forward.
- Updated root README and all three task READMEs to state:
  - I used LLM tools under my direction.
  - I provided implementation goals/direction, debugging observations, experiment results where relevant, and review feedback.
  - I reviewed plans and code changes frequently.
  - I gave corrections where needed and approved the direction before moving on.
  - I reviewed and verified submitted code, outputs, metrics/numeric answers, and documentation before packaging.
- Synced the updated task READMEs back to the working folders.
- Rebuilt `/home/stafford99/BeeX/beex_submission.zip`.
- Verification passed:
  - Extracted zip to `/tmp/beex_submission_check`.
  - Grep confirmed the updated direction/review/approval wording in root README and all three task READMEs.
  - Zip hygiene scan still found no `__pycache__`, `.venv`, ROS `build/devel`, augmented outputs, or smoke outputs.

Task 1 simulator-source restoration review:

- User pointed out that `bx_grid_world.cpp` had been modified from the original BeeX-provided source.
- Re-read Task 1 README compatibility note:
  - local simulator dependency modifications are permissible for development on other operating systems
  - official evaluation uses the original, unmodified simulation environment
- Restored `bx_grid_world.cpp` from the user-provided original attachment in both:
  - `01 Mission Planning/task/bx_grid_world.cpp`
  - `submission/01 Mission Planning/task/bx_grid_world.cpp`
- Verification passed:
  - `cmp` confirmed local `bx_grid_world.cpp` matches the original attachment byte-for-byte.
  - `cmp` confirmed submission `bx_grid_world.cpp` matches the original attachment byte-for-byte.
  - `unzip -p beex_submission.zip .../bx_grid_world.cpp | cmp` confirmed zipped `bx_grid_world.cpp` matches the original attachment byte-for-byte.
  - `bash compile.sh` passed with the original simulator source.
  - Docker-shim compile command passed with the original simulator source:
    - `g++ -std=c++17 bx_grid_world.cpp -o /tmp/bx_grid_world_docker_shim -I docker/include -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_highgui -lutil`
  - `bash tests/run_local_tests.sh` passed from the working Task 1 folder.
  - `bash tests/run_local_tests.sh` passed from the copied submission Task 1 folder.
- Final local Docker compatibility approach:
  - Keep BeeX's original `#include <opencv2/opencv.hpp>` in `bx_grid_world.cpp`.
  - Add Docker-only shim `docker/include/opencv2/opencv.hpp` that includes only `core.hpp`, `highgui.hpp`, and `imgproc.hpp`.
  - Update Docker simulator build command to include `-I/work/docker/include` before `-I/usr/include/opencv4`.
- Rebuilt `/home/stafford99/BeeX/beex_submission.zip`.
  - Zip size remains `93M`.
  - Zip scan found no `__pycache__`, `.venv`, ROS `build/devel`, augmented output, or smoke output directories.

Task 1 Docker simulator helper repair plan:

- [x] Patch the working Docker helper so `simulator` rebuilds `bx_grid_world` when `ldd` reports either `not found` or `not a dynamic executable`.
- [x] Apply the same helper patch to the submission copy and rebuild `beex_submission.zip`.
- [x] Verify shell syntax, lightweight Task 1 tests, and the helper condition with a local stale-binary check.
- [x] Record the root cause and verification result here and in `tasks/lessons.md`.

Task 1 Docker simulator helper repair review:

- User observed `./docker/task1_docker.sh simulator` printing `not a dynamic executable`.
- Root cause:
  - The helper only rebuilt `bx_grid_world` when `ldd` reported `not found`.
  - `ldd` prints `not a dynamic executable` for a stale/script-like incompatible binary, and that message was not captured by the old condition.
- Change made:
  - Updated both the working helper and submission helper to check `ldd ./bx_grid_world 2>&1 | grep -Eq 'not found|not a dynamic executable'`.
  - This keeps BeeX's original `bx_grid_world.cpp` untouched and rebuilds the simulator binary inside the Docker container when needed.
- Verification:
  - `bash -n` passed for the working and submission Docker helpers.
  - `ldd .../task1_docker.sh 2>&1 | grep -Eq 'not found|not a dynamic executable'` confirmed the helper condition catches the observed message.
  - Working Task 1 local tests passed.
  - Submission Task 1 local tests passed after rerunning sequentially.
  - First parallel test attempt hit a `/tmp/task1_protocol_parser_test` artifact collision (`Text file busy`), so I stopped and reran the affected tests sequentially.
  - Rebuilt `/home/stafford99/BeeX/beex_submission.zip`.
  - Zip check confirmed the patched helper is included.
  - Zip hygiene scan found no `__pycache__`, `.venv`, ROS `build/devel`, augmented output, or smoke output directories.
  - Zip `bx_grid_world.cpp` still matches the original BeeX-provided simulator source byte-for-byte.

Task 1 Docker X11 authorization plan:

- [x] Update Docker helper usage text to recommend granting X11 access to the container's `root` user before launching the OpenCV simulator window.
- [x] Update Task 1 solution READMEs in the working and submission folders with the same X11 command and fallback.
- [x] Rebuild `beex_submission.zip`.
- [x] Verify helper syntax, Task 1 local tests, zip content, and that the original simulator source remains unchanged.

Task 1 Docker X11 authorization review:

- User observed:
  - `Authorization required, but no authorization protocol specified`
  - `Gtk-WARNING ... cannot open display: :0`
  - simulator still reached `TCP Server listening on port 8091`
- Interpretation:
  - The previous stale-binary issue is fixed.
  - Docker can launch the simulator, but the OpenCV/GTK display window is denied by the host X server.
  - The container runs the simulator as `root`, so `xhost +local:docker` is not consistently sufficient.
- Change made:
  - Updated working and submission Docker helper usage text to recommend `xhost +SI:localuser:root`.
  - Added `xhost +local:` as a broader fallback for desktop sessions that do not support the local-user rule.
  - Updated working and submission Task 1 solution READMEs with the same guidance.
- Verification:
  - `bash -n` passed for the working and submission Docker helpers.
  - Working Task 1 local tests passed.
  - Submission Task 1 local tests passed.
  - Rebuilt `/home/stafford99/BeeX/beex_submission.zip`.
  - Zip-level grep confirmed the X11 commands are included in the packaged Task 1 README and Docker helper.
  - Zip hygiene scan found no `__pycache__`, `.venv`, ROS `build/devel`, augmented output, or smoke output directories.
  - Zip `bx_grid_world.cpp` still matches the original BeeX-provided simulator source byte-for-byte.

Task 1 local simulator keep-alive plan:

- [x] Add an opt-in local simulator mode that logs the pipeline success once but keeps the simulator loop running when requested.
- [x] Add a local Docker helper command for the keep-alive simulator mode.
- [x] Document the local-only command without changing the packaged submission simulator.
- [x] Compile and run lightweight Task 1 checks.

Task 1 local simulator keep-alive review:

- User wanted the GridWorld simulator to keep running after printing `Success! Pipeline goal reached...` so the robot can still be inspected or moved afterward.
- Change made in the working Task 1 folder only:
  - Added `BEEX_KEEP_SIM_RUNNING_AFTER_SUCCESS=1` support to local `bx_grid_world.cpp`.
  - The simulator now logs success once, marks the pipeline goal complete, and continues TCP, UDP, serial, and render updates when that environment variable is enabled.
  - Default behavior is unchanged: without the environment variable, success still returns `false` from `spinOnce()` and exits the simulator loop.
  - Added local Docker helper command `./docker/task1_docker.sh simulator-keepalive`.
  - Documented the local-only keep-alive command in the working Task 1 `SOLUTION_README.md`.
- Submission protection:
  - Did not copy this simulator behavior change into `submission/01 Mission Planning/task/bx_grid_world.cpp`.
  - Did not rebuild `beex_submission.zip` with this local-only change.
  - Verified the submission folder simulator still matches BeeX's original source byte-for-byte.
  - Verified the existing zip simulator still matches BeeX's original source byte-for-byte.
  - Verified the existing zip does not contain the keep-alive symbols.
- Verification:
  - `bash -n docker/task1_docker.sh` passed.
  - `bash compile.sh` passed with the local keep-alive-capable simulator source.
  - Docker-shim compile command passed:
    - `g++ -std=c++17 bx_grid_world.cpp -o /tmp/bx_grid_world_keepalive_check -I docker/include -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_highgui -lutil`
  - `bash tests/run_local_tests.sh` passed.

Final submission confidence check plan:

- [x] Confirm whether the observed Task 1 success timing is appropriate by checking the simulator success criteria and provided run log.
- [x] Verify the packaged submission still contains the intended files and excludes development/build artifacts.
- [x] Re-run lightweight tests from the submission copies of Tasks 1, 2, and 3.
- [x] Re-scan submission documentation for local-only paths, LLM-use disclosure, and handoff clarity.
- [x] Record the final result and any caveats.

Final submission confidence check review:

- Task 1 success timing:
  - The attached simulator log reports success at `Robot Pos: (8.39, 9.85)` with `Elapsed time: 183824 ms`.
  - This is appropriate, not overconfident, because the simulator itself owns the hidden waypoint success criteria.
  - In the original simulator, `hasReachedPipelineGoal()` pops each hidden pipeline waypoint when the robot is within `0.3 m`; success is printed only after the waypoint list is empty.
  - The provided simulator creates the final hidden point at Northing `10.0 m` with randomized Easting up to `+/-9 m`, so a success near `N=9.85` is inside the simulator's own tolerance.
  - Later warnings around `N=10.80` and `N=11.30` are consistent with the local keep-alive simulator continuing after success; those positions are already beyond the pipeline's final waypoint region.
- Submission/archive checks:
  - `rg` found no local machine paths in `submission`.
  - `zipgrep` found no local machine paths in `beex_submission.zip`.
  - Zip hygiene scan found no `__pycache__`, `.venv`, ROS `build/devel`, augmented output, or smoke output directories.
  - Expected files are present in the submission folder, including Task 1 ROS nodes, Task 2 selected checkpoint/metrics, and Task 3 code/answers.
  - Submission folder `bx_grid_world.cpp` matches the original BeeX-provided simulator source byte-for-byte.
  - Zipped `bx_grid_world.cpp` matches the original BeeX-provided simulator source byte-for-byte.
  - `zipgrep` confirmed the local-only keep-alive debug symbols are not present in `beex_submission.zip`.
  - LLM-use disclosure is present in the root README and all three submitted task READMEs.
- Tests run from `submission`:
  - Task 1: `bash tests/run_local_tests.sh` passed.
  - Task 2: `.venv/bin/python -B -m unittest test_task1_ml.py` passed, `7` tests.
  - Task 3: `python3 -B -m unittest test_task3_cv.py` passed, `5` tests.
  - `python3 -m json.tool outputs_cpu_anode/metrics.json` passed.
- Tests run from extracted zip:
  - Extracted archive to `/tmp/beex_final_zip_check.OFew4p`.
  - Task 1 extracted test passed.
  - Task 2 extracted unit tests passed, `7` tests.
  - Task 3 extracted unit tests passed, `5` tests.
- Note:
  - One optional metrics-summary probe initially used the wrong JSON key (`metrics`) and failed with `KeyError`; I inspected the actual metrics schema and confirmed the metrics are valid under `validation_metrics`, `test_metrics`, and `final_metrics`.

GitHub push plan:

- [x] Initialize a Git repository for `/home/stafford99/BeeX`.
- [x] Add a protective `.gitignore` so local virtual environments, extracted datasets, ROS build folders, exploratory ML outputs, and raw checkpoints are not committed.
- [x] Add a root `README.md` that points reviewers to the final submission package and extracted source folders.
- [x] Confirm no candidate tracked file exceeds GitHub's individual file limit.
- [x] Re-run lightweight submission tests before committing.
- [x] Commit the prepared assignment files.
- [x] Push `main` to `https://github.com/StaffordHo/BeeX.git`.

GitHub push review:

- Repository was initialized locally on branch `main`.
- Remote `origin` was set to `https://github.com/StaffordHo/BeeX.git`.
- Candidate tracked file count before staging: `73`.
- Largest candidate tracked file is `beex_submission.zip` at `96,896,557` bytes.
- Files intentionally ignored:
  - `.venv/`
  - extracted `interview_dataset/`
  - `interview_dataset.zip`
  - ROS `build/`, `devel/`, `install/`, and `log/`
  - raw `*.pt`, `*.pth`, and `*.ckpt` checkpoints
  - exploratory `outputs*` folders
- The final submission zip is intentionally tracked because it is the packaged handoff artifact.
- Initial commit created:
  - `4ff8fe9 Add BeeX robotics assignment submission`
- Push result:
  - `main` pushed to `https://github.com/StaffordHo/BeeX.git`
  - Local `main` now tracks `origin/main`.
  - GitHub warned that `beex_submission.zip` is larger than the recommended `50 MB`, but the push succeeded because it is below the hard individual-file limit.
- Verification before commit:
  - Task 1 submission local tests passed.
  - Task 2 submission unit tests passed, `7` tests.
  - Task 3 submission unit tests passed, `5` tests.
