# BeeX Robotics Software Engineer Assignment Submission

This folder contains my prepared submission files for the three interview tasks.

## LLM Use

I used LLM tools, including OpenAI Codex/ChatGPT-style assistance, while completing this assignment. I directed the LLM with the task goals, implementation choices, debugging observations, and review feedback. I reviewed the plans and changes frequently, gave corrections and suggestions during debugging, and approved the direction before moving on. I reviewed and verified the submitted code, outputs, and documentation before packaging.

Recommended handoff order:

1. Read this file.
2. Open each task's `SOLUTION_README.md`.
3. Run the lightweight tests listed below.
4. For Task 1 full runtime verification, use the Docker workflow or Ubuntu 20.04 + ROS Noetic.
5. For Task 2 retraining, place the provided `interview_dataset.zip` next to `task1_ml.py`; the script can extract it automatically.

## Task 1: Mission Planning

Path:

```text
01 Mission Planning/task
```

Included:

- ROS 1/catkin source under `ros_ws/src`
- original provided simulator source and compile script
- Docker workflow for Ubuntu 22.04 host / ROS Noetic container use
- local non-ROS test wrapper
- solution README

Local non-ROS tests:

```bash
cd "01 Mission Planning/task"
bash tests/run_local_tests.sh
```

Full runtime needs ROS Noetic and the simulator, so I documented that workflow in the task README rather than making the local test wrapper depend on ROS.

## Task 2: Machine Learning

Path:

```text
02 Machine Learning/task
```

Included:

- `task1_ml.py`
- CPU-safe and non-Torch requirements files
- unit tests
- solution README
- selected non-augmented training artifacts in `outputs_cpu_anode`

I selected the non-augmented model for submission because it has the stronger visible held-out anode result:

- test `anode IoU=0.5570`
- test `anode F1=0.7155`
- test `member IoU=0.5694`
- test `member F1=0.7256`

Unit tests:

```bash
cd "02 Machine Learning/task"
python -B -m unittest test_task1_ml.py
```

The submitted model artifacts are under `outputs_cpu_anode`. I intentionally did not include the dataset archive or virtual environment in the submission package.

## Task 3: Computer Vision

Path:

```text
03 Computer Vision
```

Included:

- `task3_cv.py`
- `answers.txt`
- `test_task3_cv.py`
- solution README

The Python solution uses `numpy` only. It does not import OpenCV.

I handled the PDF note that both cameras produce `1280 x 720 px` images but may have different intrinsic matrices by estimating each camera independently. Camera #1 and camera #2 have separate `K` matrices in the answer.

Unit tests:

```bash
cd "03 Computer Vision"
python3 -B -m unittest test_task3_cv.py
```

The generated numeric answer is in `answers.txt`.
