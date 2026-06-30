# Project Lessons

## Workflow Rules

- Start non-trivial work by reading the relevant README/PDF and local files, then write a checked plan to `tasks/todo.md`.
- Do not begin implementation until the plan checkpoint is confirmed.
- Use subagents for independent exploration or verification when the task has separable parts.
- If a path goes sideways, stop and re-plan instead of layering fixes on top of confusion.
- Keep changes minimal, readable, and scoped to the current assignment task.
- Verify before marking work done: run builds/tests/smoke checks, record what passed, and call out anything not run.
- After a user correction, add the repeated mistake pattern and prevention rule here before continuing.

## Corrections

- 2026-06-21: When confirming assignment context, explicitly verify the exact README/PDF paths and distinguish parent-level `README.pdf` files from task-level `README.md` files. Do not imply a PDF exists inside a task folder unless the file is actually there.
- 2026-06-21: Treat the workflow document as binding for this project: plan first in `tasks/todo.md`, check in before implementation, track progress as work is completed, document verification/results, and update lessons after corrections.
- 2026-06-21: When assignment starter/simulator code is provided, use it as the local style and protocol reference. Add concise comments that connect implementation choices back to the relevant simulator behavior, while avoiding redundant line-by-line narration.
- 2026-06-21: When giving shell commands for this workspace, include either the absolute path or explicitly say the command assumes the current directory is `/home/stafford99/BeeX`. Do not rely on implicit repo-root context.
- 2026-06-21: For Docker workflows, check daemon access explicitly and give `docker ps`, `newgrp docker`/log-out guidance, plus an interactive `sudo -E` fallback. A Docker permission error is environment setup, not a project build failure.
- 2026-06-21: When Docker builds fail from apt/cache disk pressure, reduce dependency scope before telling the user to free disk. For OpenCV, avoid `libopencv-dev` if the code only needs a few modules.
- 2026-06-21: `newgrp docker` only refreshes Docker group access for the current terminal. When a workflow needs multiple terminals, tell the user to run `newgrp docker` in each terminal or fully log out/in before launching the multi-terminal commands.
- 2026-06-21: For POSIX serial/TTY reads with `VMIN=0` and `VTIME=0`, `read()` may return `0` simply because no bytes are currently available. Do not treat zero-byte reads as device closure in that mode; wait and retry.
- 2026-06-21: Bind-mounted Docker workspaces can contain host-built binaries linked against incompatible host libraries. For container workflows, rebuild native binaries inside the container or auto-detect unresolved shared libraries before running them.
- 2026-06-22: In mission logs, repeated low-FG reacquisition after successful northward tracking can mean the robot has passed the pipeline terminus, not that it lost the pipe mid-track. Before tuning search wider or longer, check whether the behavior should stop/finish at the northern endpoint.
- 2026-06-22: For Python ML assignments, avoid pinning old exact `torch`/`torchvision` versions without checking the user's active Python version. Python 3.13 needs newer wheels, so prefer compatible ranges or provide a Python 3.10/3.11 environment option.
- 2026-06-23: For PyTorch installs, avoid a plain broad `torch>=...` requirement on modern Python when disk is limited; pip may choose CUDA-enabled wheels and pull many large `nvidia-*` packages. Provide an explicit CPU-wheel install path for smoke tests and document GPU installs separately.
- 2026-06-23: Submission-facing assignment documentation should read as first-person singular from the candidate. Avoid team/passive phrasing such as "we", "our", "this solution", or "LLM assistance was used"; prefer "I implemented", "I used", and "I verified".
- 2026-06-23: Do not modify provided simulator/starter source for submission unless the assignment explicitly requires it. If local compatibility needs a change, keep it in Docker/build helpers or document it separately, and restore the original provided file before packaging.
- 2026-06-23: Submission-facing run instructions must be relocatable. Avoid absolute paths like `/home/stafford99/...` and long local workspace paths; use paths relative to the extracted submission/task folder.
- 2026-06-23: LLM-use declarations should accurately describe the user's directed workflow: the user provided implementation direction, debugging suggestions, frequent reviews, and approvals before moving forward. Do not imply the LLM independently owned the solution decisions.
- 2026-06-23: When using `ldd` in Docker/helper scripts, capture stderr and treat both `not found` and `not a dynamic executable` as stale or incompatible binary conditions. Rebuild inside the target container instead of trying to run a host-built or script-like binary.
- 2026-06-23: For Dockerized GUI apps running as container `root`, X11 setup should grant access to `root` explicitly with `xhost +SI:localuser:root`. Do not assume `xhost +local:docker` will authorize the actual process user on every Ubuntu desktop session.
