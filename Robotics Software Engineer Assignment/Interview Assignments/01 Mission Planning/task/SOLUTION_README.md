# Autonomous Subsea Pipeline Tracking Solution

## Overview

I kept the simulator boundary split requested in the assignment:

- `ros_ws/src/driver`: ROS 1 driver package for simulator TCP, UDP, and serial communication.
- `ros_ws/src/mission`: ROS 1 mission package that only uses ROS topics and services.

I left the provided simulator source unchanged for submission. For local Docker compatibility, I keep OpenCV workarounds in the Docker helper files rather than editing the submitted `bx_grid_world.cpp`.

For local debugging only, I added an opt-in simulator keep-alive mode in my working copy. The submitted simulator copy remains the original BeeX source. The keep-alive mode logs pipeline success once, then continues TCP, UDP, serial, and window updates so I can inspect or move the robot after success.

## Simulator Reference

I based the implementation on the structure and protocol comments in `bx_grid_world.cpp`:

- `GameWorld::handleTcpClient()` defines the `Q` handshake and movement bytes.
- `GameWorld::writeOdomToUdp()` defines the `<%.3fN,%.3fE>` odometry format.
- `GameWorld::writeSensorToSerial()` defines the `%.4fV==` FG sensor line format.
- `GameWorld::spinOnce()` shows the 10 Hz update cadence and 0.05 m command steps.

I mirrored the simulator's sectioned style for TCP, UDP, serial, and runtime state, while keeping protocol parsing in a small testable header.

## Mission Strategy

I implemented the mission planner as a finite state machine driven by `/odom` and `/fg_readings`.

1. Search with north-first zigzag sweeps over nearby northing rows until the filtered FG signal rises above the background/noise threshold.
2. Center on the pipe by scanning east-west at fixed northing and returning to the easting with the strongest FG value.
3. Acquire the southern endpoint by stepping south, using the previous centerline estimate first, and backing up to the last strong centered reading once sustained loss is observed.
4. Track north by predicting the next centerline easting from recent centered samples. If the predicted point already has a strong FG reading, accept it immediately; otherwise, perform a small east-west centering scan.
5. Reacquire with expanding local scans if the pipe signal is lost before the terminus.
6. Treat sustained northward loss after valid centerline tracking as the northern endpoint, then make a short fine-grained endpoint pass instead of searching indefinitely beyond the pipe. In the provided simulator, success is also detected internally when the hidden waypoints are reached.

I exposed thresholds and scan widths as ROS parameters in `mission_node.cpp` so I can tune the mission without changing the driver. Search now uses `search_half_width_m = 5.0 m`, `search_row_step_m = 0.50 m`, and north-first zigzag rows to avoid spending early time south of the start. Normal tracking uses `row_step_m = 0.50 m`, `center_scan_radius_m = 0.75 m`, and `center_accept_threshold = 0.65`; I reserve wider scans for reacquisition. I clamp motion targets inside a conservative `+/-45 m` operating box to preserve margin inside the simulator's 100 m x 100 m area.

## ROS Interface

I made the `driver` package publish:

- `/odom` as `geometry_msgs/Point`, with `x = Easting`, `y = Northing`, `z = 0`.
- `/fg_readings` as `std_msgs/Float32`.

I made the `driver` package provide:

- `/command` as `driver/Command`.

Service request:

```text
string command
```

Accepted command strings are `W/A/S/D`, `north/south/east/west`, and `N/E/S/W`.

## Build

All commands in this section assume the current directory is this task folder:

```bash
cd "path/to/01 Mission Planning/task"
```

### Docker on Ubuntu 22.04

For Ubuntu 22.04 hosts, use the provided Docker workflow to run Ubuntu 20.04 + ROS Noetic without touching a local ROS 2 Humble installation.

The simulator and ROS nodes should run in the same container because `bx_grid_world.cpp` creates a pseudo-terminal such as `/dev/pts/<X>` and passes that path to the driver through the TCP handshake. Separate containers do not reliably share that pseudo-terminal namespace.

Allow local X11 access for the simulator window. The container runs the simulator as `root`, so I grant access to that local user:

```bash
xhost +SI:localuser:root
```

If the desktop session does not support the server-interpreted local-user rule, use the broader local fallback:

```bash
xhost +local:
```

Build and start the shared container:

```bash
./docker/task1_docker.sh build
./docker/task1_docker.sh start
```

Build the simulator and ROS workspace inside Docker:

```bash
./docker/task1_docker.sh sim-build
./docker/task1_docker.sh ros-build
```

The Docker simulator build uses a minimal OpenCV link line instead of the provided `compile.sh` because the full `libopencv-dev` meta-package can exceed Docker's available apt cache space on smaller installations. A Docker-only OpenCV include shim lets the original `bx_grid_world.cpp` keep its provided `#include <opencv2/opencv.hpp>`. The native workflow below still uses `compile.sh`.

If `./bx_grid_world` was previously built on the Ubuntu 22.04 host, it may fail inside Docker with an OpenCV shared-library error such as `libopencv_highgui.so.4.5d`. Run `./docker/task1_docker.sh sim-build` to rebuild the binary inside the Ubuntu 20.04 container. The `simulator` helper also performs this check automatically.

Run the mission in four host terminals:

Keep the first three terminals running. Start `mission` only after `driver` prints that it connected to the simulator.

If you open a fresh terminal, `cd` into the extracted task folder first:

```bash
cd "path/to/01 Mission Planning/task"
```

```bash
./docker/task1_docker.sh roscore
```

```bash
./docker/task1_docker.sh simulator
```

For local debugging, this variant keeps the simulator open after the pipeline goal is reached:

```bash
./docker/task1_docker.sh simulator-keepalive
```

```bash
./docker/task1_docker.sh driver
```

```bash
./docker/task1_docker.sh mission
```

Expected driver startup:

```text
Connected to simulator: serial=/dev/pts/<X> udp=127.0.0.1:<Y>
```

If `roscore` says another master is already running, do not start a second one. Keep using the existing ROS master and continue with simulator, driver, and mission.

Useful Docker checks:

```bash
./docker/task1_docker.sh parser-test
./docker/task1_docker.sh shell
./docker/task1_docker.sh stop
```

If Docker reports permission denied for `/var/run/docker.sock`, either run the helper through `sudo` or add your user to the `docker` group and start a new login session:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
docker ps
./docker/task1_docker.sh build
```

If `newgrp docker` does not refresh the group membership for your desktop session, fully log out and log back in. For a one-off run, use interactive sudo:

```bash
sudo -E ./docker/task1_docker.sh build
sudo -E ./docker/task1_docker.sh start
```

If you open separate terminals for `roscore`, simulator, driver, and mission before logging out/in, run this once in each terminal:

```bash
newgrp docker
cd "path/to/01 Mission Planning/task"
```

### Native Ubuntu 20.04

Build the simulator:

```bash
bash compile.sh
```

Build the ROS workspace on Ubuntu 20.04 with ROS Noetic:

```bash
cd ros_ws
catkin_make
source devel/setup.bash
```

## Run

Terminal 1:

```bash
roscore
```

Terminal 2:

```bash
cd "path/to/01 Mission Planning/task"
./bx_grid_world
```

Terminal 3:

```bash
cd "path/to/01 Mission Planning/task/ros_ws"
source devel/setup.bash
rosrun driver driver_node
```

Terminal 4:

```bash
cd "path/to/01 Mission Planning/task/ros_ws"
source devel/setup.bash
rosrun mission mission_node
```

Useful smoke checks:

```bash
rostopic echo /odom
rostopic echo /fg_readings
rosservice call /command "command: 'north'"
```

## Tests

For host-side checks that do not require ROS Noetic, I run:

```bash
bash tests/run_local_tests.sh
```

This compiles and runs the C++ protocol parser test, checks the Docker helper shell syntax, and validates the Docker Compose file structure when `docker compose` is available.

## Maintainability Notes

- `ros_ws/src/driver/include/driver/protocol.hpp` contains the parser helpers and can be tested without ROS.
- `ros_ws/src/driver/src/driver_node.cpp` owns simulator TCP, UDP, serial, and `/command` service behavior.
- `ros_ws/src/mission/src/mission_node.cpp` owns mission strategy and exposes key thresholds as ROS parameters.
- `tests/run_local_tests.sh` is the fastest regression check after editing parser or Docker helper logic.
- I kept `bx_grid_world.cpp` as the original provided simulator source; Docker-only compatibility lives under `docker/`.

## Local Verification Performed

My current development environment does not have ROS or `catkin_make` installed, so I still need to run full ROS runtime verification on ROS Noetic.

I verified locally:

- Simulator builds with `bash compile.sh`.
- Protocol parser test builds with `g++ -std=c++17`.
- Protocol parser test passes for TCP handshake, FG serial, UDP odometry, and command parsing.

## LLM Use

I used LLM tools, including OpenAI Codex/ChatGPT-style assistance, under my direction while completing this task. I provided the mission-planning goals, implementation direction, Docker/debugging observations, and review feedback. I reviewed plans and code changes frequently, gave corrections where needed, and approved the direction before moving on. I reviewed and verified the submitted code and documentation before packaging.
