#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${TASK_DIR}/docker-compose.yml"
SERVICE="task1"
SIM_BUILD_CMD="g++ -std=c++17 bx_grid_world.cpp -o bx_grid_world -I/work/docker/include -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_highgui -lutil"

usage() {
    cat <<'EOF'
Usage: docker/task1_docker.sh <command>

Commands:
  build          Build the ROS Noetic Docker image.
  start          Start the shared Task 1 container.
  stop           Stop the shared Task 1 container.
  shell          Open a shell inside the shared container.
  sim-build      Build bx_grid_world inside the container.
  ros-build      Build ros_ws with catkin_make inside the container.
  parser-test    Build and run the standalone protocol parser test.
  roscore        Run roscore inside the shared container.
  ros-check      Check whether ROS master is reachable.
  simulator      Run ./bx_grid_world inside the shared container.
  simulator-keepalive
                 Run ./bx_grid_world and keep it open after pipeline success.
  driver         Run rosrun driver driver_node inside the shared container.
  mission        Run rosrun mission mission_node inside the shared container.

Recommended terminal flow:
  ./docker/task1_docker.sh build
  ./docker/task1_docker.sh start
  ./docker/task1_docker.sh sim-build
  ./docker/task1_docker.sh ros-build

Then open four terminals and run:
  ./docker/task1_docker.sh roscore
  ./docker/task1_docker.sh simulator
  ./docker/task1_docker.sh driver
  ./docker/task1_docker.sh mission

For the OpenCV simulator window, allow local X11 access on the host first:
  xhost +SI:localuser:root
If that is not supported by your desktop session, use the broader local fallback:
  xhost +local:
EOF
}

require_docker_access() {
    if ! docker info >/dev/null 2>&1; then
        cat >&2 <<'EOF'
Docker is installed, but this shell cannot access the Docker daemon.

Preferred fix:
  sudo usermod -aG docker "$USER"
  newgrp docker
  docker ps

Important: newgrp only refreshes the current terminal. Run "newgrp docker"
in each new terminal before using this helper, or fully log out and log back in.

After Docker access works, rerun:
  ./docker/task1_docker.sh build

One-off alternative:
  sudo -E ./docker/task1_docker.sh build
  sudo -E ./docker/task1_docker.sh start

EOF
        exit 1
    fi
}

compose() {
    require_docker_access
    docker compose -f "${COMPOSE_FILE}" "$@"
}

ensure_started() {
    compose up -d "${SERVICE}" >/dev/null
}

exec_in_container() {
    ensure_started
    compose exec -T "${SERVICE}" bash -lc "$1"
}

case "${1:-}" in
    build)
        compose build "${SERVICE}"
        ;;
    start)
        ensure_started
        ;;
    stop)
        compose down
        ;;
    shell)
        ensure_started
        compose exec "${SERVICE}" bash
        ;;
    sim-build)
        exec_in_container "cd /work && ${SIM_BUILD_CMD}"
        ;;
    ros-build)
        exec_in_container "source /opt/ros/noetic/setup.bash && cd /work/ros_ws && catkin_make"
        ;;
    parser-test)
        exec_in_container "cd /work && g++ -std=c++17 -Wall -Wextra -Wpedantic -I ros_ws/src/driver/include ros_ws/src/driver/test/test_protocol_parsers.cpp -o /tmp/task1_protocol_parser_test && /tmp/task1_protocol_parser_test"
        ;;
    roscore)
        exec_in_container "source /opt/ros/noetic/setup.bash && if rosnode list >/dev/null 2>&1; then echo 'ROS master is already running at' \"\${ROS_MASTER_URI}\"; echo 'Use the existing roscore and start simulator/driver/mission.'; sleep infinity; else roscore; fi"
        ;;
    ros-check)
        exec_in_container "source /opt/ros/noetic/setup.bash && rosnode list"
        ;;
    simulator)
        exec_in_container "cd /work && if [ ! -x ./bx_grid_world ] || ldd ./bx_grid_world 2>&1 | grep -Eq 'not found|not a dynamic executable'; then ${SIM_BUILD_CMD}; fi && ./bx_grid_world"
        ;;
    simulator-keepalive)
        exec_in_container "cd /work && ${SIM_BUILD_CMD} && BEEX_KEEP_SIM_RUNNING_AFTER_SUCCESS=1 ./bx_grid_world"
        ;;
    driver)
        exec_in_container "source /opt/ros/noetic/setup.bash && source /work/ros_ws/devel/setup.bash && rosrun driver driver_node"
        ;;
    mission)
        exec_in_container "source /opt/ros/noetic/setup.bash && source /work/ros_ws/devel/setup.bash && rosrun mission mission_node"
        ;;
    ""|-h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: $1" >&2
        usage >&2
        exit 2
        ;;
esac
