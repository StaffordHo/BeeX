#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${TASK_DIR}"

echo "[Task 1] Compiling protocol parser test"
g++ -std=c++17 -Wall -Wextra -Wpedantic \
    -I ros_ws/src/driver/include \
    ros_ws/src/driver/test/test_protocol_parsers.cpp \
    -o /tmp/task1_protocol_parser_test

echo "[Task 1] Running protocol parser test"
/tmp/task1_protocol_parser_test

echo "[Task 1] Checking Docker helper shell syntax"
bash -n docker/task1_docker.sh

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    echo "[Task 1] Checking Docker Compose file structure"
    docker compose -f docker-compose.yml config >/dev/null
else
    echo "[Task 1] Skipping Docker Compose config check because docker compose is unavailable"
fi

echo "[Task 1] Local non-ROS tests passed"
