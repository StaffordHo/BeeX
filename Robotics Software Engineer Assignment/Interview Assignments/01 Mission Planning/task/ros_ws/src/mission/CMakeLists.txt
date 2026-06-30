cmake_minimum_required(VERSION 3.0.2)
project(mission)

add_compile_options(-std=c++17 -Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  driver
  geometry_msgs
  roscpp
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS driver geometry_msgs roscpp std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(mission_node src/mission_node.cpp)
add_dependencies(mission_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(mission_node ${catkin_LIBRARIES})
