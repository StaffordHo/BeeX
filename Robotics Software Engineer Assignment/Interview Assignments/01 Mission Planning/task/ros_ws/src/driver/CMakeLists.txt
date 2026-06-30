cmake_minimum_required(VERSION 3.0.2)
project(driver)

add_compile_options(-std=c++17 -Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  roscpp
  std_msgs
)

add_service_files(
  FILES
  Command.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS geometry_msgs message_runtime roscpp std_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(driver_node src/driver_node.cpp)
add_dependencies(driver_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(driver_node ${catkin_LIBRARIES})

add_executable(protocol_parser_test test/test_protocol_parsers.cpp)
target_include_directories(protocol_parser_test PRIVATE include)
