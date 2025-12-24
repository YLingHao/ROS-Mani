cmake_minimum_required(VERSION 3.0.2)
project(mobile_pkg)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  message_runtime
  roscpp
  sensor_msgs
  std_msgs
  geometry_msgs
  waterplus_map_tools
  wpb_mani_behaviors
  image_transport
  cv_bridge
  moveit_ros_move_group
  moveit_ros_planning_interface
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
  pcl_ros
)

## 确保正确找到 OpenCV
find_package(OpenCV REQUIRED)

## 确保正确找到 PCL
find_package(PCL REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS 
    roscpp
    std_msgs
    geometry_msgs
    waterplus_map_tools
    wpb_mani_behaviors
    image_transport
    cv_bridge
    moveit_ros_move_group
    moveit_ros_planning_interface
    tf2_ros
    tf2_geometry_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

## Declare cruise_node executable (now includes MoveIt logic)
add_executable(xm src/xm.cpp)
add_dependencies(xm ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(xm
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

## Declare test_node executable
add_executable(test_node src/test_node.cpp)
add_dependencies(test_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(test_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)

## Declare test_node executable
add_executable(xche src/xche.cpp)
add_dependencies(xche ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(xche
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)


