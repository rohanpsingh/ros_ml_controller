cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(ros_ml_controller)

find_package(catkin REQUIRED COMPONENTS
   rospy
   std_msgs
   message_generation
)

add_service_files(
  DIRECTORY srv
  FILES
  ControlStep.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS std_msgs
)

catkin_install_python(PROGRAMS scripts/controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
