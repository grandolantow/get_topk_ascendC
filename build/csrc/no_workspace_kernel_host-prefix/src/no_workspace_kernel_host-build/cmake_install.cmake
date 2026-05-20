# Install script for directory: /usr/local/Ascend/ascend-toolkit/latest/tools/tikcpp/ascendc_kernel_cmake/host_project

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/s886374/kvoffload/ascend-kernel/build/csrc/no_workspace_kernel_host_dir")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "RELEASE")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./objects-RELEASE/host_bisheng_obj" TYPE FILE RENAME "home/s886374/kvoffload/ascend-kernel/csrc/ops/cache_miss_topk/op_kernel/cache_miss_topk.cpp.o" FILES "/home/s886374/kvoffload/ascend-kernel/build/csrc/no_workspace_kernel_host-prefix/src/no_workspace_kernel_host-build/CMakeFiles/host_bisheng_obj.dir//home/s886374/kvoffload/ascend-kernel/csrc/ops/cache_miss_topk/op_kernel/cache_miss_topk.cpp.o")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./objects-RELEASE/host_bisheng_obj" TYPE FILE RENAME "home/s886374/kvoffload/ascend-kernel/csrc/ops/helloworld/op_kernel/kernel_helloworld.cpp.o" FILES "/home/s886374/kvoffload/ascend-kernel/build/csrc/no_workspace_kernel_host-prefix/src/no_workspace_kernel_host-build/CMakeFiles/host_bisheng_obj.dir//home/s886374/kvoffload/ascend-kernel/csrc/ops/helloworld/op_kernel/kernel_helloworld.cpp.o")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/s886374/kvoffload/ascend-kernel/build/csrc/no_workspace_kernel_host-prefix/src/no_workspace_kernel_host-build/CMakeFiles/host_bisheng_obj.dir/install-cxx-module-bmi-RELEASE.cmake" OPTIONAL)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/s886374/kvoffload/ascend-kernel/build/csrc/no_workspace_kernel_host-prefix/src/no_workspace_kernel_host-build/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/s886374/kvoffload/ascend-kernel/build/csrc/no_workspace_kernel_host-prefix/src/no_workspace_kernel_host-build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
