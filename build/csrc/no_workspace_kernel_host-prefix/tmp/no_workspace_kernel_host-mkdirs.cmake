# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/usr/local/Ascend/ascend-toolkit/latest/tools/tikcpp/ascendc_kernel_cmake/host_project")
  file(MAKE_DIRECTORY "/usr/local/Ascend/ascend-toolkit/latest/tools/tikcpp/ascendc_kernel_cmake/host_project")
endif()
file(MAKE_DIRECTORY
  "/home/s886374/kvoffload/ascend-kernel/build/csrc/no_workspace_kernel_host-prefix/src/no_workspace_kernel_host-build"
  "/home/s886374/kvoffload/ascend-kernel/build/csrc/no_workspace_kernel_host-prefix"
  "/home/s886374/kvoffload/ascend-kernel/build/csrc/no_workspace_kernel_host-prefix/tmp"
  "/home/s886374/kvoffload/ascend-kernel/build/csrc/no_workspace_kernel_host-prefix/src/no_workspace_kernel_host-stamp"
  "/home/s886374/kvoffload/ascend-kernel/build/csrc/no_workspace_kernel_host-prefix/src"
  "/home/s886374/kvoffload/ascend-kernel/build/csrc/no_workspace_kernel_host-prefix/src/no_workspace_kernel_host-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/s886374/kvoffload/ascend-kernel/build/csrc/no_workspace_kernel_host-prefix/src/no_workspace_kernel_host-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/s886374/kvoffload/ascend-kernel/build/csrc/no_workspace_kernel_host-prefix/src/no_workspace_kernel_host-stamp${cfgdir}") # cfgdir has leading slash
endif()
