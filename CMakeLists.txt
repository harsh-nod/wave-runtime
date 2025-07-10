# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.19...3.27)
project(wave_runtime)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# =============================================================================
# Global Configuration
# =============================================================================

# Set ROCm root directory
set(ROCM_ROOT "/opt/rocm" CACHE PATH "ROCm installation root directory")

# =============================================================================
# Platform Detection
# =============================================================================

function(detect_platform)
  if(WIN32)
    set(PLATFORM_WINDOWS TRUE PARENT_SCOPE)
    message(STATUS "Detected platform: Windows")
  elseif(APPLE)
    set(PLATFORM_MACOS TRUE PARENT_SCOPE)
    message(STATUS "Detected platform: macOS")
  elseif(UNIX AND NOT APPLE)
    set(PLATFORM_LINUX TRUE PARENT_SCOPE)
    message(STATUS "Detected platform: Linux")
  else()
    message(WARNING "Unknown platform detected")
  endif()
endfunction()

# =============================================================================
# Dependency Checking
# =============================================================================

function(check_dependencies)
  set(ALL_DEPS_AVAILABLE TRUE PARENT_SCOPE)

  # Only build on Linux
  if(NOT PLATFORM_LINUX)
    message(STATUS "Build only supported on Linux - skipping build")
    set(ALL_DEPS_AVAILABLE FALSE PARENT_SCOPE)
    return()
  endif()

  # Add ROCm paths for HIP package discovery
  list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")
  file(GLOB subdirectories "${ROCM_ROOT}/lib/cmake/*")
  foreach(subdir ${subdirectories})
    if(IS_DIRECTORY ${subdir})
      list(APPEND CMAKE_PREFIX_PATH "${subdir}")
    endif()
  endforeach()

  # Check nanobind
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE nanobind_result
    ERROR_QUIET)

  if(nanobind_result)
    message(STATUS "nanobind not found - skipping build")
    set(ALL_DEPS_AVAILABLE FALSE PARENT_SCOPE)
    return()
  endif()

  find_package(nanobind CONFIG REQUIRED)

  # Check PyTorch
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.__path__[0])"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE torch_PATH
    RESULT_VARIABLE torch_result
    ERROR_QUIET)

  if(torch_result)
    message(STATUS "PyTorch not found - skipping build")
    set(ALL_DEPS_AVAILABLE FALSE PARENT_SCOPE)
    return()
  endif()

  # Check CUDA availability
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.cuda.is_available())"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE torch_cuda_available
    RESULT_VARIABLE torch_cuda_result
    ERROR_QUIET)

  if(torch_cuda_result)
    message(STATUS "PyTorch CUDA not available - skipping build")
    set(ALL_DEPS_AVAILABLE FALSE PARENT_SCOPE)
    return()
  endif()

  # Check HIP
  find_package(hip QUIET)
  if(NOT hip_FOUND)
    message(STATUS "HIP not found - skipping build")
    set(ALL_DEPS_AVAILABLE FALSE PARENT_SCOPE)
    return()
  endif()

  # Set variables for parent scope
  set(nanobind_ROOT ${nanobind_ROOT} PARENT_SCOPE)
  set(torch_PATH ${torch_PATH} PARENT_SCOPE)
  set(torch_cuda_available ${torch_cuda_available} PARENT_SCOPE)
endfunction()

# =============================================================================
# Platform-Specific Configuration
# =============================================================================

function(configure_platform_specific)
  # Set HIP definitions
  add_definitions(-D__HIP_PLATFORM_AMD__ -DUSE_ROCM)
endfunction()

function(configure_pytorch_libraries)
  # Set PyTorch library paths (Linux only)
  set(TORCH_LIBRARIES ${torch_PATH}/lib/libtorch.so ${torch_PATH}/lib/libc10.so PARENT_SCOPE)

  # Set CXX flags
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE torch_cxx_abi
    RESULT_VARIABLE torch_abi_result
    ERROR_QUIET)
  set(TORCH_CXX_FLAGS -D_GLIBCXX_USE_CXX11_ABI=${torch_cxx_abi} PARENT_SCOPE)
endfunction()

function(configure_target_linking TARGET_NAME)
  target_link_libraries(${TARGET_NAME} PRIVATE nanobind hip::host ${TORCH_LIBRARIES})
endfunction()

function(apply_nanobind_optimizations TARGET_NAME)
  nanobind_opt_size(${TARGET_NAME})
  nanobind_lto(${TARGET_NAME})
  nanobind_set_visibility(${TARGET_NAME})
  nanobind_strip(${TARGET_NAME})
  nanobind_disable_stack_protector(${TARGET_NAME})
  nanobind_extension(${TARGET_NAME})

  # Apply CXX flags
  if(TORCH_CXX_FLAGS)
    nanobind_compile_options(${TARGET_NAME} PRIVATE ${TORCH_CXX_FLAGS})
  endif()

  nanobind_link_options(${TARGET_NAME})
endfunction()

# =============================================================================
# Main Build Logic
# =============================================================================

# Detect platform
detect_platform()

# Check dependencies
check_dependencies()

# Build if all dependencies are available
if(ALL_DEPS_AVAILABLE)
  message(STATUS "All dependencies found - proceeding with build")

  # Configure platform-specific settings
  configure_platform_specific()

  # Configure PyTorch
  include_directories(${torch_PATH}/include ${torch_PATH}/include/torch/csrc/api/include/)
  link_directories(${torch_PATH}/lib)
  configure_pytorch_libraries()

  # Build nanobind library
  nanobind_build_library(nanobind SHARED)

  # Create main target
  add_library(wave_runtime MODULE runtime.cpp)
  set(TORCH_LIBRARIES ${torch_PATH}/lib/libtorch.so ${torch_PATH}/lib/libc10.so)
  target_link_libraries(wave_runtime PRIVATE nanobind hip::host ${TORCH_LIBRARIES})
  set_target_properties(wave_runtime PROPERTIES LINK_WHAT_YOU_USE TRUE)

  # Configure linking and optimizations
  apply_nanobind_optimizations(wave_runtime)

  # Install target
  install(TARGETS wave_runtime DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

else()
  message(STATUS "Some dependencies are missing - skipping build")
endif()
