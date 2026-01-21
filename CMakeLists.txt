cmake_minimum_required(VERSION 3.20)
project(Tangent VERSION 1.0.0 LANGUAGES CXX)

# Project configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(TANGENT_BUILD_TESTS "Build tests" ON)
option(TANGENT_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(TANGENT_USE_SPDLOG "Enable spdlog logging" OFF)
option(TANGENT_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(TANGENT_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(TANGENT_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

# Validate sanitizer options (ASAN and TSAN are incompatible)
if(TANGENT_ENABLE_ASAN AND TANGENT_ENABLE_TSAN)
  message(FATAL_ERROR "ASAN and TSAN cannot be enabled simultaneously")
endif()

# Create header-only library target
add_library(Tangent INTERFACE)
add_library(Tangent::Tangent ALIAS Tangent)

target_include_directories(Tangent INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(Tangent INTERFACE cxx_std_20)

# Dependencies
include(cmake/Dependencies.cmake)

# Link dependencies (use $<BUILD_INTERFACE> to avoid export issues with FetchContent)
target_link_libraries(Tangent INTERFACE
  $<BUILD_INTERFACE:Eigen3::Eigen>
  $<BUILD_INTERFACE:Sophus::Sophus>
)

if(TANGENT_USE_SPDLOG)
  target_compile_definitions(Tangent INTERFACE TANGENT_USE_SPDLOG)
  target_link_libraries(Tangent INTERFACE $<BUILD_INTERFACE:spdlog::spdlog>)
endif()

# Testing
if(TANGENT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

# Benchmarks
if(TANGENT_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

# Installation
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS Tangent EXPORT TangentTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT TangentTargets
  FILE TangentTargets.cmake
  NAMESPACE Tangent::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Tangent
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/TangentConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "cmake/TangentConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/TangentConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Tangent
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/TangentConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/TangentConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Tangent
)
