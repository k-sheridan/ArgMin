include(FetchContent)

# Eigen3 (required) - use git submodule
message(STATUS "Using Eigen3 from git submodule at extern/eigen")
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_BLAS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_LAPACK OFF CACHE BOOL "" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/extern/eigen)

# Sophus (required) - use git submodule
message(STATUS "Using Sophus from git submodule at extern/sophus")
set(BUILD_SOPHUS_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SOPHUS_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/extern/sophus)

# spdlog (optional)
if(TANGENT_USE_SPDLOG)
  find_package(spdlog QUIET)
  if(NOT spdlog_FOUND)
    message(STATUS "spdlog not found, fetching from source...")
    message(STATUS "Tip: Ubuntu/Debian: sudo apt install libspdlog-dev")
    message(STATUS "     macOS: brew install spdlog")
    FetchContent_Declare(spdlog
      GIT_REPOSITORY https://github.com/gabime/spdlog.git
      GIT_TAG v1.13.0
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(spdlog)
  else()
    message(STATUS "Found spdlog: ${spdlog_DIR}")
  endif()
endif()

# GoogleTest (for tests only)
if(TANGENT_BUILD_TESTS)
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    message(STATUS "GoogleTest not found, fetching from source...")
    message(STATUS "Tip: Ubuntu/Debian: sudo apt install libgtest-dev")
    message(STATUS "     macOS: brew install googletest")
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
      GIT_SHALLOW TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  else()
    message(STATUS "Found GoogleTest: ${GTest_DIR}")
  endif()
endif()

# Google Benchmark (for benchmarks only)
if(TANGENT_BUILD_BENCHMARKS)
  find_package(benchmark QUIET)
  if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found, fetching from source...")
    message(STATUS "Tip: Ubuntu 22.04+: sudo apt install libbenchmark-dev")
    message(STATUS "     macOS: brew install google-benchmark")
    FetchContent_Declare(googlebenchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG v1.8.3
      GIT_SHALLOW TRUE
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
  else()
    message(STATUS "Found Google Benchmark: ${benchmark_DIR}")
  endif()
endif()
