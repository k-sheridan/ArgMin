Getting Started
===============

Installation
------------

Add ArgMin to your CMake project using FetchContent:

.. code-block:: cmake

   include(FetchContent)
   FetchContent_Declare(argmin
     GIT_REPOSITORY https://github.com/k-sheridan/ArgMin.git
     GIT_TAG main
   )
   FetchContent_MakeAvailable(argmin)
   target_link_libraries(your_target PRIVATE ArgMin::ArgMin)


Requirements
------------

- C++20 compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.20+
- Eigen 3.4+ (auto-fetched if not found)
- Sophus 1.22+ (auto-fetched if not found)


Testing & Benchmarking
----------------------

Using Docker:

.. code-block:: bash

   docker-compose up test       # Run all ArgMin tests
   docker-compose up benchmark  # Run performance benchmarks


Basic Usage
-----------

Refer to `test/TestArgMinExampleProblem.cpp <https://github.com/k-sheridan/ArgMin/blob/master/test/TestArgMinExampleProblem.cpp>`_
for a complete example on how to use the optimizer.
