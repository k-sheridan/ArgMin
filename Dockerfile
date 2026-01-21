# Multi-stage Dockerfile for Tangent library
# Provides isolated build and test environment

# Build stage
FROM ubuntu:22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies and Tangent runtime dependencies
# Note: libsophus-dev not available in Ubuntu 22.04, will use FetchContent
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ninja-build \
    libssl-dev \
    libeigen3-dev \
    libspdlog-dev \
    libgtest-dev \
    libbenchmark-dev \
    && rm -rf /var/lib/apt/lists/*

# Install modern C++20 compiler (GCC 12)
RUN apt-get update && apt-get install -y \
    gcc-12 \
    g++-12 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /tangent

# Copy source files
COPY . .

# Configure CMake with all options enabled
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++-12 \
    -DTANGENT_BUILD_TESTS=ON \
    -DTANGENT_BUILD_BENCHMARKS=ON \
    -DTANGENT_USE_SPDLOG=ON

# Build everything
RUN cmake --build build --parallel $(nproc)

# Test stage
FROM builder AS test
WORKDIR /tangent/build
CMD ["ctest", "--output-on-failure", "--verbose"]

# Benchmark stage
FROM builder AS benchmark
WORKDIR /tangent/build
CMD ["./bench/TangentBenchmarks"]

# Development stage - includes all tools for interactive use
FROM builder AS dev

# Install additional development tools
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    clang-format \
    clang-tidy \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tangent

# Default to bash for interactive use
CMD ["/bin/bash"]

# Documentation stage - generates Sphinx + Breathe documentation
FROM ubuntu:22.04 AS docs

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    doxygen \
    graphviz \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Sphinx and dependencies
COPY docs/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /tangent

CMD ["sh", "-c", "doxygen Doxyfile && sphinx-build -b html docs docs/_build/html"]

# AddressSanitizer (ASAN) build stage - for memory safety testing
FROM ubuntu:22.04 AS builder-asan

ENV DEBIAN_FRONTEND=noninteractive

# Install same dependencies as builder
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ninja-build \
    libssl-dev \
    libeigen3-dev \
    libspdlog-dev \
    libgtest-dev \
    libbenchmark-dev \
    gcc-12 \
    g++-12 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tangent
COPY . .

# Configure with ASAN flags
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=g++-12 \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" \
    -DTANGENT_BUILD_TESTS=ON \
    -DTANGENT_BUILD_BENCHMARKS=ON \
    -DTANGENT_USE_SPDLOG=ON

RUN cmake --build build --parallel $(nproc)

# Set ASAN runtime options
ENV ASAN_OPTIONS=detect_leaks=1:check_initialization_order=1:strict_init_order=1

# ASAN test stage
FROM builder-asan AS test-asan
WORKDIR /tangent/build
CMD ["./test/TangentTests", "--gtest_color=yes"]

# ASAN benchmark stage
FROM builder-asan AS benchmark-asan
WORKDIR /tangent/build
CMD ["./bench/TangentBenchmarks", "--benchmark_repetitions=3"]

# ThreadSanitizer (TSAN) build stage - for data race detection
FROM ubuntu:22.04 AS builder-tsan

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ninja-build \
    libssl-dev \
    libeigen3-dev \
    libspdlog-dev \
    libgtest-dev \
    libbenchmark-dev \
    gcc-12 \
    g++-12 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tangent
COPY . .

# Configure with TSAN flags
# Note: Skip gtest_discover_tests during build (TSAN needs special runtime privileges)
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=g++-12 \
    -DCMAKE_CXX_FLAGS="-fsanitize=thread -g" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" \
    -DTANGENT_BUILD_TESTS=ON \
    -DTANGENT_BUILD_BENCHMARKS=ON \
    -DTANGENT_USE_SPDLOG=ON \
    -DCMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST

RUN cmake --build build --parallel $(nproc)

# Set TSAN runtime options
ENV TSAN_OPTIONS=second_deadlock_stack=1:history_size=7

# TSAN test stage
FROM builder-tsan AS test-tsan
WORKDIR /tangent/build
CMD ["./test/TangentTests", "--gtest_color=yes"]

# UndefinedBehaviorSanitizer (UBSAN) build stage - for UB detection
FROM ubuntu:22.04 AS builder-ubsan

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ninja-build \
    libssl-dev \
    libeigen3-dev \
    libspdlog-dev \
    libgtest-dev \
    libbenchmark-dev \
    gcc-12 \
    g++-12 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tangent
COPY . .

# Configure with UBSAN flags
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=g++-12 \
    -DCMAKE_CXX_FLAGS="-fsanitize=undefined -fno-omit-frame-pointer -g" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=undefined" \
    -DTANGENT_BUILD_TESTS=ON \
    -DTANGENT_BUILD_BENCHMARKS=ON \
    -DTANGENT_USE_SPDLOG=ON

RUN cmake --build build --parallel $(nproc)

# Set UBSAN runtime options
ENV UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1

# UBSAN test stage
FROM builder-ubsan AS test-ubsan
WORKDIR /tangent/build
CMD ["./test/TangentTests", "--gtest_color=yes"]

# UBSAN benchmark stage
FROM builder-ubsan AS benchmark-ubsan
WORKDIR /tangent/build
CMD ["./bench/TangentBenchmarks", "--benchmark_repetitions=3"]
