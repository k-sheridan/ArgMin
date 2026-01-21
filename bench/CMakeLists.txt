add_executable(TangentBenchmarks
  MainBenchmark.cpp
  SlotMapBenchmark.cpp
  SparseBlockMatrixBenchmark.cpp
)

target_link_libraries(TangentBenchmarks PRIVATE
  Tangent::Tangent
  benchmark::benchmark
  benchmark::benchmark_main
)

# Apply sanitizer flags if enabled
if(TANGENT_ENABLE_ASAN)
  target_compile_options(TangentBenchmarks PRIVATE -fsanitize=address -fno-omit-frame-pointer -g)
  target_link_options(TangentBenchmarks PRIVATE -fsanitize=address)
endif()

if(TANGENT_ENABLE_TSAN)
  target_compile_options(TangentBenchmarks PRIVATE -fsanitize=thread -g)
  target_link_options(TangentBenchmarks PRIVATE -fsanitize=thread)
endif()

if(TANGENT_ENABLE_UBSAN)
  target_compile_options(TangentBenchmarks PRIVATE -fsanitize=undefined -fno-omit-frame-pointer -g)
  target_link_options(TangentBenchmarks PRIVATE -fsanitize=undefined)
endif()

# Ensure MSVC uses the same runtime library
if(MSVC)
  set_target_properties(TangentBenchmarks PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
  )
endif()
