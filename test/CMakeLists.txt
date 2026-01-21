add_executable(TangentTests
  TestTangent.cpp
  TestTangentExampleProblem.cpp
  TestSlotMap.cpp
  TestSlotArray.cpp
  TestParallelAlgorithms.cpp
  TestEdgeCases.cpp
  TestVariableTypes.cpp
  TestMarginalization.cpp
  TestSSEOptimizer.cpp
)

target_link_libraries(TangentTests PRIVATE
  Tangent::Tangent
  GTest::gtest
  GTest::gtest_main
)

# Apply sanitizer flags if enabled
if(TANGENT_ENABLE_ASAN)
  target_compile_options(TangentTests PRIVATE -fsanitize=address -fno-omit-frame-pointer -g)
  target_link_options(TangentTests PRIVATE -fsanitize=address)
endif()

if(TANGENT_ENABLE_TSAN)
  target_compile_options(TangentTests PRIVATE -fsanitize=thread -g)
  target_link_options(TangentTests PRIVATE -fsanitize=thread)
endif()

if(TANGENT_ENABLE_UBSAN)
  target_compile_options(TangentTests PRIVATE -fsanitize=undefined -fno-omit-frame-pointer -g)
  target_link_options(TangentTests PRIVATE -fsanitize=undefined)
endif()

# Ensure MSVC uses the same runtime library as GoogleTest
if(MSVC)
  set_target_properties(TangentTests PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
  )
endif()

include(GoogleTest)
gtest_discover_tests(TangentTests)
