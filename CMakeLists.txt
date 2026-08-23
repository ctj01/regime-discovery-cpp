cmake_minimum_required(VERSION 3.20)
project(market_regime_detector CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(mrd_core STATIC
  src/csv_loader.cpp
  src/features.cpp
  src/standardize.cpp
  src/brute_index.cpp
  src/hnsw_index.cpp
  src/neighbor_graph.cpp
  src/hdbscan.cpp
  src/regime_validate.cpp
)
target_include_directories(mrd_core PUBLIC include)
target_compile_options(mrd_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
add_library(mrd::core ALIAS mrd_core)

add_executable(mrd src/main.cpp)
target_link_libraries(mrd PRIVATE mrd::core)
target_compile_options(mrd PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(mrd_tests
  tests/features_test.cpp
  tests/standardize_test.cpp
  tests/nn_index_test.cpp
  tests/regime_test.cpp
)
target_link_libraries(mrd_tests PRIVATE mrd::core)
target_include_directories(mrd_tests PRIVATE third_party)
target_compile_options(mrd_tests PRIVATE -Wall -Wextra)
add_test(NAME mrd_tests COMMAND mrd_tests)
