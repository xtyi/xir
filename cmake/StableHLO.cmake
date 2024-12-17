set(STABLEHLO_ENABLE_LLD ON)

add_subdirectory(
  ${CMAKE_SOURCE_DIR}/externals/stablehlo
  ${CMAKE_BINARY_DIR}/stablehlo
  EXCLUDE_FROM_ALL)

include_directories(${CMAKE_SOURCE_DIR}/externals/stablehlo)
include_directories(${CMAKE_BINARY_DIR}/stablehlo)

message(STATUS ${CMAKE_SOURCE_DIR}/externals/stablehlo)
message(STATUS ${CMAKE_BINARY_DIR}/stablehlo)
