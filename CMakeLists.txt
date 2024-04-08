cmake_minimum_required(VERSION 3.4.1)

project(det_rk3588)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")

# skip 3rdparty lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/det_rk3588)

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# arch
set(LIB_ARCH aarch64)

include_directories(${CMAKE_SOURCE_DIR})

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/3rdparty/runtime/Linux/librknn_api)
set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty)

# opencv
find_package(OpenCV REQUIRED)

# rga
# comes from https://github.com/airockchip/librga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/3rdparty/rga/)
set(RGA_LIB ${RGA_PATH}/libs/Linux//gcc-${LIB_ARCH}/librga.so)
include_directories( ${RGA_PATH}/include)

# main executable
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(main
  src/main.cpp
  src/preprocess.cpp
  src/postprocess.cpp
)
target_link_libraries(main
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
)

add_executable(main_video
  src/main_video.cpp
  src/preprocess.cpp
  src/postprocess.cpp
  src/rknn_model.cpp
)
target_link_libraries(main_video
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
)

# install target and libraries
install(TARGETS main DESTINATION ./)
install(TARGETS main_video DESTINATION ./)

install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(DIRECTORY model/ DESTINATION ./model)
file(GLOB IMAGE_FILES "model/*.jpg")
file(GLOB LABEL_FILE "model/*.txt")
install(FILES ${IMAGE_FILES} DESTINATION ./model/)
install(FILES ${LABEL_FILE} DESTINATION ./model/)
