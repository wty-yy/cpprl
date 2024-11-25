cmake_minimum_required(VERSION 3.10)
project(env_test VERSION 1.0)


set(EXECUTABLES
    # test_ale tests/test_ale.cpp
    # test_fmt tests/test_fmt.cpp
    test_ale_vec_env tests/test_ale_vec_env.cpp
    # test tests/test.cpp
)

list(LENGTH EXECUTABLES EXECUTABLES_LENGTH)
math(EXPR NUM_TARGETS_MINUS_ONE "${EXECUTABLES_LENGTH} / 2 - 1")

foreach(i RANGE 0 ${NUM_TARGETS_MINUS_ONE})
  math(EXPR NAME_INDEX "${i} * 2")
  math(EXPR FILE_INDEX "${i} * 2 + 1")

  list(GET EXECUTABLES ${NAME_INDEX} EXEC_NAME)
  list(GET EXECUTABLES ${FILE_INDEX} SRC_FILE)

  add_executable(${EXEC_NAME} ${SRC_FILE})

  target_include_directories(${EXEC_NAME} PUBLIC env)

  ### Torch ###
  find_package(Torch REQUIRED)
  target_link_libraries(${EXEC_NAME} PUBLIC "${TORCH_LIBRARIES}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  set_property(TARGET ${EXEC_NAME} PROPERTY CXX_STANDARD 17)


  ### Tensorboard ###
  find_package(tensorboard_logger REQUIRED)
  target_link_libraries(${EXEC_NAME} PUBLIC ${TENSORBOARD_LOGGER_LIBRARIES} protobuf)

  ### Env Render ###
  # find_package(SDL2 REQUIRED)
  # target_link_libraries(${EXEC_NAME} PUBLIC ${SDL2_LIBRARIES})
  # target_link_libraries(${EXEC_NAME} PUBLIC sfml-graphics sfml-window sfml-system)

  ### ALE ###
  find_package(ale REQUIRED)
  target_link_libraries(${EXEC_NAME} PUBLIC ale::ale-lib)

  ### fmt ###
  find_package(fmt REQUIRED)
  target_link_libraries(${EXEC_NAME} PUBLIC fmt::fmt)

  add_compile_options(-O3)
  
  ### OpenCV2 ###
  find_package(OpenCV 4 REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${EXEC_NAME} PUBLIC ${OpenCV_LIBS})


endforeach()