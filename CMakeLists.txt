cmake_minimum_required(VERSION 2.8)
include($ENV{ANS_ROOT}/qmake/quora_macros.cmake)

external_lib(glog glog)
external_lib(gflags gflags)
external_lib(third_party/gperftools gperftools)
external_lib(thrift thrift)

quora_lib(third_party/folly)
include_directories($ENV{ANS_ROOT}/third_party)

add_library(boosting_lib STATIC
  $ENV{ANS_ROOT}/boosting/Concurrency.cpp
  $ENV{ANS_ROOT}/boosting/DataSet.cpp
  $ENV{ANS_ROOT}/boosting/Config.cpp
  $ENV{ANS_ROOT}/boosting/Gbm.cpp
  $ENV{ANS_ROOT}/boosting/TreeRegressor.cpp)

target_link_libraries(boosting_lib
  glog
  gflags
  folly
  gperftools
  thrift
  rt
  pthread)

add_executable(train
  $ENV{ANS_ROOT}/boosting/Train.cpp)

target_link_libraries(train
  boosting_lib)
