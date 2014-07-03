cmake_minimum_required (VERSION 3.0)
project("Gbm Training")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -O2")

include_directories("/usr/local/include")

ADD_LIBRARY(folly STATIC IMPORTED)
set_property(TARGET folly PROPERTY IMPORTED_LOCATION /usr/local/lib/libfolly.a)

ADD_LIBRARY(thrift STATIC IMPORTED)
set_property(TARGET thrift PROPERTY IMPORTED_LOCATION /usr/local/lib/libthrift.a)

set(gflags_SHARED    FALSE)
set(gflags_NOTHREADS FALSE)
find_package(gflags REQUIRED)

ADD_LIBRARY(glog STATIC IMPORTED)
set_property(TARGET glog PROPERTY IMPORTED_LOCATION /usr/local/lib/libglog.a)

ADD_LIBRARY(double-conversion STATIC IMPORTED)
set_property(TARGET double-conversion PROPERTY IMPORTED_LOCATION /usr/local/lib/libdouble-conversion.a)

add_executable(train
   Concurrency.cpp
   Config.cpp
   DataSet.cpp
   Gbm.cpp
   Train.cpp
   TreeRegressor.cpp)

target_link_libraries(train
     pthread
     double-conversion
     folly
     thrift
     gflags
     glog)
