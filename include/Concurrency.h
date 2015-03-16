#pragma once

#include <atomic>
#include "thrift/concurrency/Monitor.h"
#include "thrift/concurrency/PosixThreadFactory.h"
#include "thrift/concurrency/ThreadManager.h"
#include "gflags/gflags.h"

DECLARE_int32(num_threads);

namespace boosting {

class Concurrency {

 public:

  static boost::shared_ptr<apache::thrift::concurrency::ThreadManager>
    threadManager;

  static void initThreadManager();

};

class CounterMonitor {

 public:

  explicit CounterMonitor(int n) : counter_(n) {}

  void init(int size);

  void decrement();

  int wait();

 private:

  apache::thrift::concurrency::Monitor monitor_;

  std::atomic<int> counter_;

};

}
