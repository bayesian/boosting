#include "Concurrency.h"

DEFINE_int32(num_threads, 0,
             "number of threads to use in loading & evaluation");

using namespace apache::thrift::concurrency;

namespace boosting {

boost::shared_ptr<ThreadManager>
  Concurrency::threadManager =
    boost::shared_ptr<ThreadManager>(NULL);

void Concurrency::initThreadManager() {
  if (FLAGS_num_threads > 0) {
    threadManager = ThreadManager::newSimpleThreadManager(FLAGS_num_threads);
    threadManager->threadFactory(
      boost::shared_ptr<ThreadFactory>(
        new PosixThreadFactory));
    threadManager->start();
  }
}

void CounterMonitor::init(int size) {
  counter_ = size;
}

void CounterMonitor::decrement() {
  if (atomic_fetch_sub(&counter_, 1) == 1) {
    monitor_.notifyAll();
  }
}

int CounterMonitor::wait() {
  return monitor_.waitForever();
}

};
