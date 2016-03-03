/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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
