/* Copyright 2015,2016 Tao Xu
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

#include <boost/scoped_array.hpp>
#include <vector>

namespace boosting {

// Implementing a few simple function could extend Gbm to different
// loss functions, like l2 loss (least square), logloss (logistic
// regression), huber loss (robust regression), lambdaloss (lambda
// rank), etc.
class GbmFun {
 public:
  virtual double getLeafVal(const std::vector<int>& subset,
                            const boost::scoped_array<double>& y) const = 0;

  virtual double getF0(const std::vector<double>& y) const = 0;

  virtual void getGradient(const std::vector<double>& y,
                           const boost::scoped_array<double>& F,
                           boost::scoped_array<double>& grad) const = 0;

  virtual double getInitLoss(const std::vector<double>& y) const = 0;

  virtual double getExampleLoss(const double y, const double f) const = 0;

  virtual void accumulateExampleLoss(const double y, const double f) = 0;

  virtual double getReduction() const = 0;

  virtual int getNumExamples() const = 0;

  virtual double getLoss() const = 0;
};


class LeastSquareFun : public GbmFun {
 public:
  LeastSquareFun() : numExamples_(0), sumy_(0.0), sumy2_(0.0), l2_(0.0) {
  }

  double getLeafVal(const std::vector<int>& subset,
                    const boost::scoped_array<double>& y) const {

    double sum = 0;
    for (const auto& id : subset) {
      sum += y[id];
    }
    return sum/subset.size();
  }

  double getF0(const std::vector<double>& yvec) const {
    double sum = 0.0;
    for (const auto& y : yvec) {
      sum += y;
    }
    return sum/yvec.size();
  }

  void getGradient(const std::vector<double>& y,
                   const boost::scoped_array<double>& F,
                   boost::scoped_array<double>& grad) const {

    int size = y.size();

    for (int i = 0; i < size; i++) {
      grad[i] = y[i] - F[i];
    }
  }

  double getInitLoss(const std::vector<double>& yvec) const {
    double sumy = 0.0;
    double sumy2 = 0.0;

    for (const auto& y : yvec) {
      sumy += y;
      sumy2 += y*y;
    }

    return sumy2 - sumy * sumy/yvec.size();
  }

  double getExampleLoss(const double y, const double f) const {
    return (y - f) * (y - f);
  }

  void accumulateExampleLoss(const double y, const double f) {
    sumy_ += y;
    numExamples_ += 1;
    sumy2_ += y * y;
    l2_ += getExampleLoss(y, f);
  }

  double getReduction() const {
    return 1.0 - l2_/(sumy2_ - sumy_ * sumy_/numExamples_);
  }

  int getNumExamples() const {
    return numExamples_;
  }

  double getLoss() const {
    return l2_;
  }

 private:
  int numExamples_;
  double sumy_;
  double sumy2_;
  double l2_;
};

}
