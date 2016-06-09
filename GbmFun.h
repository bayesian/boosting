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

#include <boost/scoped_array.hpp>
#include <vector>
#include "glog/logging.h"

namespace boosting {

// Implementing a few simple function could extend Gbm to different
// loss functions, like l2 loss (least square), logloss (logistic
// regression), huber loss (robust regression), lambdaloss (lambda
// rank), etc.
class GbmFun {
 public:
  virtual double getLeafVal(const std::vector<int>& subset,
                            const boost::scoped_array<double>& y,
                            const std::vector<double>* wts = NULL) const = 0;

  virtual double getF0(const std::vector<double>& y,
                       const std::vector<double>* wts = NULL) const = 0;

  virtual void getGradient(const std::vector<double>& y,
                           const boost::scoped_array<double>& F,
                           boost::scoped_array<double>& grad,
                           const std::vector<double>* wts = NULL) const = 0;

  virtual double getInitLoss(const std::vector<double>& y,
                             const std::vector<double>* wts = NULL) const = 0;

  virtual double getExampleLoss(const double y, const double f, const double w) const = 0;

  virtual void accumulateExampleLoss(const double y, const double f, const double w) = 0;

  virtual double getReduction() const = 0;

  virtual int getNumExamples() const = 0;

  virtual double getLoss() const = 0;
};


class LeastSquareFun : public GbmFun {
 public:
  LeastSquareFun() : numExamples_(0), sumy_(0.0), sumy2_(0.0), l2_(0.0), sumw_(0.0) {
  }

  double getLeafVal(const std::vector<int>& subset,
                    const boost::scoped_array<double>& y, const std::vector<double>* wts = NULL) const {
    double sumwy = 0;
    double sumw = 0;
    for (const auto& id : subset) {
      double w = ((wts != NULL) ? (*wts)[id] : 1.0);
      sumw += w;
      sumwy += w * y[id];
    }
    return sumwy/sumw;
  }

  double getF0(const std::vector<double>& yvec, const std::vector<double>* wts = NULL) const {
    double sumwy = 0;
    double sumw = 0;
    for (int i = 0; i < yvec.size(); i++) {
      double w = ((wts != NULL) ? (*wts)[i] : 1.0);
      sumw += w;
      sumwy += w * yvec[i];
    }
    return sumwy/sumw;
  }

  void getGradient(const std::vector<double>& y,
                   const boost::scoped_array<double>& F,
                   boost::scoped_array<double>& grad,
                   const std::vector<double>* wts = NULL) const {

    int size = y.size();

    for (int i = 0; i < size; i++) {
      grad[i] = y[i] - F[i];
    }
  }

  double getInitLoss(const std::vector<double>& yvec,
                     const std::vector<double>* wts = NULL) const {

    double sumy = 0.0;
    double sumy2 = 0.0;
    double sumw = 0.0;

    for (int i = 0; i < yvec.size(); i++) {
      double w = ((wts != NULL) ? (*wts)[i] : 1.0);
      double y = yvec[i];

      sumw += w;
      sumy += w*y;
      sumy2 += w*y*y;
    }

    return sumy2 - sumy * sumy/sumw;
  }

  double getExampleLoss(const double y, const double f, const double w) const {
    return w * (y - f) * (y - f);
  }

  void accumulateExampleLoss(const double y, const double f, const double w) {
    sumy_ += w * y;
    numExamples_ += 1;
    sumw_ += w;
    sumy2_ += w * y * y;

    l2_ += getExampleLoss(y, f, w);
  }

  double getReduction() const {
    return 1.0 - l2_/(sumy2_ - sumy_ * sumy_/sumw_);
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
  double sumw_;
};

}
