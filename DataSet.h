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
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "glog/logging.h"

namespace boosting {

class Config;

enum FeatureEncoding {
  EMPTY   = 0,
  BYTE    = 1,
  SHORT   = 2,
  DOUBLE  = 3
};

// different representation of a single feature vec
// compressed to byte/short for significant memory saving
// and much faster splits
struct FeatureData {
  std::vector<double> transitions;
  FeatureEncoding encoding;
  std::unique_ptr<std::vector<uint8_t>> bvec;
  std::unique_ptr<std::vector<uint16_t>> svec;
  std::unique_ptr<std::vector<double>> fvec;

  void shrink_to_fit() {
    if (encoding == BYTE) {
      bvec->shrink_to_fit();
    } else if (encoding == SHORT) {
      svec->shrink_to_fit();
    } else if (encoding == DOUBLE) {
      fvec->shrink_to_fit();
    }
    return;
  }
};

template<class T> class TreeNode;

// in memory representation of raw data read from a list of data
// files, then intelligently compress the data into the format
// suitable for the boosting training process
class DataSet {
 public:
  DataSet(const Config& cfg, int bucketingThresh, int examplesThresh=-1);

  bool addVector(const boost::scoped_array<double>& fvec, double target);

  bool getRow(const std::string& line,
              double* target,
              boost::scoped_array<double>& fvec,
              double* cmpValue = NULL) const;

  bool getEvalColumns(const std::string& line,
		      boost::scoped_array<std::string>& feval) const;

  int getNumExamples() const {
    return numExamples_;
  }

  void getFeatureVec(const int eid, boost::scoped_array<uint16_t>& fvec) const {
    for (int i = 0; i < numFeatures_; i++) {
      if (features_[i].encoding == EMPTY) {
        fvec[i] = 0;
      } else if (features_[i].encoding == BYTE) {
        fvec[i] = (*features_[i].bvec)[eid];
      } else if (features_[i].encoding == SHORT) {
        fvec[i] = (*features_[i].svec)[eid];
      } else {
        CHECK(false) << "invalid types";
      }
    }
  }

  double getPrediction(TreeNode<uint16_t>* tree, int eid) const;

  void close() {
    bucketize();
    for (int i = 0; i < numFeatures_; i++) {
      auto &f = features_[i];
      f.shrink_to_fit();
    }

    targets_.shrink_to_fit();
  }

 private:
  void bucketize();

  const Config& cfg_;
  const int bucketingThresh_;
  const int examplesThresh_;

  //state of data loading process
  bool preBucketing_;
  int numExamples_;
  int numFeatures_;

  boost::scoped_array<FeatureData> features_;
  std::vector<double> targets_;

  friend class TreeRegressor;
  friend class Gbm;
};

// partition subset into left and right, depending
// on how the values of fvec compare to fv
template<class T> void split(const std::vector<int>& subset,
                             std::vector<int>* left,
                             std::vector<int>* right,
                             const std::vector<T>& fvec,
                             uint16_t fv) {

  for (auto id : subset) {
    if (fvec[id] <= fv) {
      left->push_back(id);
    } else {
      right->push_back(id);
    }
  }
}

}
