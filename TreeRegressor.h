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

#include <cstdint>
#include <vector>
#include <boost/scoped_array.hpp>

namespace boosting {

class DataSet;
template<class T> class TreeNode;
class GbmFun;

// Build regression trees from DataSet
class TreeRegressor {
 public:
  TreeRegressor(const DataSet& ds,
                const boost::scoped_array<double>& y,
                const GbmFun& fun);

  // Return the root of a regression tree with desired specifications, based on
  // a random sampling of the data in ds_ and a random sampling of the features.
  // Also set the feature importance vector (fimps) = total gains from
  // splitting along each feature (most entries will be 0).
  TreeNode<uint16_t>* getTree(
    const int numLeaves,
    const double exampleSamplingRate,
    const double featureSamplingRate,
    double fimps[]);

  ~TreeRegressor();

 private:

  // Node in a binary regression tree, computed based on a sampling of the data
  // (given by subset); responsible for cleaning up subset upon destruction
  struct SplitNode {

    explicit SplitNode(const std::vector<int>* subset);

    const std::vector<int>* subset;  // which subset of the data we're using
    int fid;        // which feature to split along
    uint16_t fv;    // value of said feature, at which to split
    double gain;    // gain in prediction accuracy from this split
    bool selected;  // internal node of regression tree, as opposed to leaf

    SplitNode* left;   // left child in a regression tree
    SplitNode* right;  // right child in a regression tree

    ~SplitNode() {
      delete subset;
    }
  };

  // More than a histogram in the basic sense of the word, because our
  // data has two dimensions. Make buckets based on the x-dimension,
  // and within each bucket keep track of not only the number of
  // observations (as in a basic histogram), but also the sum of y-values
  // of those observations.
  struct Histogram {
    const int num;                   // number of buckets
    std::vector<double> weight;      // number of observations in each bucket
    std::vector<double> sumwy;       // sum of y-values of those observations
    const int totalWeight;
    const double totalWeightedSum;

    Histogram(int n, double sumw, double sumwy)
    : num(n),
      weight(num, 0.0),
      sumwy(num, 0.0),
      totalWeight(sumw),
      totalWeightedSum(sumwy) {
    }
  };

  template<class T>
    void buildHistogram(const std::vector<int>& subset,
                        const std::vector<T>& fvec,
			const std::vector<double>* weights,
                        Histogram& hist) const;

  // Choose the x-value such that, by splitting the data at that value, we
  // minimize the total sum-of-squares error
  static void getBestSplitFromHistogram(
    const TreeRegressor::Histogram& hist,
    int* idx,
    double* gain);

  // Based on a sampling of the data (given by *subset) and a random sampling
  // of features (given by featureSamplingRate), find a splitting that maximizes
  // prediction accuracy, unless terminal==true, in which case just return a
  // sentry.
  // Upon finish, also push to working queues (frontiers_ and allSplits_)
  SplitNode* getBestSplit(const std::vector<int>* subset,
                          double featureSamplingRate,
                          bool terminal);

  // Partition split.subset into left and right according to the splitting
  // specified by split.fid and split.fv
  void splitExamples(const SplitNode& split,
                     std::vector<int>* left,
                     std::vector<int>* right);

  // Return root of a regression tree for data in subset with numSplits internal
  // nodes (i.e., numSplits+1 leaves) by greedily selecting the splits with the
  // biggest gain.
  SplitNode* getBestSplits(const std::vector<int>* subset,
                           const int numSplits,
                           double featureSamplingRate);

  // Recursively construct a tree of ParitionNode's and LeafNode's from
  // a tree of SplitNode's. The point is that SplitNode's carry some working
  // data (e.g., about which data points belong to them) that we should
  // throw away when the computation is finished.
  TreeNode<uint16_t>* getTreeHelper(SplitNode* root, double fimps[]);

  const DataSet& ds_;
  const boost::scoped_array<double>& y_;
  const GbmFun& fun_;

  // working queue to select best numSplits splits
  // could replace with priority queue if necessary
  std::vector<SplitNode*> frontiers_;

  // memory management, to delete SplitNode's upon destruction
  std::vector<SplitNode*> allSplits_;

};

template<class T>
  void TreeRegressor::buildHistogram(const std::vector<int>& subset,
                                     const std::vector<T>& fvec,
				     const std::vector<double>* weights,
                                     Histogram& hist) const {

  if (weights == NULL) {
    for(auto id : subset) {
      const T& v = fvec[id];

      hist.weight[v] += 1.0;
      hist.sumwy[v] += y_[id];
    }
  } else {
    for(auto id : subset) {
      const T& v = fvec[id];
      double w = (*weights)[id];

      hist.weight[v] += w;
      hist.sumwy[v] += w * y_[id];
    }
  }
}
}
