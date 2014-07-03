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
    const int num;             // number of buckets
    std::vector<int> cnt;      // number of observations in each bucket
    std::vector<double> sumy;  // sum of y-values of those observations
    const int totalCnt;
    const double totalSum;

    Histogram(int n, int cnt, double sum)
    : num(n),
      cnt(num, 0),
      sumy(num, 0.0),
      totalCnt(cnt),
      totalSum(sum) {
    }
  };

  template<class T>
    void buildHistogram(const std::vector<int>& subset,
                        const std::vector<T>& fvec,
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
                                     Histogram& hist) const {

  for(auto id : subset) {
    const T& v = fvec[id];

    hist.cnt[v] += 1;
    hist.sumy[v] += y_[id];
  }
}

}
