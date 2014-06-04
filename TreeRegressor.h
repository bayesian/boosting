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

  TreeNode<uint16_t>* getTree(
    const int numLeaves,
    const double exampleSamplingRate,
    const double featureSamplingRate,
    double fimps[]);

  ~TreeRegressor();

 private:

  struct SplitNode {

    explicit SplitNode(const std::vector<int>* subset);

    const std::vector<int>* subset;
    int fid;
    uint16_t fv;
    double gain;
    bool selected;

    SplitNode* left;
    SplitNode* right;

    ~SplitNode() {
      delete subset;
    }
  };

  struct Histogram {
    const int num;
    std::vector<int> cnt;
    std::vector<double> sumy;
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
                        Histogram& hist);

  static void getBestSplitFromHistogram(
    const TreeRegressor::Histogram& hist,
    int* idx,
    double* gain);

  // upon finish, also push to working queues
  SplitNode* getBestSplit(const std::vector<int>* subset,
                          double featureSamplingRate,
                          bool terminal);

  void splitExamples(const SplitNode& split,
                     std::vector<int>* left,
                     std::vector<int>* right);

  SplitNode* getBestSplits(const std::vector<int>* subset,
                           const int k,
                           double featureSamplingRate);

  TreeNode<uint16_t>* getTreeHelper(SplitNode* root, double fimps[]);

  const DataSet& ds_;
  const boost::scoped_array<double>& y_;
  const GbmFun& fun_;

  // working queue to select best K splits
  std::vector<SplitNode*> frontiers_;

  // memory management
  std::vector<SplitNode*> allSplits_;

};

template<class T>
  void TreeRegressor::buildHistogram(const std::vector<int>& subset,
                                     const std::vector<T>& fvec,
                                     Histogram& hist) {

  for(auto id : subset) {
    const T& v = fvec[id];

    hist.cnt[v] += 1;
    hist.sumy[v] += y_[id];
  }
}

}
