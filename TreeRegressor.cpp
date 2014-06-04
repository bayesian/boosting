#include "boosting/TreeRegressor.h"

#include <cstdlib>

#include "boosting/Tree.h"
#include "boosting/GbmFun.h"
#include "boosting/DataSet.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_int32(min_leaf_examples, 256,
             "minimum number of data points in the leaf");

namespace boosting {

using namespace std;

inline double rand01() {
  return ((double)rand()/(double)RAND_MAX);
}

TreeRegressor::SplitNode::SplitNode(const vector<int>* st):
  subset(st), fid(-1), fv(0), gain(0), selected(false),
  left(NULL), right(NULL) {
}

TreeRegressor::TreeRegressor(
  const DataSet& ds,
  const boost::scoped_array<double>& y,
  const GbmFun& fun) : ds_(ds), y_(y), fun_(fun) {
}

TreeRegressor::~TreeRegressor() {
  for (SplitNode* split : allSplits_) {
    delete split;
  }
}

void TreeRegressor::splitExamples(
  const SplitNode& split,
  vector<int>* left,
  vector<int>* right) {

  const int fid = split.fid;
  const uint16_t fv = split.fv;

  auto &f = ds_.features_[fid];

  if (f.encoding == BYTE) {
    boosting::split<uint8_t>(*(split.subset), left, right, *(f.bvec), fv);
  } else {
    CHECK(f.encoding == SHORT);
    boosting::split<uint16_t>(*(split.subset), left, right, *(f.svec), fv);
  }
}

void TreeRegressor::getBestSplitFromHistogram(
  const TreeRegressor::Histogram& hist,
  int* idx,
  double* gain) {

  double lossBefore = -1.0 * hist.totalSum * hist.totalSum / hist.totalCnt;

  int cntLeft = 0;
  double sumLeft = 0.0;

  double bestGain = 0.0;
  int bestIdx = -1;

  for (int i = 0; i < hist.num - 1; i++) {

    cntLeft += hist.cnt[i];
    sumLeft += hist.sumy[i];

    double sumRight = hist.totalSum - sumLeft;
    int cntRight = hist.totalCnt - cntLeft;

    if (cntLeft < FLAGS_min_leaf_examples || cntRight < FLAGS_min_leaf_examples) {
        continue;
    }

    double lossAfter = -1.0 * sumLeft * sumLeft / cntLeft;
    if (cntRight != 0) {
      lossAfter += - 1.0 * sumRight * sumRight / cntRight;
    }

    double gain = lossBefore - lossAfter;
    if (gain > bestGain) {
      bestGain = gain;
      bestIdx = i;
    }
  }

  *idx = bestIdx;
  *gain = bestGain;
}

TreeRegressor::SplitNode*
TreeRegressor::getBestSplit(const vector<int>* subset,
                            double featureSamplingRate,
                            bool terminal) {

  SplitNode* split = new SplitNode(subset);
  if (terminal) {
    allSplits_.push_back(split);
    return split;
  }

  int bestFid = -1;
  int bestFv = 0;
  double bestGain = 0.0;
  double totalSum = 0.0;

  for (auto& id : *subset) {
    totalSum += y_[id];
  }

  for (int fid = 0; fid < ds_.numFeatures_; fid++) {
    const auto& f = ds_.features_[fid];

    if (f.encoding == EMPTY || rand01() < featureSamplingRate) {
      continue;
    }

    Histogram hist(f.transitions.size() + 1, subset->size(), totalSum);

    if (f.encoding == BYTE) {
      buildHistogram<uint8_t>(*subset, *(f.bvec), hist);
    } else {
      CHECK(f.encoding == SHORT);
      buildHistogram<uint16_t>(*subset, *(f.svec), hist);
    }

    int fv;
    double gain;
    getBestSplitFromHistogram(hist, &fv, &gain);

    if (gain > bestGain) {
      bestFid = fid;
      bestGain = gain;
      bestFv = fv;
    }
  }
  split->fid = bestFid;
  split->fv = bestFv;
  split->gain = bestGain;

  frontiers_.push_back(split);
  allSplits_.push_back(split);
  return split;
}


TreeNode<uint16_t>* TreeRegressor::getTree(
  const int numLeaves,
  const double exampleSamplingRate,
  const double featureSamplingRate,
  double fimps[]) {

  vector<int>* subset = new vector<int>();
  for (int i = 0; i < ds_.getNumExamples(); i++) {
    if (rand01() < exampleSamplingRate) {
      subset->push_back(i);
    }
  }

  SplitNode* root = getBestSplits(subset, numLeaves - 1, featureSamplingRate);
  return getTreeHelper(root, fimps);
}

TreeNode<uint16_t>* TreeRegressor::getTreeHelper(
  SplitNode* split,
  double fimps[]) {

  if (split == NULL) {
    return NULL;
  } else if (!split->selected) {
    double fvote = fun_.getLeafVal(*(split->subset), y_);
    LOG(INFO) << "leaf:  " << fvote << ", #examples:"
              << split->subset->size();

    return new LeafNode<uint16_t>(fvote);

  } else {
    LOG(INFO) << "select split: " << split->fid << ":" << split->fv
              << " gain: " << split->gain << ", #examples:"
              << split->subset->size() << ", min partition: "
              << std::min(split->left->subset->size(), split->right->subset->size());

    fimps[split->fid] += split->gain;

    PartitionNode<uint16_t>* node = new PartitionNode<uint16_t>(split->fid, split->fv);
    node->setLeft(getTreeHelper(split->left, fimps));
    node->setRight(getTreeHelper(split->right, fimps));

    return node;
  }
}

TreeRegressor::SplitNode* TreeRegressor::getBestSplits(
  const vector<int>* subset, const int k, double featureSamplingRate) {

  CHECK(subset != NULL);

  SplitNode* firstSplit = getBestSplit(subset, featureSamplingRate, false);
  int numSelected = 0;

  do {
    double bestGain = 0;
    vector<SplitNode*>::iterator best_it;

    for (auto it = frontiers_.begin(); it != frontiers_.end(); it++) {
      if ((*it)->gain > bestGain) {
        bestGain = (*it)->gain;
        best_it = it;
      }
    }

    (*best_it)->selected = true;
    numSelected++;

    SplitNode* bestSplit = *best_it;
    frontiers_.erase(best_it);

    vector<int>* left = new vector<int>();
    vector<int>* right = new vector<int>();

    splitExamples(*bestSplit, left, right);
    bool terminal = (numSelected == k);

    bestSplit->left = getBestSplit(left, featureSamplingRate, terminal);
    bestSplit->right = getBestSplit(right, featureSamplingRate, terminal);
  } while (numSelected < k);

  return firstSplit;
}

}

