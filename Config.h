#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace boosting {

// Specifying the training parameters and data format
struct Config {

  bool readConfig(const std::string& fileName);

  int getNumFeatures() const {
    return trainIdx_.size();
  }

  int getNumTrees() const {
    return numTrees_;
  }

  int getNumLeaves() const {
    return numLeaves_;
  }

  double getLearningRate() const {
    return learningRate_;
  }

  double getExampleSamplingRate() const {
    return exampleSamplingRate_;
  }

  double getFeatureSamplingRate() const {
    return featureSamplingRate_;
  }

  int getTargetIdx() const {
    return targetIdx_;
  }

  int getCompareIdx() const {
    return cmpIdx_;
  }

  const std::vector<int>& getTrainIdx() const {
    return trainIdx_;
  }

  bool isWeakFeature(const int fidx) const {
    return (std::find(weakIdx_.begin(), weakIdx_.end(), trainIdx_[fidx])
            != weakIdx_.end());
  }

  const std::string& getFeatureName(const int fidx) const {
    return allColumns_[trainIdx_[fidx]];
  }

  const std::vector<int>& getWeakIdx() const {
    return weakIdx_;
  }

  const std::vector<std::string>& getColumnNames() const {
    return allColumns_;
  }

  char getDelimiter() const {
    return delimiter_;
  }

 private:

  int numTrees_;
  int numLeaves_;
  double exampleSamplingRate_;
  double featureSamplingRate_;
  double learningRate_;

  int targetIdx_;
  int cmpIdx_;
  std::vector<int> trainIdx_;
  std::vector<int> weakIdx_;

  std::vector<std::string> allColumns_;
  char delimiter_;
};

}
