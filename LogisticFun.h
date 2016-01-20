#pragma once

#include "GbmFun.h"

namespace boosting {

class LogisticFun : public GbmFun {
 public:
  double getLeafVal(const std::vector<int>& subset,
		    const boost::scoped_array<double>& y) const {
    double wx = 0.0, wy = 0.0;
    for (const auto& id : subset) {
      double yi = y[id];
      wy += yi;
      wx += fabs(yi) * (2.0 - fabs(yi));
    }
    return wy / wx;
  }

  double getF0(const std::vector<double>& y) const {
    double sumy = 0.0;
    for (const auto yi  : y) {
      sumy += yi;
    }
    double ybar = sumy/y.size();
    return 0.5 * log((1.0 + ybar)/(1.0 - ybar));
  }

  void getGradient(const std::vector<double>& y,
		   const boost::scoped_array<double>& F,
		   boost::scoped_array<double>& grad) const {
    int size = y.size();
    for (int i = 0; i < size; i++) {
      grad[i] = 2.0 * y[i]/(1.0 + exp(2.0 * y[i] * F[i]));
    }
  }

  double getInitLoss(const std::vector<double>& y) const {
    int posCount = 0;
    for (const auto yi : y) {
      if (yi > 0) {
	posCount += 1;
      }
    }
    return getEntropy(posCount, y.size()) * y.size();
  }

  double getExampleLoss(const double y, const double f) const {
    return log(1.0 + exp(-2.0 * y * f));
  }

  void accumulateExampleLoss(const double y, const double f) {
    numExamples_ += 1;
    if (y > 0) {
      posCount_ += 1;
    }
    logloss_ += getExampleLoss(y, f);
  }

  double getReduction() const {
    double entropy = getEntropy(posCount_, numExamples_);
    return 1.0 - logloss_/(entropy * numExamples_);
  }

  int getNumExamples() const {
    return numExamples_;
  }

  double getLoss() const {
    return logloss_;
  }

 private:
  static double getEntropy(int posCount, int numExamples) {
    double posProb = double(posCount)/numExamples;
    return -(posProb * log(posProb) + (1 - posProb) * log(1.0 - posProb));
  }
  
  int numExamples_;
  int posCount_;
  double logloss_;
};
}
