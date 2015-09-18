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
			    const std::vector<double>& pos,
                            const boost::scoped_array<double>& y) const = 0;

  virtual double getF0(const std::vector<double>& y,
		       const std::vector<double>& pos) const = 0;

  virtual void getGradient(const std::vector<double>& y,
			   const std::vector<double>& pos,
                           const boost::scoped_array<double>& F,
                           boost::scoped_array<double>& grad) const = 0;

  virtual double getInitLoss(const std::vector<double>& y,
			     const std::vector<double>& pos) const = 0;

  virtual double getExampleLoss(const double y, const double pos, const double f) const = 0;

  virtual void accumulateExampleLoss(const double y, const double pos, const double f) = 0;

  virtual double getReduction() const = 0;
};


class LeastSquareFun : public GbmFun {
 public:
  LeastSquareFun() : numExamples_(0), sumy_(0.0), sumy2_(0.0), l2_(0.0), sumpos2_(0.0) {
  }

  double getLeafVal(const std::vector<int>& subset,
		    const std::vector<double>& pos,
                    const boost::scoped_array<double>& y) const {

    double sum = 0;
    double wpos = 0;
    for (const auto& id : subset) {
      sum += y[id];
      wpos += pos[id] * pos[id];
    }
    return sum/wpos;
  }

  double getF0(const std::vector<double>& yvec, const std::vector<double>& pos) const {
    double sum = 0.0;
    double wpos = 0.0;
    int size = yvec.size();
    for (int i = 0; i < size; i++) {
      sum += yvec[i] * pos[i];
      wpos += pos[i] * pos[i];
    }
    return sum/wpos;
  }

  void getGradient(const std::vector<double>& y,
		   const std::vector<double>& pos,
                   const boost::scoped_array<double>& F,
                   boost::scoped_array<double>& grad) const {

    int size = y.size();

    for (int i = 0; i < size; i++) {
      grad[i] = pos[i]*(y[i] - F[i] * pos[i]);
    }
  }

  double getInitLoss(const std::vector<double>& yvec, 
		     const std::vector<double>& pos) const {
    double sumy = 0.0;
    double sumy2 = 0.0;
    double sumpos = 0.0;
    int num = yvec.size();
    for (int i = 0; i < num; i++) {
      sumy += yvec[i] * pos[i];
      sumy2 += yvec[i] * yvec[i];
      sumpos += pos[i] * pos[i];
    }

    return sumy2 - sumy * sumy/sumpos;
  }

  double getExampleLoss(const double y, const double pos, const double f) const {
    return (y - pos * f) * (y - pos * f);
  }

  void accumulateExampleLoss(const double y, const double pos, const double f) {
    sumy_ += y * pos;
   
    numExamples_ += 1;
    sumpos2_ += pos * pos;
    sumy2_ += y * y;
    l2_ += getExampleLoss(y, pos, f);
  }

  double getReduction() const {
    return 1.0 - l2_/(sumy2_ - sumy_ * sumy_/sumpos2_);
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
  double sumpos2_;
};

}
